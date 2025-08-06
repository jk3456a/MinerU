"""
无锁高性能异步GPU-CPU内存传输管理器
专门为OCR场景优化，使用预分配资源池和环形缓冲区实现无锁设计
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from collections import defaultdict


class LockFreeAsyncTransferManager:
    """无锁异步传输管理器 - 单生产者模型"""
    
    def __init__(self, device='cuda', max_inflight=4, enable_debug=False):
        """
        初始化无锁异步传输管理器
        
        Args:
            device: CUDA设备
            max_inflight: 最大并发传输数
            enable_debug: 是否启用调试信息
        """
        self.device = device
        self.max_inflight = max_inflight
        self.enable_debug = enable_debug or os.getenv('MINERU_ASYNC_DEBUG', 'false').lower() == 'true'
        
        # 检查CUDA可用性
        self.enabled = torch.cuda.is_available()
        if not self.enabled:
            print("CUDA not available, async transfer disabled")
            return
            
        # 预分配资源池
        self.transfer_streams = [torch.cuda.Stream(device=device) for _ in range(max_inflight)]
        self.gpu_events = [torch.cuda.Event() for _ in range(max_inflight)]
        
        # 无锁环形缓冲区
        self.ring_buffer_size = max_inflight * 2
        self.pending_transfers = [None] * self.ring_buffer_size
        self.write_idx = 0  # 只有生产者写
        self.read_idx = 0   # 只有消费者读
        
        # 预分配的pinned memory池
        self.buffer_pools = defaultdict(list)  # shape -> [buffers]
        self.buffer_pool_size = max_inflight * 2
        
        # 性能统计
        self.stats = {
            'total_transfers': 0,
            'total_transfer_time': 0.0,
            'buffer_hits': 0,
            'buffer_misses': 0,
            'completed_transfers': 0
        }
        
        # 预热系统
        if self.enabled:
            self._prewarm_system()
    
    def _prewarm_system(self):
        """系统预热 - 预分配资源和初始化CUDA上下文"""
        if self.enable_debug:
            print("Prewarming async transfer system...")
            
        # 预热CUDA上下文
        dummy = torch.randn(1, 1, device=self.device)
        dummy.cpu()
        
        # 预热所有CUDA流
        for stream in self.transfer_streams:
            with torch.cuda.stream(stream):
                dummy_op = torch.randn(1, 1, device=self.device)
                dummy_op.cpu()
        
        # 预分配常见的OCR buffer shapes
        common_shapes = [
            (1, 5531),      # 单个样本输出
            (4, 5531),      # 小批量
            (8, 5531),      # 中等批量
            (16, 5531),     # 标准批量
            (32, 5531),     # 大批量
        ]
        
        for shape in common_shapes:
            self._preallocate_buffers_for_shape(shape, torch.float32)
        
        # 同步所有操作
        torch.cuda.synchronize()
        
        if self.enable_debug:
            print(f"System prewarmed with {len(self.buffer_pools)} buffer shapes")
    
    def _preallocate_buffers_for_shape(self, shape: tuple, dtype: torch.dtype):
        """为特定shape预分配buffer池"""
        key = (shape, dtype)
        if key not in self.buffer_pools:
            self.buffer_pools[key] = []
            
        # 预分配到池大小
        while len(self.buffer_pools[key]) < self.buffer_pool_size:
            try:
                buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
                self.buffer_pools[key].append(buffer)
            except RuntimeError as e:
                if self.enable_debug:
                    print(f"Failed to preallocate pinned memory for {shape}: {e}")
                break
    
    def _get_buffer_from_pool(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """从池中获取buffer（O(1)时间复杂度）"""
        key = (shape, dtype)
        
        if key in self.buffer_pools and self.buffer_pools[key]:
            self.stats['buffer_hits'] += 1
            return self.buffer_pools[key].pop()
        
        # 未命中，需要分配新buffer
        self.stats['buffer_misses'] += 1
        try:
            buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError:
            # 如果pinned memory分配失败，使用普通内存
            buffer = torch.empty(shape, dtype=dtype)
            
        return buffer
    
    def _return_buffer_to_pool(self, buffer: torch.Tensor, shape: tuple, dtype: torch.dtype):
        """归还buffer到池中"""
        key = (shape, dtype)
        
        # 只保留不超过池大小的buffer
        if len(self.buffer_pools[key]) < self.buffer_pool_size:
            self.buffer_pools[key].append(buffer)
    
    def async_transfer(self, gpu_tensor: torch.Tensor) -> Optional[int]:
        """
        启动异步GPU到CPU的传输
        
        Args:
            gpu_tensor: GPU上的tensor
            
        Returns:
            transfer_id: 传输ID，用于后续查询结果
            None: 如果CUDA不可用或传输失败
        """
        if not self.enabled:
            return None
            
        # 获取当前slot（无锁操作）
        slot_id = self.write_idx % self.max_inflight
        stream = self.transfer_streams[slot_id]
        event = self.gpu_events[slot_id]
        
        # 记录当前计算流的完成点
        event.record(torch.cuda.current_stream(device=self.device))
        
        # 获取预分配的buffer
        shape = gpu_tensor.shape
        dtype = gpu_tensor.dtype
        cpu_buffer = self._get_buffer_from_pool(shape, dtype)
        
        # 在传输流上执行异步拷贝
        with torch.cuda.stream(stream):
            # 等待计算完成
            stream.wait_event(event)
            # 执行异步拷贝
            cpu_buffer.copy_(gpu_tensor, non_blocking=True)
        
        # 存储到环形缓冲区
        ring_idx = self.write_idx % self.ring_buffer_size
        self.pending_transfers[ring_idx] = {
            'cpu_buffer': cpu_buffer,
            'stream': stream,
            'shape': shape,
            'dtype': dtype,
            'slot_id': slot_id
        }
        
        self.write_idx += 1
        self.stats['total_transfers'] += 1
        
        return ring_idx
    
    def poll_completed_transfers(self) -> List[Tuple[int, np.ndarray]]:
        """
        非阻塞轮询已完成的传输
        
        Returns:
            已完成的传输列表 [(transfer_id, numpy_array), ...]
        """
        if not self.enabled:
            return []
            
        completed = []
        
        # 检查从read_idx开始的传输
        while self.read_idx < self.write_idx:
            ring_idx = self.read_idx % self.ring_buffer_size
            transfer = self.pending_transfers[ring_idx]
            
            if transfer is None:
                break
                
            # 非阻塞检查是否完成
            if transfer['stream'].query():
                # 转换为numpy（零拷贝）
                cpu_buffer = transfer['cpu_buffer']
                result = cpu_buffer.detach().numpy()
                
                # 回收buffer到池中
                self._return_buffer_to_pool(
                    cpu_buffer, 
                    transfer['shape'], 
                    transfer['dtype']
                )
                
                completed.append((ring_idx, result))
                
                # 清理并前进读指针
                self.pending_transfers[ring_idx] = None
                self.read_idx += 1
                self.stats['completed_transfers'] += 1
            else:
                # FIFO顺序，如果当前未完成，后续的也不会完成
                break
        
        return completed
    
    def wait_all_transfers(self) -> List[Tuple[int, np.ndarray]]:
        """
        等待所有未完成的传输（阻塞）
        
        Returns:
            所有传输结果
        """
        if not self.enabled:
            return []
            
        results = []
        
        # 首先尝试非阻塞收集
        results.extend(self.poll_completed_transfers())
        
        # 等待剩余的传输
        while self.read_idx < self.write_idx:
            ring_idx = self.read_idx % self.ring_buffer_size
            transfer = self.pending_transfers[ring_idx]
            
            if transfer is not None:
                # 阻塞等待
                transfer['stream'].synchronize()
                
                # 获取结果
                cpu_buffer = transfer['cpu_buffer']
                result = cpu_buffer.detach().numpy()
                
                # 回收buffer
                self._return_buffer_to_pool(
                    cpu_buffer,
                    transfer['shape'],
                    transfer['dtype']
                )
                
                results.append((ring_idx, result))
                
                # 清理并前进
                self.pending_transfers[ring_idx] = None
                self.read_idx += 1
                self.stats['completed_transfers'] += 1
        
        return results
    
    def get_stats(self) -> Dict:
        """获取性能统计信息"""
        stats = self.stats.copy()
        stats['pending_transfers'] = self.write_idx - self.read_idx
        stats['buffer_pool_sizes'] = {
            str(k): len(v) for k, v in self.buffer_pools.items()
        }
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_transfers': 0,
            'total_transfer_time': 0.0,
            'buffer_hits': 0,
            'buffer_misses': 0,
            'completed_transfers': 0
        }


class BatchAsyncProcessor:
    """批量异步处理器 - 管理批次结果"""
    
    def __init__(self, async_manager: LockFreeAsyncTransferManager):
        self.async_manager = async_manager
        self.batch_queue = []  # [(transfer_id, batch_info), ...]
        
    def add_batch(self, gpu_tensor: torch.Tensor, batch_info: Dict) -> Optional[int]:
        """
        添加一个批次进行异步传输
        
        Args:
            gpu_tensor: GPU tensor
            batch_info: 批次信息（indices, postprocess_func等）
            
        Returns:
            transfer_id: 传输ID
        """
        transfer_id = self.async_manager.async_transfer(gpu_tensor)
        if transfer_id is not None:
            self.batch_queue.append((transfer_id, batch_info))
        return transfer_id
    
    def collect_ready_batches(self) -> List[Tuple[Dict, np.ndarray]]:
        """
        收集已完成的批次结果
        
        Returns:
            [(batch_info, numpy_array), ...]
        """
        if not self.batch_queue:
            return []
            
        # 获取已完成的传输
        completed_transfers = dict(self.async_manager.poll_completed_transfers())
        
        results = []
        remaining_queue = []
        
        for transfer_id, batch_info in self.batch_queue:
            if transfer_id in completed_transfers:
                results.append((batch_info, completed_transfers[transfer_id]))
            else:
                remaining_queue.append((transfer_id, batch_info))
        
        self.batch_queue = remaining_queue
        return results
    
    def wait_all_batches(self) -> List[Tuple[Dict, np.ndarray]]:
        """等待所有批次完成"""
        # 等待所有传输
        all_transfers = dict(self.async_manager.wait_all_transfers())
        
        results = []
        for transfer_id, batch_info in self.batch_queue:
            if transfer_id in all_transfers:
                results.append((batch_info, all_transfers[transfer_id]))
        
        self.batch_queue = []
        return results

