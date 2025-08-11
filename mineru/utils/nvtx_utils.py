"""
NVTX性能分析工具宏
统一管理NVTX功能的开启和关闭，避免在代码中来回注释

使用方法：
1. 通过环境变量控制：
   export MINERU_NVTX_ENABLE=true  # 开启NVTX
   export MINERU_NVTX_ENABLE=false # 关闭NVTX

2. 在代码中使用：
   from mineru.utils.nvtx_utils import nvtx_range_push, nvtx_range_pop
   
   nvtx_range_push("operation_name")
   # 你的代码
   nvtx_range_pop()

3. 使用上下文管理器（推荐）：
   from mineru.utils.nvtx_utils import nvtx_range
   
   with nvtx_range("operation_name"):
       # 你的代码
"""

import os
import functools
from contextlib import contextmanager
from typing import Optional, Any


class NVTXManager:
    """NVTX管理器"""
    
    def __init__(self):
        self._enabled = None
        self._torch_available = None
        self._cuda_nvtx_available = None
        self._cpu_nvtx_available = None
        
    @property
    def enabled(self) -> bool:
        """检查NVTX是否启用"""
        if self._enabled is None:
            # 从环境变量读取配置，默认为False
            env_value = os.getenv('MINERU_NVTX_ENABLE', 'false').lower()
            self._enabled = env_value in ('true', '1', 'yes', 'on')
        return self._enabled
    
    def _detect_cuda_nvtx(self) -> bool:
        if self._cuda_nvtx_available is None:
            try:
                import torch  # noqa: WPS433
                self._torch_available = True
                self._cuda_nvtx_available = (
                    hasattr(torch, "cuda") and hasattr(torch.cuda, "nvtx") and torch.cuda.is_available()
                )
            except Exception:
                self._torch_available = False
                self._cuda_nvtx_available = False
        return bool(self._cuda_nvtx_available)

    def _detect_cpu_nvtx(self) -> bool:
        if self._cpu_nvtx_available is None:
            try:
                import nvtx  # type: ignore  # noqa: F401, WPS433
                self._cpu_nvtx_available = True
            except Exception:
                self._cpu_nvtx_available = False
        return bool(self._cpu_nvtx_available)

    def backend(self) -> str:
        """返回实际使用的后端: 'cuda' | 'cpu' | 'none'"""
        if not self.enabled:
            return "none"
        backend_pref = os.getenv('MINERU_NVTX_BACKEND', 'auto').lower()
        if backend_pref == 'cuda':
            return 'cuda' if self._detect_cuda_nvtx() else 'none'
        if backend_pref == 'cpu':
            return 'cpu' if self._detect_cpu_nvtx() else 'none'
        # auto: cuda 优先，其次 cpu
        if self._detect_cuda_nvtx():
            return 'cuda'
        if self._detect_cpu_nvtx():
            return 'cpu'
        return 'none'
    
    def force_enable(self):
        """强制启用NVTX（用于调试）"""
        self._enabled = True
        
    def force_disable(self):
        """强制禁用NVTX"""
        self._enabled = False
        
    def reset(self):
        """重置状态，重新从环境变量读取配置"""
        self._enabled = None


# 全局管理器实例
_nvtx_manager = NVTXManager()


def nvtx_range_push(name: str, domain: Optional[str] = None) -> None:
    """
    开始NVTX范围标记
    
    Args:
        name: 范围名称
        domain: 域名（可选）
    """
    backend = _nvtx_manager.backend()
    if backend == 'none':
        return
    try:
        if backend == 'cuda':
            import torch  # noqa: WPS433
            if domain:
                torch.cuda.nvtx.range_push(name, domain)
            else:
                torch.cuda.nvtx.range_push(name)
        elif backend == 'cpu':
            import nvtx  # type: ignore  # noqa: WPS433
            # python-nvtx 不支持 domain 参数，忽略之
            nvtx.range_push(name)
    except Exception:
        pass


def nvtx_range_pop() -> None:
    """结束NVTX范围标记"""
    backend = _nvtx_manager.backend()
    if backend == 'none':
        return
    try:
        if backend == 'cuda':
            import torch  # noqa: WPS433
            torch.cuda.nvtx.range_pop()
        elif backend == 'cpu':
            import nvtx  # type: ignore  # noqa: WPS433
            nvtx.range_pop()
    except Exception:
        pass


@contextmanager
def nvtx_range(name: str, domain: Optional[str] = None):
    """
    NVTX范围上下文管理器（推荐使用）
    
    Args:
        name: 范围名称
        domain: 域名（可选）
        
    Example:
        with nvtx_range("my_operation"):
            # 你的代码
            pass
    """
    nvtx_range_push(name, domain)
    try:
        yield
    finally:
        nvtx_range_pop()


def nvtx_annotate(name: Optional[str] = None, domain: Optional[str] = None):
    """
    NVTX函数装饰器
    
    Args:
        name: 范围名称，如果为None则使用函数名
        domain: 域名（可选）
        
    Example:
        @nvtx_annotate("my_function")
        def my_function():
            pass
            
        @nvtx_annotate()  # 使用函数名作为范围名
        def another_function():
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            range_name = name if name is not None else func.__name__
            with nvtx_range(range_name, domain):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# 便利函数
def enable_nvtx():
    """启用NVTX"""
    _nvtx_manager.force_enable()


def disable_nvtx():
    """禁用NVTX"""
    _nvtx_manager.force_disable()


def is_nvtx_enabled() -> bool:
    """检查NVTX是否启用"""
    return _nvtx_manager.enabled and (_nvtx_manager.backend() in ('cuda', 'cpu'))


def get_nvtx_status() -> dict:
    """获取NVTX状态信息"""
    return {
        "enabled": _nvtx_manager.enabled,
        "backend": _nvtx_manager.backend(),
        "env_value": os.getenv('MINERU_NVTX_ENABLE', 'false'),
        "effective_enabled": is_nvtx_enabled()
    }


# 兼容性别名
nvtx_push = nvtx_range_push
nvtx_pop = nvtx_range_pop
