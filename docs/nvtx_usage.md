# NVTX性能分析工具使用指南

本文档说明如何使用MinerU项目中的NVTX性能分析宏工具。

## 概述

NVTX (NVIDIA Tools Extension) 是NVIDIA提供的性能分析工具，可以帮助开发者分析CUDA程序的性能瓶颈。本项目提供了一个统一的宏工具 `nvtx_utils.py`，让您可以方便地开启或关闭NVTX功能，而无需在代码中来回注释。

## 快速开始

### 1. 启用NVTX

通过环境变量控制NVTX功能：

```bash
# 启用NVTX
export MINERU_NVTX_ENABLE=true

# 运行程序
python examples/ocr_pdf_with_mineru.py
```

### 2. 使用nsys进行性能分析

```bash
# 设置环境变量
export MINERU_NVTX_ENABLE=true

# 使用nsys分析
nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    -o mineru_profile \
    python examples/ocr_pdf_with_mineru.py

# 查看结果
nsys-ui mineru_profile.nsys-rep
```

### 3. 使用ncu进行详细分析

```bash
# 设置环境变量
export MINERU_NVTX_ENABLE=true

# 使用ncu分析
ncu --set full --nvtx \
    --target-processes all \
    --replay-mode kernel \
    -o mineru_detailed \
    python examples/ocr_pdf_with_mineru.py

# 查看结果
ncu-ui mineru_detailed.ncu-rep
```

## 环境变量配置

| 环境变量 | 可选值 | 默认值 | 说明 |
|---------|--------|--------|------|
| `MINERU_NVTX_ENABLE` | `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off` | `false` | 控制NVTX功能开关 |

## 编程接口

### 导入模块

```python
from mineru.utils.nvtx_utils import (
    nvtx_range_push, nvtx_range_pop, nvtx_range, nvtx_annotate,
    enable_nvtx, disable_nvtx, is_nvtx_enabled, get_nvtx_status
)
```

### 使用方法

#### 1. 上下文管理器（推荐）

```python
from mineru.utils.nvtx_utils import nvtx_range

with nvtx_range("operation_name"):
    # 你的代码
    pass

# 支持嵌套
with nvtx_range("outer_operation"):
    with nvtx_range("inner_operation"):
        # 你的代码
        pass
```

#### 2. 手动调用

```python
from mineru.utils.nvtx_utils import nvtx_range_push, nvtx_range_pop

nvtx_range_push("operation_name")
try:
    # 你的代码
    pass
finally:
    nvtx_range_pop()
```

#### 3. 装饰器

```python
from mineru.utils.nvtx_utils import nvtx_annotate

@nvtx_annotate("custom_function_name")
def my_function():
    pass

@nvtx_annotate()  # 使用函数名作为范围名
def another_function():
    pass
```

#### 4. 动态控制

```python
from mineru.utils.nvtx_utils import enable_nvtx, disable_nvtx, is_nvtx_enabled

# 强制启用（忽略环境变量）
enable_nvtx()

# 强制禁用
disable_nvtx()

# 检查状态
if is_nvtx_enabled():
    print("NVTX已启用")
```

#### 5. 状态查询

```python
from mineru.utils.nvtx_utils import get_nvtx_status

status = get_nvtx_status()
print(f"状态信息: {status}")
# 输出: {'enabled': True, 'nvtx_available': True, 'env_value': 'true', 'effective_enabled': True}
```

## 最佳实践

### 1. 性能敏感代码

对于性能敏感的代码段，使用有意义的标签名：

```python
with nvtx_range("batch_inference"):
    # 批量推理代码
    pass

with nvtx_range(f"ocr_batch_{batch_size}_images"):
    # OCR批处理代码
    pass
```

### 2. 分层分析

使用嵌套的NVTX范围来分析不同层级的性能：

```python
with nvtx_range("complete_pipeline"):
    with nvtx_range("data_loading"):
        # 数据加载
        pass
    
    with nvtx_range("preprocessing"):
        # 预处理
        pass
    
    with nvtx_range("model_inference"):
        with nvtx_range("forward_pass"):
            # 前向传播
            pass
        with nvtx_range("postprocessing"):
            # 后处理
            pass
```

### 3. 批处理优化

对批处理操作使用描述性标签：

```python
with nvtx_range(f"det_batch_{batch_size}_images_{target_h}x{target_w}"):
    batch_results = detector.batch_predict(images, batch_size)
```

## 性能影响

- **NVTX关闭时**: 几乎零性能开销，函数调用直接返回
- **NVTX开启时**: 每次调用有微小开销（通常 < 1μs）
- **自动检测**: 如果PyTorch或CUDA不可用，自动禁用

## 故障排除

### 1. NVTX不工作

检查状态：
```python
from mineru.utils.nvtx_utils import get_nvtx_status
print(get_nvtx_status())
```

常见问题：
- 环境变量未设置: `export MINERU_NVTX_ENABLE=true`
- PyTorch未安装CUDA支持
- CUDA不可用

### 2. nsys/ncu无法显示NVTX标记

确保：
- 使用了 `-t nvtx` 或 `--nvtx` 参数
- 程序正确启用了NVTX
- 运行在支持CUDA的环境中

### 3. 性能测试

运行示例程序验证功能：
```bash
export MINERU_NVTX_ENABLE=true
python examples/nvtx_usage_example.py
```

## 参考资料

- [NVIDIA nsys 文档](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA ncu 文档](https://docs.nvidia.com/nsight-compute/)
- [PyTorch NVTX 文档](https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile)
