"""
NVTX 工具（生产环境禁用版）
本文件将所有 NVTX 接口实现为无操作（no-op），确保生产环境中不会触发任何 NVTX 标记。

保留函数签名以兼容现有导入：
- nvtx_range_push / nvtx_range_pop
- nvtx_range 上下文管理器
- nvtx_annotate 装饰器
- enable_nvtx / disable_nvtx / is_nvtx_enabled / get_nvtx_status
"""

from contextlib import contextmanager
from typing import Optional, Any, Dict
import functools


def nvtx_range_push(name: str, domain: Optional[str] = None) -> None:
    return None


def nvtx_range_pop() -> None:
    return None


@contextmanager
def nvtx_range(name: str, domain: Optional[str] = None):
    yield


def nvtx_annotate(name: Optional[str] = None, domain: Optional[str] = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def enable_nvtx():
    return None


def disable_nvtx():
    return None


def is_nvtx_enabled() -> bool:
    return False


def get_nvtx_status() -> Dict[str, Any]:
    return {
        "enabled": False,
        "nvtx_available": False,
        "env_value": "false",
        "effective_enabled": False,
    }


nvtx_push = nvtx_range_push
nvtx_pop = nvtx_range_pop
