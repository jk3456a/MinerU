"""
Torch Profiler 工具封装

目标：
- 通过环境变量或参数，零侵入启用/禁用 PyTorch Profiler
- 提供统一的上下文管理器与装饰器，便于在训练/推理中插入 profile
- 支持 TensorBoard 时间线、Chrome trace 导出、表格统计打印
- 兼容 NVTX（若可用）以便与 Nsight/TimeLine 联动

环境变量（均为可选）：
- MINERU_TORCH_PROF_ENABLE: 1/0，是否启用（默认 0）
- MINERU_TORCH_PROF_ACTIVITIES: "cpu,cuda" 或 "cpu"/"cuda"（默认 cpu,cuda）
- MINERU_TORCH_PROF_RECORD_SHAPES: 1/0（默认 1）
- MINERU_TORCH_PROF_PROFILE_MEMORY: 1/0（默认 1）
- MINERU_TORCH_PROF_WITH_STACK: 1/0（默认 0）
- MINERU_TORCH_PROF_SCHEDULE: 形如 "wait=1,warmup=1,active=3,repeat=1"（默认同示例）
- MINERU_TORCH_PROF_TB_DIR: TensorBoard 日志目录（默认 logs/tb_profiler/<tag>/<run_id>）
- MINERU_TORCH_PROF_EXPORT_CHROME: 路径/文件名，若设定则在退出时导出 trace.json（默认空=不导出）
- MINERU_TORCH_PROF_TABLE_ON_EXIT: 1/0 退出时打印表格（默认 1）
- MINERU_TORCH_PROF_TABLE_SORT_BY: 表格排序字段（默认 self_cuda_time_total）
- MINERU_TORCH_PROF_TABLE_ROW_LIMIT: 表格行数限制（默认 30）

使用示例：

    from mineru.utils.torch_profiler_utils import create_profiler

    with create_profiler("train") as prof:
        for step, (x, y) in enumerate(loader):
            # 你的训练逻辑
            loss = model(x).sub(y).abs().mean()
            loss.backward()
            optim.step(); optim.zero_grad(set_to_none=True)
            prof.step()  # 每个 iteration 都要调用

    # 或者标注代码段：
    from mineru.utils.torch_profiler_utils import record_function
    with record_function("data_loading"):
        batch = next(iter(loader))

    # 或者装饰器：
    from mineru.utils.torch_profiler_utils import profiled_function
    @profiled_function("forward")
    def forward(x):
        return model(x)
"""

from __future__ import annotations

import os
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, Optional


def _str_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _try_import_torch():
    try:
        import torch  # type: ignore  # noqa: WPS433
        return torch
    except Exception:
        return None


def _cuda_available(torch_mod) -> bool:
    try:
        return bool(getattr(torch_mod, "cuda", None)) and bool(torch_mod.cuda.is_available())
    except Exception:
        return False


def is_torch_profiler_enabled() -> bool:
    """是否启用 Torch Profiler（环境变量优先）。"""
    return _str_to_bool(os.getenv("MINERU_TORCH_PROF_ENABLE"), False)


@dataclass
class ProfilerConfig:
    tag: str
    record_shapes: bool
    profile_memory: bool
    with_stack: bool
    activities: tuple[str, ...]
    schedule_spec: str
    tb_dir: Optional[str]
    export_chrome: Optional[str]
    table_on_exit: bool
    table_sort_by: str
    table_row_limit: int


def _default_config(tag: str) -> ProfilerConfig:
    run_id = os.getenv("MINERU_RUN_ID") or "-"
    tb_dir_env = os.getenv("MINERU_TORCH_PROF_TB_DIR")
    tb_dir = tb_dir_env or os.path.join("logs", "tb_profiler", tag, run_id)
    try:
        row_limit = int(os.getenv("MINERU_TORCH_PROF_TABLE_ROW_LIMIT", "30"))
    except Exception:
        row_limit = 30
    return ProfilerConfig(
        tag=tag,
        record_shapes=_str_to_bool(os.getenv("MINERU_TORCH_PROF_RECORD_SHAPES"), True),
        profile_memory=_str_to_bool(os.getenv("MINERU_TORCH_PROF_PROFILE_MEMORY"), True),
        with_stack=_str_to_bool(os.getenv("MINERU_TORCH_PROF_WITH_STACK"), False),
        activities=tuple(
            a.strip().lower() for a in os.getenv("MINERU_TORCH_PROF_ACTIVITIES", "cpu,cuda").split(",") if a.strip()
        ),
        schedule_spec=os.getenv("MINERU_TORCH_PROF_SCHEDULE", "wait=1,warmup=1,active=3,repeat=1"),
        tb_dir=tb_dir,
        export_chrome=os.getenv("MINERU_TORCH_PROF_EXPORT_CHROME") or None,
        table_on_exit=_str_to_bool(os.getenv("MINERU_TORCH_PROF_TABLE_ON_EXIT"), True),
        table_sort_by=os.getenv("MINERU_TORCH_PROF_TABLE_SORT_BY", "self_cuda_time_total"),
        table_row_limit=row_limit,
    )


def _build_activities(torch_mod, activities: Iterable[str]):
    from torch.profiler import ProfilerActivity  # type: ignore

    activity_objs = []
    names = {a.strip().lower() for a in activities}
    if "cpu" in names:
        activity_objs.append(ProfilerActivity.CPU)
    if "cuda" in names and _cuda_available(torch_mod):
        activity_objs.append(ProfilerActivity.CUDA)
    if not activity_objs:
        # fallback 至 CPU
        activity_objs.append(ProfilerActivity.CPU)
    return activity_objs


def _parse_schedule(torch_profiler_mod, spec: str):
    # spec 形如 "wait=1,warmup=1,active=3,repeat=1"
    parts = {}
    for seg in (spec or "").split(","):
        if not seg:
            continue
        if "=" in seg:
            k, v = seg.split("=", 1)
            k = k.strip()
            try:
                parts[k] = int(v)
            except Exception:
                pass
    sched = getattr(torch_profiler_mod, "schedule")
    return sched(
        wait=int(parts.get("wait", 1)),
        warmup=int(parts.get("warmup", 1)),
        active=int(parts.get("active", 3)),
        repeat=int(parts.get("repeat", 1)),
    )


class _NullProfiler:
    """与 torch.profiler.profile 接口相似的空对象。"""

    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False

    def step(self) -> None:
        pass

    def key_averages(self):
        return self

    def table(self, *args, **kwargs) -> str:
        return ""

    def export_chrome_trace(self, *args, **kwargs) -> None:
        pass


class ManagedProfiler:
    """托管的 Profiler：统一 step/打印/导出行为。"""

    def __init__(self, config: ProfilerConfig, enabled: bool):
        self._config = config
        self._enabled = enabled
        self._torch = None
        self._prof = _NullProfiler()
        self._tb_ready = None

    def __enter__(self) -> "ManagedProfiler":
        if not self._enabled:
            return self
        self._torch = _try_import_torch()
        if self._torch is None:
            return self
        try:
            from torch import profiler as torch_profiler  # type: ignore

            activities = _build_activities(self._torch, self._config.activities)
            schedule = _parse_schedule(torch_profiler, self._config.schedule_spec)

            on_trace_ready = None
            if self._config.tb_dir:
                os.makedirs(self._config.tb_dir, exist_ok=True)
                from torch.profiler import tensorboard_trace_handler  # type: ignore

                on_trace_ready = tensorboard_trace_handler(self._config.tb_dir)
                self._tb_ready = True

            self._prof = torch_profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=on_trace_ready,
                record_shapes=self._config.record_shapes,
                profile_memory=self._config.profile_memory,
                with_stack=self._config.with_stack,
            )
            self._prof.__enter__()
        except Exception:
            # 出错时静默降级
            self._prof = _NullProfiler()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._enabled and hasattr(self._prof, "__exit__"):
                self._prof.__exit__(exc_type, exc, tb)
        finally:
            try:
                if self._enabled and self._config.export_chrome and hasattr(self._prof, "export_chrome_trace"):
                    # 确保目录存在
                    export_path = self._config.export_chrome
                    export_dir = os.path.dirname(export_path) or "."
                    os.makedirs(export_dir, exist_ok=True)
                    self._prof.export_chrome_trace(export_path)
            except Exception:
                pass
            try:
                if self._enabled and self._config.table_on_exit:
                    # 打印表格到 stdout
                    table_str = self.table(
                        sort_by=self._config.table_sort_by, row_limit=self._config.table_row_limit
                    )
                    if table_str:
                        print(table_str)
            except Exception:
                pass
        return False

    # 公共 API
    def step(self) -> None:
        try:
            self._prof.step()
        except Exception:
            pass

    def table(self, sort_by: Optional[str] = None, row_limit: Optional[int] = None) -> str:
        try:
            sort_by = sort_by or self._config.table_sort_by
            row_limit = row_limit or self._config.table_row_limit
            return self._prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
        except Exception:
            return ""

    @property
    def prof(self):  # noqa: D401
        """返回底层 torch.profiler 对象或空对象。"""
        return self._prof


def create_profiler(
    tag: str,
    *,
    enabled: Optional[bool] = None,
    tb_log_dir: Optional[str] = None,
    schedule_spec: Optional[str] = None,
    record_shapes: Optional[bool] = None,
    profile_memory: Optional[bool] = None,
    with_stack: Optional[bool] = None,
    activities: Optional[Iterable[str]] = None,
    export_chrome: Optional[str] = None,
    table_on_exit: Optional[bool] = None,
    table_sort_by: Optional[str] = None,
    table_row_limit: Optional[int] = None,
) -> ManagedProfiler:
    """创建托管 Profiler。

    - 当 disabled 时返回空实现，调用安全无副作用
    - 建议在训练循环外包一层 with，并在每个 iteration 调用 step()
    """
    cfg = _default_config(tag)
    if tb_log_dir is not None:
        cfg.tb_dir = tb_log_dir
    if schedule_spec is not None:
        cfg.schedule_spec = schedule_spec
    if record_shapes is not None:
        cfg.record_shapes = bool(record_shapes)
    if profile_memory is not None:
        cfg.profile_memory = bool(profile_memory)
    if with_stack is not None:
        cfg.with_stack = bool(with_stack)
    if activities is not None:
        cfg.activities = tuple(a.strip().lower() for a in activities)
    if export_chrome is not None:
        cfg.export_chrome = export_chrome
    if table_on_exit is not None:
        cfg.table_on_exit = bool(table_on_exit)
    if table_sort_by is not None:
        cfg.table_sort_by = table_sort_by
    if table_row_limit is not None:
        cfg.table_row_limit = int(table_row_limit)

    enabled_final = is_torch_profiler_enabled() if enabled is None else bool(enabled)
    return ManagedProfiler(cfg, enabled_final)


@contextmanager
def record_function(name: str):
    """兼容 torch.profiler.record_function 与 NVTX 的范围标注。"""
    # NVTX（可选）
    try:
        from mineru.utils.nvtx_utils import nvtx_range  # noqa: WPS433
    except Exception:  # noqa: WPS429
        nvtx_range = None  # type: ignore

    # Torch profiler record_function（可选）
    torch_mod = _try_import_torch()
    prof_rec_ctx = None
    if torch_mod is not None:
        try:
            from torch.profiler import record_function as torch_record_function  # type: ignore

            prof_rec_ctx = torch_record_function(name)
        except Exception:
            prof_rec_ctx = None

    # 组合作为嵌套上下文
    if nvtx_range is not None and prof_rec_ctx is not None:
        with nvtx_range(name):
            with prof_rec_ctx:
                yield
        return

    if nvtx_range is not None:
        with nvtx_range(name):
            yield
        return

    if prof_rec_ctx is not None:
        with prof_rec_ctx:
            yield
        return

    # 两者都不可用时直接透传
    yield


def profiled_function(name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """函数装饰器：进入函数体时自动标注 record_function/NVTX。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tag = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with record_function(tag):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def print_table(prof_or_manager: Any, *, sort_by: Optional[str] = None, row_limit: Optional[int] = None) -> str:
    """打印（并返回）Profiler 表格，若不可用则返回空串。"""
    try:
        if isinstance(prof_or_manager, ManagedProfiler):
            return prof_or_manager.table(sort_by=sort_by, row_limit=row_limit)
        # 直接是 torch profiler 对象
        sort_by = sort_by or os.getenv("MINERU_TORCH_PROF_TABLE_SORT_BY", "self_cuda_time_total")
        try:
            row_limit = int(row_limit or os.getenv("MINERU_TORCH_PROF_TABLE_ROW_LIMIT", "30"))
        except Exception:
            row_limit = 30
        table_str = prof_or_manager.key_averages().table(sort_by=sort_by, row_limit=row_limit)
        print(table_str)
        return table_str
    except Exception:
        return ""


