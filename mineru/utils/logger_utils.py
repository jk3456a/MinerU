import logging
import os
import json
import uuid
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import time
from typing import Dict, Any, Optional


_LOGGER_CACHE: Dict[str, logging.Logger] = {}
_RUN_ID: str = os.environ.get("MINERU_RUN_ID", "")


def _str_to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def set_run_id(run_id: Optional[str] = None) -> str:
    """Set a global run id used in log formatting. Returns the active run id."""
    global _RUN_ID
    _RUN_ID = run_id or _RUN_ID or str(uuid.uuid4())[:8]
    # ensure env mirrors, helpful for child processes
    os.environ.setdefault("MINERU_RUN_ID", _RUN_ID)
    _install_log_record_factory()
    return _RUN_ID


def _install_log_record_factory():
    """Install a LogRecordFactory that injects run_id into all records."""
    base_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):  # type: ignore[no-redef]
        record = base_factory(*args, **kwargs)
        # inject run_id attribute for formatter
        try:
            setattr(record, "run_id", _RUN_ID or os.environ.get("MINERU_RUN_ID", "-"))
        except Exception:
            # best-effort: never break logging
            setattr(record, "run_id", "-")
        return record

    logging.setLogRecordFactory(record_factory)


def get_logger(
    name: str,
    log_dir: str | None = None,
    file_name: str | None = None,
    *,
    enable_console: bool | None = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """Create or get a configured logger.

    Features:
    - Persistent file logging with rotation
    - Macro control via env vars
    - Optional console logging

    Env vars:
    - MINERU_LOG_ENABLE: 1/0 to enable/disable logging (default 1)
    - MINERU_LOG_DIR: directory for logs (default 'logs')
    - MINERU_LOG_LEVEL: DEBUG/INFO/WARN/ERROR (default INFO)
    - MINERU_LOG_CONSOLE: 1/0 to enable console (default 1)
    - MINERU_LOG_MAX_BYTES: rotate size bytes (default 10MB)
    - MINERU_LOG_BACKUP_COUNT: number of rotated files to keep (default 7)
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    # macros
    log_enabled = _str_to_bool(os.environ.get("MINERU_LOG_ENABLE"), True)
    if log_dir is None:
        log_dir = os.environ.get("MINERU_LOG_DIR", "logs")
    if enable_console is None:
        enable_console = _str_to_bool(os.environ.get("MINERU_LOG_CONSOLE"), True)
    if level is None:
        level = os.environ.get("MINERU_LOG_LEVEL", "INFO")

    os.makedirs(log_dir, exist_ok=True)
    if file_name is None:
        file_name = f"{name}.log"
    log_path = os.path.join(log_dir, file_name)

    logger = logging.getLogger(name)
    logger.propagate = False

    # Determine level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "WARNING": logging.WARN,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger.setLevel(level_map.get(str(level).upper(), logging.INFO))

    # if disabled, attach NullHandler and return
    if not log_enabled:
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        _LOGGER_CACHE[name] = logger
        return logger

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | [%(run_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # rotating file handler for persistence
        try:
            max_bytes = int(os.environ.get("MINERU_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
        except Exception:
            max_bytes = 10 * 1024 * 1024
        try:
            backup_count = int(os.environ.get("MINERU_LOG_BACKUP_COUNT", "7"))
        except Exception:
            backup_count = 7

        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logger.level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    _LOGGER_CACHE[name] = logger
    # ensure factory installed so %(run_id)s works even if not explicitly set
    _install_log_record_factory()
    return logger


@contextmanager
def range_timer(logger: logging.Logger, tag: str):
    """A context manager that logs start/end perf_counter with duration for a tag."""
    start_ts = time.perf_counter()
    try:
        logger.info(f"[{tag}] start={start_ts:.6f}")
        yield
    finally:
        end_ts = time.perf_counter()
        duration = end_ts - start_ts
        logger.info(f"[{tag}] end={end_ts:.6f} dur={duration:.6f}s")


def _default_engineering_fingerprint() -> Dict[str, Any]:
    """Build a default fingerprint from env and runtime that reflects engineering-level config.

    Includes:
    - All env vars starting with MINERU_
    - CUDA_VISIBLE_DEVICES and PYTORCH_CUDA_ALLOC_CONF
    - Thread controls for OMP/MKL/OPENBLAS
    """
    keys = [
        *[k for k in os.environ.keys() if k.startswith("MINERU_")],
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]
    fp: Dict[str, Any] = {}
    for k in sorted(set(keys)):
        fp[k] = os.environ.get(k)
    # Basic platform cues
    try:
        fp["platform"] = os.uname().sysname
        fp["platform_release"] = os.uname().release
    except Exception:
        pass
    return fp


def log_engineering_change(
    logger: logging.Logger,
    fingerprint: Optional[Dict[str, Any]] = None,
    *,
    state_dir: Optional[str] = None,
    tag: str = "project",
) -> None:
    """Persist a fingerprint and log a timestamp if an engineering-level difference is detected.

    - fingerprint: dict to compare. If None, uses a default from env/runtime.
    - state_dir: directory to store the state file (default: <log_dir>/.state)
    - tag: differentiate multiple fingerprints.
    """
    if fingerprint is None:
        fingerprint = _default_engineering_fingerprint()

    # find a reasonable state dir co-located with logs
    log_dir = os.environ.get("MINERU_LOG_DIR", "logs")
    if state_dir is None:
        state_dir = os.path.join(log_dir, ".state")
    os.makedirs(state_dir, exist_ok=True)

    state_path = os.path.join(state_dir, f"fingerprint_{tag}.json")
    prev: Optional[Dict[str, Any]] = None
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as fi:
                prev = json.load(fi)
        except Exception:
            prev = None

    if prev != fingerprint:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.warning(
            f"ENGINEERING_DIFF_DETECTED [{tag}] at {now}. Persisting new fingerprint."
        )
        # Optionally, log a brief diff size
        try:
            prev_size = len(prev or {})
            curr_size = len(fingerprint or {})
            logger.info(f"fingerprint(prev={prev_size}, curr={curr_size})")
        except Exception:
            pass
        try:
            with open(state_path, "w") as fo:
                json.dump(fingerprint, fo, ensure_ascii=False, indent=2)
        except Exception:
            # best effort only
            pass
    else:
        logger.debug(f"No engineering diff for tag={tag}")


