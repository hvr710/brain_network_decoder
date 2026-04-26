import json
import math
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


UNC_TO_MNT = {
    "dataset1": ("\\\\10.16.57.94\\dataset1", "/mnt/dataset1"),
    "dataset3": ("\\\\10.16.93.90\\dataset3", "/mnt/dataset3"),
    "dataset4": ("\\\\10.20.33.82\\dataset4", "/mnt/dataset4"),
}

A800_DATA_ROOTS = {
    "dataset1": "TABLE3_DATASET1_ROOT",
    "dataset3": "TABLE3_DATASET3_ROOT",
    "dataset4": "TABLE3_DATASET4_ROOT",
}

A800_DEFAULT_ROOTS = {
    "dataset1": "/vePFS-0x0d/nzh/data/dataset1",
    "dataset3": "/vePFS-0x0d/nzh/data/dataset3",
    "dataset4": "/vePFS-0x0d/nzh/data/dataset4",
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_task_config(config_path: str, task_id: str) -> Dict[str, Any]:
    config = load_yaml(config_path)
    defaults = config.get("defaults", {})
    tasks = config.get("tasks", {})
    if task_id not in tasks:
        raise KeyError(f"Unknown task_id: {task_id}")
    merged = deep_merge(defaults, tasks[task_id])
    merged["task_id"] = task_id
    return merged


def list_task_ids(config_path: str) -> List[str]:
    config = load_yaml(config_path)
    return list(config.get("tasks", {}).keys())


def canonical_label_key(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def resolve_path(path: str) -> str:
    if not path:
        return path
    normalized = path.replace("\\\\", "\\\\").replace("/", os.sep)
    if os.path.exists(normalized):
        return normalized

    for dataset_name, (unc_root, mnt_root) in UNC_TO_MNT.items():
        local_root = os.environ.get(A800_DATA_ROOTS[dataset_name], A800_DEFAULT_ROOTS[dataset_name])
        if path.startswith(unc_root):
            local_candidate = path.replace(unc_root, local_root).replace("\\", "/")
            if os.path.exists(local_candidate):
                return local_candidate
            posix_candidate = path.replace(unc_root, mnt_root).replace("\\", "/")
            if os.path.exists(posix_candidate):
                return posix_candidate
        if path.startswith(mnt_root):
            local_candidate = path.replace(mnt_root, local_root).replace("/", os.sep)
            if os.path.exists(local_candidate):
                return local_candidate
            unc_candidate = path.replace(mnt_root, unc_root).replace("/", "\\")
            if os.path.exists(unc_candidate):
                return unc_candidate
    return path


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def task_dir_name(task_id: str, task_dir_name_override: Optional[str] = None, task_dir_suffix: Optional[str] = None) -> str:
    if task_dir_name_override:
        return task_dir_name_override
    if task_dir_suffix:
        return f"{task_id}_{task_dir_suffix}"
    return task_id


def task_run_dir(
    outputs_root: str,
    task_id: str,
    fold: int,
    seed: int,
    *,
    task_dir_name_override: Optional[str] = None,
    task_dir_suffix: Optional[str] = None,
) -> str:
    return str(
        Path(outputs_root)
        / task_dir_name(task_id, task_dir_name_override, task_dir_suffix)
        / f"fold{fold}"
        / f"seed{seed}"
    )


def update_run_state(run_dir: str, state: str, **extra: Any) -> None:
    payload = {
        "state": state,
        "updated_at": now_ts(),
        **extra,
    }
    write_json(str(Path(run_dir) / "run_state.json"), payload)


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        pass

    if platform.system().lower().startswith("win"):
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    return False


def query_gpus() -> List[Dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []

    gpus = []
    for raw_line in result.stdout.strip().splitlines():
        if not raw_line.strip():
            continue
        index_str, memory_free_str, util_str = [part.strip() for part in raw_line.split(",")]
        gpus.append(
            {
                "index": int(index_str),
                "memory_free_mb": int(memory_free_str),
                "utilization_gpu": int(util_str),
            }
        )
    return gpus


def lock_dir(outputs_root: str) -> str:
    return ensure_dir(str(Path(outputs_root) / "_gpu_locks"))


def gpu_lock_path(outputs_root: str, gpu_index: int) -> str:
    return str(Path(lock_dir(outputs_root)) / f"gpu{gpu_index}.lock")


def read_lock(lock_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(lock_path):
        return None
    try:
        return read_json(lock_path)
    except (json.JSONDecodeError, OSError):
        return None


def clear_stale_lock(lock_path: str) -> bool:
    payload = read_lock(lock_path)
    if payload is None:
        return False
    pid = int(payload.get("pid", -1))
    if is_process_alive(pid):
        return False
    try:
        os.remove(lock_path)
    except OSError:
        return False
    return True


def acquire_gpu_lock(
    outputs_root: str,
    gpu_index: int,
    *,
    pid: int,
    task_id: str,
    fold: int,
    seed: int,
) -> Tuple[bool, str]:
    path = gpu_lock_path(outputs_root, gpu_index)
    clear_stale_lock(path)
    payload = {
        "pid": pid,
        "task_id": task_id,
        "fold": fold,
        "seed": seed,
        "gpu_index": gpu_index,
        "start_time": now_ts(),
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return True, path
    except FileExistsError:
        return False, path


def release_gpu_lock(lock_path: Optional[str]) -> None:
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError:
        pass


def estimate_required_mem_mb(outputs_root: str, task_id: str, default_mb: int = 32000) -> int:
    task_dir = Path(outputs_root) / task_id
    if not task_dir.exists():
        return default_mb

    peaks: List[float] = []
    for metrics_path in task_dir.rglob("best_metrics.json"):
        try:
            payload = read_json(str(metrics_path))
        except OSError:
            continue
        peak = payload.get("peak_gpu_mem_mb")
        if isinstance(peak, (int, float)) and peak > 0:
            peaks.append(float(peak))

    if not peaks:
        return default_mb
    historical_peak = max(peaks)
    return max(default_mb, int(math.ceil(historical_peak * 1.05)))


def choose_gpu(
    outputs_root: str,
    required_mem_mb: int,
    util_threshold: int = 30,
) -> Optional[Dict[str, int]]:
    candidates = []
    for gpu in query_gpus():
        if gpu["utilization_gpu"] > util_threshold:
            continue
        if gpu["memory_free_mb"] < required_mem_mb:
            continue
        lock_path = gpu_lock_path(outputs_root, gpu["index"])
        clear_stale_lock(lock_path)
        if os.path.exists(lock_path):
            continue
        candidates.append(gpu)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item["memory_free_mb"], reverse=True)
    return candidates[0]


@dataclass
class LockGuard:
    lock_path: Optional[str] = None

    def release(self) -> None:
        release_gpu_lock(self.lock_path)
        self.lock_path = None

    def install_signal_handlers(self) -> None:
        def _cleanup_and_raise(signum: int, _frame: Any) -> None:
            self.release()
            raise KeyboardInterrupt(f"Interrupted with signal {signum}")

        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                signal.signal(sig, _cleanup_and_raise)

