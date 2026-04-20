from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from table3_utils import deep_merge, ensure_dir, load_yaml, now_ts, write_json


DEFAULT_PROBE_CONFIG: Dict[str, Any] = {
    "num_runs": 5,
    "base_seed": 23,
    "seed_stride": 1,
    "base_split_index": 1,
    "modes": ["linear_probe", "mlp_probe"],
    "mlp_probe": {
        "hidden_dim": 64,
        "dropout": 0.1,
        "num_layers": 2,
        "input_norm": "layernorm",
        "batch_size": 128,
        "epochs": 200,
        "max_patience": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
}


def load_probe_config(path: str) -> Dict[str, Any]:
    payload = load_yaml(path) if path else {}
    payload = payload or {}
    return deep_merge(DEFAULT_PROBE_CONFIG, payload)


def resolve_base_split_index(task_cfg: Mapping[str, Any], requested_index: int) -> int:
    fold_ids = list(task_cfg.get("fold_ids") or [1])
    if len(fold_ids) <= 1:
        return int(fold_ids[0])
    if requested_index not in fold_ids:
        raise ValueError(
            f"Task {task_cfg.get('task_id', '<unknown>')} does not contain base split {requested_index}. "
            f"Available fold_ids={fold_ids}"
        )
    return int(requested_index)


def base_split_tag(base_split_index: int) -> str:
    return f"base_split{int(base_split_index)}"


def feature_cache_dir(output_root: str, task_id: str, base_split_index: int) -> Path:
    return Path(output_root) / "feature_cache" / task_id / base_split_tag(base_split_index)


def feature_bundle_path(output_root: str, task_id: str, base_split_index: int) -> Path:
    return feature_cache_dir(output_root, task_id, base_split_index) / "feature_bundle.npz"


def feature_meta_path(output_root: str, task_id: str, base_split_index: int) -> Path:
    return feature_cache_dir(output_root, task_id, base_split_index) / "feature_meta.json"


def feature_log_path(output_root: str, task_id: str, base_split_index: int) -> Path:
    return feature_cache_dir(output_root, task_id, base_split_index) / "extract.log"


def probe_mode_dir(output_root: str, mode: str, task_id: str, base_split_index: int) -> Path:
    return Path(output_root) / mode / task_id / base_split_tag(base_split_index)


def probe_run_dir(
    output_root: str,
    mode: str,
    task_id: str,
    base_split_index: int,
    run_idx: int,
    run_seed: int,
) -> Path:
    return probe_mode_dir(output_root, mode, task_id, base_split_index) / f"run_{run_idx}_seed_{run_seed}"


def launcher_log_path(
    output_root: str,
    stage: str,
    task_id: str,
    base_split_index: int,
    mode: Optional[str] = None,
) -> Path:
    if stage == "extract":
        return feature_cache_dir(output_root, task_id, base_split_index) / "launcher_stdout.log"
    if mode is None:
        raise ValueError("mode is required for probe launcher log path.")
    return probe_mode_dir(output_root, mode, task_id, base_split_index) / "launcher_stdout.log"


class TimestampLogger:
    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        ensure_dir(str(self.log_path.parent))

    def log(self, message: str) -> None:
        line = f"[{now_ts()}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def save_feature_bundle(
    output_root: str,
    task_id: str,
    base_split_index: int,
    split_payloads: Mapping[str, Mapping[str, Sequence[Any]]],
    meta: Mapping[str, Any],
) -> None:
    cache_dir = feature_cache_dir(output_root, task_id, base_split_index)
    ensure_dir(str(cache_dir))
    arrays: Dict[str, np.ndarray] = {}
    for split_name, payload in split_payloads.items():
        for key, values in payload.items():
            if key == "features":
                arrays[f"{split_name}_{key}"] = np.asarray(values, dtype=np.float32)
            elif key == "targets":
                values_np = np.asarray(values)
                if values_np.dtype.kind in {"U", "S", "O"}:
                    arrays[f"{split_name}_{key}"] = values_np.astype(object)
                else:
                    arrays[f"{split_name}_{key}"] = values_np
            else:
                arrays[f"{split_name}_{key}"] = np.asarray(values, dtype=object)
    np.savez_compressed(feature_bundle_path(output_root, task_id, base_split_index), **arrays)
    write_json(str(feature_meta_path(output_root, task_id, base_split_index)), dict(meta))


def load_feature_bundle(output_root: str, task_id: str, base_split_index: int) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    bundle_path = feature_bundle_path(output_root, task_id, base_split_index)
    meta_path = feature_meta_path(output_root, task_id, base_split_index)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Feature bundle not found: {bundle_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Feature meta not found: {meta_path}")
    bundle = np.load(bundle_path, allow_pickle=True)
    split_payloads: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        split_payloads[split_name] = {
            "features": bundle[f"{split_name}_features"],
            "targets": bundle[f"{split_name}_targets"],
            "sample_ids": bundle[f"{split_name}_sample_ids"],
            "subject_ids": bundle[f"{split_name}_subject_ids"],
            "source_splits": bundle[f"{split_name}_source_splits"],
        }
    meta = load_yaml(str(meta_path)) if meta_path.suffix in {".yaml", ".yml"} else None
    if meta is None:
        import json

        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    return split_payloads, meta


def feature_cache_ready(output_root: str, task_id: str, base_split_index: int) -> bool:
    bundle = feature_bundle_path(output_root, task_id, base_split_index)
    meta = feature_meta_path(output_root, task_id, base_split_index)
    state = feature_cache_dir(output_root, task_id, base_split_index) / "run_state.json"
    if not bundle.exists() or not meta.exists() or not state.exists():
        return False
    try:
        import json

        with open(state, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload.get("state") == "completed"
    except Exception:
        return False


def split_eval_subjects(subject_ids: Sequence[Any], seed: int) -> Tuple[List[int], List[int], List[str], List[str]]:
    normalized = [str(subject_id) for subject_id in subject_ids]
    unique_subjects = list(dict.fromkeys(normalized))
    if len(unique_subjects) < 2:
        raise ValueError("Need at least 2 unique subjects to split eval_pool into val/test.")
    rng = random.Random(int(seed))
    rng.shuffle(unique_subjects)
    mid = len(unique_subjects) // 2
    if mid <= 0:
        mid = 1
    if mid >= len(unique_subjects):
        mid = len(unique_subjects) - 1
    val_subjects = unique_subjects[:mid]
    test_subjects = unique_subjects[mid:]
    val_subject_set = set(val_subjects)
    test_subject_set = set(test_subjects)
    val_idx = [idx for idx, subject_id in enumerate(normalized) if subject_id in val_subject_set]
    test_idx = [idx for idx, subject_id in enumerate(normalized) if subject_id in test_subject_set]
    if not val_idx or not test_idx:
        raise ValueError("Subject-level split produced an empty validation or test partition.")
    return val_idx, test_idx, val_subjects, test_subjects


def format_mean_std(values: Sequence[float]) -> str:
    if not values:
        return "NA"
    series = pd.Series(values, dtype=float)
    return f"{series.mean():.4f} +/- {series.std(ddof=0):.4f}"


def summarize_metric_dicts(metric_payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    keys = sorted({key for payload in metric_payloads for key in payload.keys()})
    for key in keys:
        values: List[float] = []
        for payload in metric_payloads:
            value = payload.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            continue
        series = pd.Series(values, dtype=float)
        summary[key] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "values": values,
            "mean_std": format_mean_std(values),
        }
    return summary


def rows_to_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(str(path.parent))
    if rows:
        pd.DataFrame(list(rows)).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def task_output_config(
    task_cfg: Mapping[str, Any],
    probe_cfg: Mapping[str, Any],
    *,
    base_split_index: int,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_id": task_cfg["task_id"],
        "display_name": task_cfg.get("display_name", task_cfg["task_id"]),
        "task_type": task_cfg["task_type"],
        "target_key": task_cfg["target_key"],
        "base_split_index": int(base_split_index),
        "base_split_tag": base_split_tag(base_split_index),
        "probe_config": dict(probe_cfg),
    }
    if mode is not None:
        payload["mode"] = mode
    return payload
