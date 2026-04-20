from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, Ridge
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_table3 import compute_metrics  # noqa: E402
from table3_utils import now_ts, update_run_state, write_json  # noqa: E402
from LP_MLP.probe_utils import (  # noqa: E402
    TimestampLogger,
    base_split_tag,
    format_mean_std,
    load_feature_bundle,
    load_probe_config,
    probe_mode_dir,
    probe_run_dir,
    rows_to_csv,
    split_eval_subjects,
    summarize_metric_dicts,
    task_output_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LP/MLP probes on cached LCM features")
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--probe_config", type=str, default="LP_MLP/probe_defaults.yaml")
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["linear_probe", "mlp_probe"], required=True)
    parser.add_argument("--base_split_index", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_root", type=str, default="LP_MLP/outputs")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_task_cfg(config_path: str, task_id: str, bundle_meta: Mapping[str, Any]) -> Dict[str, Any]:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from table3_utils import load_task_config  # noqa: WPS433

    task_cfg = load_task_config(config_path, task_id)
    class_names = bundle_meta.get("class_names", [])
    task_cfg["label_meta"] = {"class_names": class_names}
    train_targets = np.asarray(bundle_meta.get("train_targets", []), dtype=np.float32)
    if task_cfg["task_type"] == "regression":
        if train_targets.size <= 1:
            task_cfg["train_target_std"] = None
        else:
            std = float(train_targets.std())
            task_cfg["train_target_std"] = std if std > 0 else None
    else:
        task_cfg["train_target_std"] = None
    return task_cfg


def ensure_numpy_2d(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={features.shape}")
    return features.astype(np.float32, copy=False)


def make_eval_pool(split_payloads: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    val_payload = split_payloads["val"]
    test_payload = split_payloads["test"]
    eval_pool = {
        "features": np.concatenate([ensure_numpy_2d(val_payload["features"]), ensure_numpy_2d(test_payload["features"])], axis=0),
        "targets": np.concatenate([np.asarray(val_payload["targets"]), np.asarray(test_payload["targets"])], axis=0),
        "sample_ids": np.concatenate([np.asarray(val_payload["sample_ids"], dtype=object), np.asarray(test_payload["sample_ids"], dtype=object)], axis=0),
        "subject_ids": np.concatenate([np.asarray(val_payload["subject_ids"], dtype=object), np.asarray(test_payload["subject_ids"], dtype=object)], axis=0),
        "source_splits": np.concatenate([np.asarray(val_payload["source_splits"], dtype=object), np.asarray(test_payload["source_splits"], dtype=object)], axis=0),
    }
    return eval_pool


def slice_payload(payload: Mapping[str, np.ndarray], indices: Sequence[int]) -> Dict[str, np.ndarray]:
    index_array = np.asarray(indices, dtype=np.int64)
    return {
        key: np.asarray(value)[index_array]
        for key, value in payload.items()
    }


def build_prediction_rows(
    payload: Mapping[str, np.ndarray],
    y_pred: Sequence[Any],
    *,
    run_split: str,
    mode: str,
    run_idx: int,
    run_seed: int,
    probabilities: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    y_true = np.asarray(payload["targets"])
    y_pred_arr = np.asarray(y_pred)
    for idx in range(len(y_true)):
        row: Dict[str, Any] = {
            "sample_id": payload["sample_ids"][idx],
            "subject_id": payload["subject_ids"][idx],
            "source_split": payload["source_splits"][idx],
            "run_split": run_split,
            "mode": mode,
            "run_idx": run_idx,
            "run_seed": run_seed,
            "y_true": y_true[idx].item() if hasattr(y_true[idx], "item") else y_true[idx],
            "y_pred": y_pred_arr[idx].item() if hasattr(y_pred_arr[idx], "item") else y_pred_arr[idx],
        }
        if probabilities is not None:
            for class_idx in range(probabilities.shape[1]):
                row[f"prob_{class_idx}"] = float(probabilities[idx, class_idx])
        rows.append(row)
    return rows


def build_split_manifest(
    eval_payload: Mapping[str, np.ndarray],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    run_idx: int,
    run_seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_split, indices in (("val", val_idx), ("test", test_idx)):
        for index in indices:
            rows.append(
                {
                    "run_idx": run_idx,
                    "run_seed": run_seed,
                    "run_split": run_split,
                    "sample_id": eval_payload["sample_ids"][index],
                    "subject_id": eval_payload["subject_ids"][index],
                    "source_split": eval_payload["source_splits"][index],
                }
            )
    return rows


def save_run_files(
    run_dir: Path,
    *,
    config_payload: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    val_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    split_manifest_rows: Sequence[Mapping[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(run_dir / "config.json"), dict(config_payload))
    write_json(str(run_dir / "val_metrics.json"), dict(val_metrics))
    write_json(str(run_dir / "test_metrics.json"), dict(test_metrics))
    rows_to_csv(run_dir / "val_predictions.csv", val_rows)
    rows_to_csv(run_dir / "test_predictions.csv", test_rows)
    rows_to_csv(run_dir / "split_manifest.csv", split_manifest_rows)


def build_mlp_probe(input_dim: int, output_dim: int, mlp_cfg: Mapping[str, Any]) -> nn.Sequential:
    hidden_dim = int(mlp_cfg.get("hidden_dim", 64))
    dropout = float(mlp_cfg.get("dropout", 0.1))
    num_layers = max(int(mlp_cfg.get("num_layers", 2)), 1)
    input_norm = str(mlp_cfg.get("input_norm", "layernorm")).lower()

    layers: List[nn.Module] = []
    if input_norm == "layernorm":
        layers.append(nn.LayerNorm(int(input_dim)))
    elif input_norm not in {"none", "identity"}:
        raise ValueError(f"Unsupported mlp_probe.input_norm: {input_norm}")

    in_dim = int(input_dim)
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        if targets.dtype.kind in {"i", "u"}:
            self.targets = torch.from_numpy(targets.astype(np.int64, copy=False))
        else:
            self.targets = torch.from_numpy(targets.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def standardize_features(
    train_features: np.ndarray,
    *others: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    outputs = [((train_features - mean) / std).astype(np.float32)]
    for array in others:
        outputs.append(((array - mean) / std).astype(np.float32))
    return tuple(outputs)


def evaluate_mlp(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    task_cfg: Mapping[str, Any],
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, Optional[np.ndarray]]:
    model.eval()
    features_tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        logits = model(features_tensor)
    if task_cfg["task_type"] == "classification":
        targets_tensor = torch.from_numpy(targets.astype(np.int64, copy=False)).to(device)
        loss = float(nn.CrossEntropyLoss()(logits, targets_tensor).detach().cpu().item())
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        metrics = compute_metrics(task_cfg, targets.tolist(), predictions.tolist())
        return loss, metrics, predictions, probabilities
    targets_tensor = torch.from_numpy(targets.astype(np.float32, copy=False)).to(device).view(-1, 1)
    predictions_tensor = logits.view(-1, 1)
    loss = float(nn.MSELoss()(predictions_tensor, targets_tensor).detach().cpu().item())
    predictions = predictions_tensor.view(-1).detach().cpu().numpy()
    metrics = compute_metrics(task_cfg, targets.tolist(), predictions.tolist())
    return loss, metrics, predictions, None


def run_linear_probe(
    task_cfg: Mapping[str, Any],
    train_payload: Mapping[str, np.ndarray],
    val_payload: Mapping[str, np.ndarray],
    test_payload: Mapping[str, np.ndarray],
    *,
    run_idx: int,
    run_seed: int,
    logger: TimestampLogger,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_X = ensure_numpy_2d(train_payload["features"])
    train_y = np.asarray(train_payload["targets"])
    val_X = ensure_numpy_2d(val_payload["features"])
    val_y = np.asarray(val_payload["targets"])
    test_X = ensure_numpy_2d(test_payload["features"])
    test_y = np.asarray(test_payload["targets"])

    if task_cfg["task_type"] == "classification":
        probe = LogisticRegression(C=1.0, max_iter=8000, random_state=run_seed)
        probe.fit(train_X, train_y.astype(np.int64))
        val_pred = probe.predict(val_X)
        test_pred = probe.predict(test_X)
        val_prob = probe.predict_proba(val_X)
        test_prob = probe.predict_proba(test_X)
        val_metrics = compute_metrics(task_cfg, val_y.astype(np.int64).tolist(), val_pred.astype(np.int64).tolist())
        test_metrics = compute_metrics(task_cfg, test_y.astype(np.int64).tolist(), test_pred.astype(np.int64).tolist())
        val_rows = build_prediction_rows(val_payload, val_pred, run_split="val", mode="linear_probe", run_idx=run_idx, run_seed=run_seed, probabilities=val_prob)
        test_rows = build_prediction_rows(test_payload, test_pred, run_split="test", mode="linear_probe", run_idx=run_idx, run_seed=run_seed, probabilities=test_prob)
    else:
        probe = Ridge(alpha=1.0)
        probe.fit(train_X, train_y.astype(np.float32))
        val_pred = probe.predict(val_X)
        test_pred = probe.predict(test_X)
        val_metrics = compute_metrics(task_cfg, val_y.astype(np.float32).tolist(), val_pred.astype(np.float32).tolist())
        test_metrics = compute_metrics(task_cfg, test_y.astype(np.float32).tolist(), test_pred.astype(np.float32).tolist())
        val_rows = build_prediction_rows(val_payload, val_pred, run_split="val", mode="linear_probe", run_idx=run_idx, run_seed=run_seed)
        test_rows = build_prediction_rows(test_payload, test_pred, run_split="test", mode="linear_probe", run_idx=run_idx, run_seed=run_seed)

    logger.log(
        f"LP run={run_idx} seed={run_seed} "
        f"val={json.dumps(val_metrics, ensure_ascii=False)} "
        f"test={json.dumps(test_metrics, ensure_ascii=False)}"
    )
    return dict(val_metrics), dict(test_metrics), val_rows, test_rows


def run_mlp_probe(
    task_cfg: Mapping[str, Any],
    mlp_cfg: Mapping[str, Any],
    train_payload: Mapping[str, np.ndarray],
    val_payload: Mapping[str, np.ndarray],
    test_payload: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    run_dir: Path,
    run_idx: int,
    run_seed: int,
    logger: TimestampLogger,
    smoke_test: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_X, val_X, test_X = standardize_features(
        ensure_numpy_2d(train_payload["features"]),
        ensure_numpy_2d(val_payload["features"]),
        ensure_numpy_2d(test_payload["features"]),
    )
    train_y = np.asarray(train_payload["targets"])
    val_y = np.asarray(val_payload["targets"])
    test_y = np.asarray(test_payload["targets"])

    output_dim = len(task_cfg["label_meta"]["class_names"]) if task_cfg["task_type"] == "classification" else 1
    model = build_mlp_probe(train_X.shape[1], output_dim, mlp_cfg).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(mlp_cfg.get("lr", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-4)),
    )
    train_dataset = ArrayDataset(train_X, train_y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(mlp_cfg.get("batch_size", 128)),
        shuffle=True,
        drop_last=False,
    )
    max_epochs = 2 if smoke_test else int(mlp_cfg.get("epochs", 200))
    patience_limit = 1 if smoke_test else int(mlp_cfg.get("max_patience", 20))
    loss_fn = nn.CrossEntropyLoss() if task_cfg["task_type"] == "classification" else nn.MSELoss()

    best_state: Optional[Dict[str, Any]] = None
    best_val_loss: Optional[float] = None
    best_val_metrics: Optional[Dict[str, Any]] = None
    best_test_metrics: Optional[Dict[str, Any]] = None
    best_val_rows: Optional[List[Dict[str, Any]]] = None
    best_test_rows: Optional[List[Dict[str, Any]]] = None
    wait_count = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: List[float] = []
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            if task_cfg["task_type"] == "classification":
                loss = loss_fn(outputs, batch_targets.long())
            else:
                loss = loss_fn(outputs.view(-1, 1), batch_targets.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss, val_metrics, val_pred, val_prob = evaluate_mlp(model, val_X, val_y, task_cfg, device)
        test_loss, test_metrics, test_pred, test_prob = evaluate_mlp(model, test_X, test_y, task_cfg, device)
        logger.log(
            f"MLP run={run_idx} seed={run_seed} epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} test_loss={test_loss:.6f} "
            f"val={json.dumps(val_metrics, ensure_ascii=False)} "
            f"test={json.dumps(test_metrics, ensure_ascii=False)}"
        )

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = dict(val_metrics)
            best_test_metrics = dict(test_metrics)
            best_val_rows = build_prediction_rows(
                val_payload,
                val_pred,
                run_split="val",
                mode="mlp_probe",
                run_idx=run_idx,
                run_seed=run_seed,
                probabilities=val_prob,
            )
            best_test_rows = build_prediction_rows(
                test_payload,
                test_pred,
                run_split="test",
                mode="mlp_probe",
                run_idx=run_idx,
                run_seed=run_seed,
                probabilities=test_prob,
            )
            wait_count = 0
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_dir / "best.pt")
            logger.log(f"MLP run={run_idx} seed={run_seed} new best checkpoint at epoch={epoch}")
        else:
            wait_count += 1
            if wait_count >= patience_limit:
                logger.log(
                    f"MLP run={run_idx} seed={run_seed} early stopping at epoch={epoch} "
                    f"(patience={patience_limit})"
                )
                break

    if best_state is None or best_val_metrics is None or best_test_metrics is None or best_val_rows is None or best_test_rows is None:
        raise RuntimeError("MLP probe did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    return best_val_metrics, best_test_metrics, best_val_rows, best_test_rows


def main() -> None:
    args = parse_args()
    probe_cfg = load_probe_config(args.probe_config)
    base_split_index = (
        int(args.base_split_index)
        if args.base_split_index is not None
        else int(probe_cfg.get("base_split_index", 1))
    )
    split_payloads, bundle_meta = load_feature_bundle(args.output_root, args.task_id, base_split_index)
    bundle_meta["train_targets"] = np.asarray(split_payloads["train"]["targets"]).tolist()
    task_cfg = prepare_task_cfg(args.config, args.task_id, bundle_meta)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    mode_root = probe_mode_dir(args.output_root, args.mode, args.task_id, base_split_index)
    mode_root.mkdir(parents=True, exist_ok=True)
    mode_logger = TimestampLogger(mode_root / "train.log")
    mode_logger.log(
        f"Starting mode={args.mode} task={args.task_id} base_split={base_split_tag(base_split_index)} device={device}"
    )

    train_payload = {
        key: np.asarray(value)
        for key, value in split_payloads["train"].items()
    }
    eval_pool = make_eval_pool(split_payloads)
    run_count = 1 if args.smoke_test else int(probe_cfg.get("num_runs", 5))
    base_seed = int(probe_cfg.get("base_seed", 23))
    seed_stride = int(probe_cfg.get("seed_stride", 1))
    mlp_cfg = dict(probe_cfg.get("mlp_probe", {}))

    run_manifest_rows: List[Dict[str, Any]] = []
    val_metric_payloads: List[Dict[str, Any]] = []
    test_metric_payloads: List[Dict[str, Any]] = []

    for run_idx in range(run_count):
        run_seed = base_seed + run_idx * seed_stride
        set_seed(run_seed)
        run_dir = probe_run_dir(args.output_root, args.mode, args.task_id, base_split_index, run_idx, run_seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = TimestampLogger(run_dir / "train.log")
        update_run_state(
            str(run_dir),
            "running",
            mode=args.mode,
            task_id=args.task_id,
            base_split_index=base_split_index,
            run_idx=run_idx,
            run_seed=run_seed,
        )
        try:
            val_idx, test_idx, val_subjects, test_subjects = split_eval_subjects(eval_pool["subject_ids"], run_seed)
            val_payload = slice_payload(eval_pool, val_idx)
            test_payload = slice_payload(eval_pool, test_idx)
            split_manifest_rows = build_split_manifest(eval_pool, val_idx, test_idx, run_idx=run_idx, run_seed=run_seed)
            logger.log(
                f"run={run_idx} seed={run_seed} train={len(train_payload['sample_ids'])} "
                f"run_val={len(val_idx)} run_test={len(test_idx)} "
                f"val_subjects={len(val_subjects)} test_subjects={len(test_subjects)}"
            )

            config_payload = task_output_config(task_cfg, probe_cfg, base_split_index=base_split_index, mode=args.mode)
            config_payload.update(
                {
                    "device": str(device),
                    "run_idx": run_idx,
                    "run_seed": run_seed,
                    "smoke_test": bool(args.smoke_test),
                }
            )

            if args.mode == "linear_probe":
                val_metrics, test_metrics, val_rows, test_rows = run_linear_probe(
                    task_cfg,
                    train_payload,
                    val_payload,
                    test_payload,
                    run_idx=run_idx,
                    run_seed=run_seed,
                    logger=logger,
                )
            else:
                val_metrics, test_metrics, val_rows, test_rows = run_mlp_probe(
                    task_cfg,
                    mlp_cfg,
                    train_payload,
                    val_payload,
                    test_payload,
                    device=device,
                    run_dir=run_dir,
                    run_idx=run_idx,
                    run_seed=run_seed,
                    logger=logger,
                    smoke_test=bool(args.smoke_test),
                )

            save_run_files(
                run_dir,
                config_payload=config_payload,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                val_rows=val_rows,
                test_rows=test_rows,
                split_manifest_rows=split_manifest_rows,
            )
            update_run_state(
                str(run_dir),
                "completed",
                mode=args.mode,
                task_id=args.task_id,
                base_split_index=base_split_index,
                run_idx=run_idx,
                run_seed=run_seed,
            )

            manifest_row = {
                "run_idx": run_idx,
                "run_seed": run_seed,
                "run_dir": str(run_dir),
                "val_subject_count": len(val_subjects),
                "test_subject_count": len(test_subjects),
            }
            manifest_row.update({f"val_{key}": value for key, value in val_metrics.items()})
            manifest_row.update({f"test_{key}": value for key, value in test_metrics.items()})
            run_manifest_rows.append(manifest_row)
            val_metric_payloads.append(dict(val_metrics))
            test_metric_payloads.append(dict(test_metrics))
        except Exception as exc:
            update_run_state(
                str(run_dir),
                "failed",
                mode=args.mode,
                task_id=args.task_id,
                base_split_index=base_split_index,
                run_idx=run_idx,
                run_seed=run_seed,
                error=str(exc),
            )
            raise

    rows_to_csv(mode_root / "run_manifest.csv", run_manifest_rows)
    summary_payload = task_output_config(task_cfg, probe_cfg, base_split_index=base_split_index, mode=args.mode)
    summary_payload.update(
        {
            "completed_runs": len(run_manifest_rows),
            "expected_runs": run_count,
            "val_metrics": summarize_metric_dicts(val_metric_payloads),
            "test_metrics": summarize_metric_dicts(test_metric_payloads),
            "test_metrics_compact": {
                metric_name: payload["mean_std"]
                for metric_name, payload in summarize_metric_dicts(test_metric_payloads).items()
            },
            "generated_at": now_ts(),
        }
    )
    write_json(str(mode_root / "probe_summary.json"), summary_payload)
    mode_logger.log(
        f"Finished mode={args.mode} task={args.task_id} completed_runs={len(run_manifest_rows)} "
        + ", ".join(
            f"{metric}={payload['mean_std']}"
            for metric, payload in summary_payload["test_metrics"].items()
        )
    )


if __name__ == "__main__":
    main()
