from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_table3 import (  # noqa: E402
    ATLAS_ROI_N,
    get_data_transform,
    get_model_class,
    infer_checkpoint_paths,
    resolve_repo_relative,
)
from table3_dataset import make_dataloaders  # noqa: E402
from table3_utils import LockGuard, load_task_config, update_run_state, write_json  # noqa: E402
from LP_MLP.probe_utils import (  # noqa: E402
    TimestampLogger,
    base_split_tag,
    feature_cache_dir,
    feature_log_path,
    load_probe_config,
    resolve_base_split_index,
    save_feature_bundle,
    task_output_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen LCM backbone features for LP/MLP probes")
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--probe_config", type=str, default="LP_MLP/probe_defaults.yaml")
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--base_split_index", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_root", type=str, default="LP_MLP/outputs")
    parser.add_argument("--gpu_lock_file", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def resolve_task_cfg(args: argparse.Namespace, probe_cfg: Mapping[str, Any]) -> Tuple[Dict[str, Any], int]:
    task_cfg = load_task_config(args.config, args.task_id)
    model_cfg = task_cfg["model"]
    base_split_index = (
        int(args.base_split_index)
        if args.base_split_index is not None
        else int(probe_cfg.get("base_split_index", 1))
    )
    resolved_split_index = resolve_base_split_index(task_cfg, base_split_index)
    model_cfg["pretrained_dir"] = resolve_repo_relative(model_cfg["pretrained_dir"], args.config)
    if args.num_workers is not None:
        task_cfg.setdefault("runtime", {})
        task_cfg["runtime"]["num_workers"] = args.num_workers
    return task_cfg, resolved_split_index


def build_backbone(task_cfg: Mapping[str, Any]) -> torch.nn.Module:
    model_cfg = task_cfg["model"]
    node_sz = ATLAS_ROI_N[model_cfg["atlas"]]
    input_dim = node_sz if model_cfg["node_attr"] != "BOLD" else model_cfg["bold_winsize"]
    model = get_model_class(model_cfg["models"])(
        node_sz=node_sz,
        out_channel=model_cfg["hiddim"],
        in_channel=input_dim,
        batch_size=model_cfg["batch_size"],
        device=model_cfg["device"],
        nlayer=model_cfg["nlayer"],
        heads=model_cfg["nhead"],
    )
    backbone_ckpt, _ = infer_checkpoint_paths(
        model_cfg["pretrained_dir"],
        load_dname=model_cfg.get("load_dname", "hcpa"),
    )
    backbone_state = torch.load(backbone_ckpt, map_location="cpu")
    model.load_state_dict(backbone_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def _pool_with_batch_index(node_features: torch.Tensor, batch_index: torch.Tensor, sample_count: int) -> torch.Tensor:
    if node_features.dim() != 2:
        node_features = node_features.reshape(node_features.shape[0], -1)
    feature_dim = int(node_features.shape[-1])
    pooled = torch.zeros(sample_count, feature_dim, device=node_features.device, dtype=node_features.dtype)
    counts = torch.zeros(sample_count, 1, device=node_features.device, dtype=node_features.dtype)
    pooled.index_add_(0, batch_index, node_features)
    counts.index_add_(0, batch_index, torch.ones(batch_index.shape[0], 1, device=node_features.device, dtype=node_features.dtype))
    pooled = pooled / counts.clamp_min(1.0)
    return pooled


def pool_backbone_output(raw_output: Any, batch: Any, sample_count: int) -> torch.Tensor:
    if isinstance(raw_output, (tuple, list)):
        raw_output = raw_output[0]
    if not isinstance(raw_output, torch.Tensor):
        raise TypeError(f"Unsupported backbone output type: {type(raw_output)}")

    if raw_output.dim() == 1:
        raw_output = raw_output.unsqueeze(-1)
    if raw_output.dim() >= 3 and raw_output.shape[0] == sample_count:
        return raw_output.reshape(sample_count, -1)
    if raw_output.dim() == 2 and raw_output.shape[0] == sample_count:
        return raw_output

    batch_index = getattr(batch, "batch", None)
    if batch_index is None:
        raise ValueError("Backbone output needs pooling, but batch.batch is missing.")
    if raw_output.dim() == 2 and raw_output.shape[0] == batch_index.numel():
        return _pool_with_batch_index(raw_output, batch_index, sample_count)
    if raw_output.dim() >= 3 and raw_output.shape[0] == batch_index.numel():
        flat = raw_output.reshape(raw_output.shape[0], -1)
        return _pool_with_batch_index(flat, batch_index, sample_count)

    raise ValueError(
        f"Cannot pool backbone output with shape={tuple(raw_output.shape)} for sample_count={sample_count}. "
        f"batch_nodes={int(batch_index.numel()) if batch_index is not None else 'NA'}"
    )


def target_from_batch(batch: Any, task_cfg: Mapping[str, Any]) -> np.ndarray:
    target_key = task_cfg["target_key"]
    if target_key == "age":
        return batch.age.view(-1).detach().cpu().numpy().astype(np.float32)
    if target_key == "sex":
        return batch.sex.view(-1).detach().cpu().numpy().astype(np.int64)
    if target_key == "y":
        return batch.y.view(-1).detach().cpu().numpy().astype(np.int64)
    raise KeyError(f"Unsupported target_key: {target_key}")


def extract_split(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    task_cfg: Mapping[str, Any],
    *,
    logger: TimestampLogger,
    max_batches: Optional[int] = None,
) -> Dict[str, List[Any]]:
    dataset = loader.dataset
    payload: Dict[str, List[Any]] = {
        "features": [],
        "targets": [],
        "sample_ids": [],
        "subject_ids": [],
        "source_splits": [],
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if hasattr(batch, "sample_idx"):
                sample_indices = batch.sample_idx.view(-1).detach().cpu().tolist()
            else:
                raise AttributeError("Expected collated batch to contain sample_idx.")
            batch = batch.to(device)
            backbone_output = model(batch)
            pooled = pool_backbone_output(backbone_output, batch, len(sample_indices))
            targets = target_from_batch(batch, task_cfg)
            pooled_np = pooled.detach().cpu().numpy().astype(np.float32)

            if pooled_np.shape[0] != len(sample_indices):
                raise ValueError(
                    f"Pooled feature rows={pooled_np.shape[0]} do not match sample count={len(sample_indices)}"
                )
            if targets.shape[0] != len(sample_indices):
                raise ValueError(
                    f"Target rows={targets.shape[0]} do not match sample count={len(sample_indices)}"
                )

            for row_idx, sample_index in enumerate(sample_indices):
                sample = dataset.samples[int(sample_index)]
                payload["features"].append(pooled_np[row_idx])
                payload["targets"].append(targets[row_idx].item() if hasattr(targets[row_idx], "item") else targets[row_idx])
                payload["sample_ids"].append(getattr(sample, "sample_id", None))
                payload["subject_ids"].append(getattr(sample, "subject_id", None))
                payload["source_splits"].append(getattr(sample, "split", None))
            logger.log(
                f"split={getattr(dataset.samples[0], 'split', 'unknown') if dataset.samples else 'unknown'} "
                f"batch={batch_idx + 1} samples={len(sample_indices)} pooled_shape={tuple(pooled_np.shape)}"
            )
    return payload


def main() -> None:
    args = parse_args()
    probe_cfg = load_probe_config(args.probe_config)
    task_cfg, base_split_index = resolve_task_cfg(args, probe_cfg)

    cache_dir = feature_cache_dir(args.output_root, args.task_id, base_split_index)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = TimestampLogger(feature_log_path(args.output_root, args.task_id, base_split_index))
    update_run_state(
        str(cache_dir),
        "running",
        task_id=args.task_id,
        base_split_index=base_split_index,
        base_split_tag=base_split_tag(base_split_index),
        pid=os.getpid(),
        stage="extract",
    )

    lock_guard = LockGuard(args.gpu_lock_file)
    if args.gpu_lock_file:
        lock_guard.install_signal_handlers()

    try:
        requested_device = args.device
        device = torch.device(requested_device if requested_device != "cuda" else "cuda:0")
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.log(f"CUDA unavailable, falling back to CPU from requested device={requested_device}.")
            device = torch.device("cpu")
        task_cfg["model"]["device"] = str(device)
        logger.log(
            f"Preparing feature extraction for task={args.task_id} base_split={base_split_tag(base_split_index)} "
            f"device={device}"
        )

        transform = get_data_transform(task_cfg["model"]["models"])
        datasets, loaders, audit = make_dataloaders(
            task_cfg,
            base_split_index,
            batch_size=task_cfg["model"]["batch_size"],
            node_attr=task_cfg["model"]["node_attr"],
            adj_type=task_cfg["model"]["adj_type"],
            fc_th=task_cfg["model"]["fc_th"],
            bold_winsize=task_cfg["model"]["bold_winsize"],
            transform=transform,
            num_workers=task_cfg.get("runtime", {}).get("num_workers", 0),
        )
        split_counts = audit.get("split_counts", {})
        logger.log(
            "Split summary: "
            + ", ".join(
                f"{split}:{info['samples']} samples/{info['subjects']} subjects"
                for split, info in split_counts.items()
            )
        )

        model = build_backbone(task_cfg).to(device)
        max_batches = 2 if args.smoke_test else None

        split_payloads: Dict[str, Dict[str, List[Any]]] = {}
        for split_name in ("train", "val", "test"):
            logger.log(f"Extracting split={split_name}")
            split_payloads[split_name] = extract_split(
                model,
                loaders[split_name],
                device,
                task_cfg,
                logger=logger,
                max_batches=max_batches,
            )

        feature_dim = 0
        train_features = split_payloads["train"]["features"]
        if train_features:
            feature_dim = int(np.asarray(train_features[0]).shape[-1])

        meta = task_output_config(task_cfg, probe_cfg, base_split_index=base_split_index)
        meta.update(
            {
                "feature_dim": feature_dim,
                "smoke_test": bool(args.smoke_test),
                "audit": audit,
                "split_counts": {
                    split_name: len(split_payloads[split_name]["sample_ids"])
                    for split_name in ("train", "val", "test")
                },
                "class_names": audit.get("label_meta", {}).get("class_names", []),
                "device": str(device),
            }
        )
        save_feature_bundle(args.output_root, args.task_id, base_split_index, split_payloads, meta)
        write_json(str(cache_dir / "config.json"), meta)
        logger.log(
            f"Saved feature cache to {cache_dir} feature_dim={feature_dim} "
            f"train={len(split_payloads['train']['sample_ids'])} "
            f"val={len(split_payloads['val']['sample_ids'])} "
            f"test={len(split_payloads['test']['sample_ids'])}"
        )
        update_run_state(
            str(cache_dir),
            "completed",
            task_id=args.task_id,
            base_split_index=base_split_index,
            feature_dim=feature_dim,
            smoke_test=bool(args.smoke_test),
            stage="extract",
        )
    except Exception as exc:
        update_run_state(
            str(cache_dir),
            "failed",
            task_id=args.task_id,
            base_split_index=base_split_index,
            error=str(exc),
            stage="extract",
        )
        raise
    finally:
        lock_guard.release()


if __name__ == "__main__":
    main()

