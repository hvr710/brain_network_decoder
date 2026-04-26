import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from table3_dataset import build_task_records, make_dataloaders
from table3_utils import (
    LockGuard,
    load_task_config,
    now_ts,
    release_gpu_lock,
    resolve_path,
    task_run_dir,
    update_run_state,
    write_json,
)


ATLAS_ROI_N = {
    "AAL_116": 116,
    "Gordon_333": 333,
    "Shaefer_100": 100,
    "Shaefer_200": 200,
    "Shaefer_400": 400,
    "D_160": 160,
}


def get_model_class(model_name: str) -> Any:
    if model_name == "none":
        from models import brain_identity

        return brain_identity.Identity
    if model_name == "neurodetour":
        from models import neuro_detour

        return neuro_detour.DetourTransformer
    if model_name == "neurodetourSingleFC":
        from models import neuro_detour

        return neuro_detour.DetourTransformerSingleFC
    if model_name == "neurodetourSingleSC":
        from models import neuro_detour

        return neuro_detour.DetourTransformerSingleSC
    if model_name == "bnt":
        from models import brain_net_transformer

        return brain_net_transformer.BrainNetworkTransformer
    if model_name == "braingnn":
        from models import brain_gnn

        return brain_gnn.Network
    if model_name == "bolt":
        from models import bolt

        return bolt.get_BolT
    if model_name == "graphormer":
        from models import graphormer

        return graphormer.Graphormer
    if model_name == "nagphormer":
        from models import nagphormer

        return nagphormer.TransformerModel
    if model_name == "transformer":
        from models import vanilla_model

        return vanilla_model.Transformer
    if model_name == "gcn":
        from models import vanilla_model

        return vanilla_model.GCN
    if model_name == "sage":
        from models import vanilla_model

        return vanilla_model.SAGE
    if model_name == "sgc":
        from models import vanilla_model

        return vanilla_model.SGC
    raise KeyError(f"Unsupported model: {model_name}")


def get_data_transform(model_name: str) -> Optional[Any]:
    if model_name == "graphormer":
        from models import graphormer

        return graphormer.ShortestDistance()
    if model_name == "nagphormer":
        from models import nagphormer

        return nagphormer.NAGdataTransform()
    return None


class RunLogger:
    def __init__(self, run_dir: str) -> None:
        self.log_path = Path(run_dir) / "train.log"

    def log(self, message: str) -> None:
        line = f"[{now_ts()}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LCM Table 3 finetuning")
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--task_dir_name", type=str, default=None)
    parser.add_argument("--task_dir_suffix", type=str, default=None)
    parser.add_argument("--gpu_lock_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--reserve_cuda_mem_mb", type=int, default=0)
    parser.add_argument("--reserve_cuda_poll_seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def clone_state_dict_to_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in module.state_dict().items()
    }


def reserve_cuda_cache(
    device: torch.device,
    target_reserved_mb: int,
    poll_seconds: int,
    logger: RunLogger,
) -> None:
    if device.type != "cuda" or target_reserved_mb <= 0:
        return

    mb = 1024 ** 2
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    safe_target_bytes = min(int(target_reserved_mb) * mb, max(0, total_bytes - 512 * mb))
    if safe_target_bytes <= 0:
        return

    logger.log(
        "CUDA cache reservation requested: "
        f"target={round(safe_target_bytes / mb)} MB, "
        f"reserved_now={round(torch.cuda.memory_reserved(device) / mb)} MB, "
        f"free_now={round(free_bytes / mb)} MB"
    )

    chunks: List[torch.Tensor] = []
    chunk_bytes = 256 * mb
    poll_seconds = max(1, int(poll_seconds))
    while torch.cuda.memory_reserved(device) < safe_target_bytes:
        reserved_bytes = torch.cuda.memory_reserved(device)
        needed_bytes = safe_target_bytes - reserved_bytes
        alloc_bytes = min(chunk_bytes, needed_bytes)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < alloc_bytes + 256 * mb:
            logger.log(
                "Waiting for CUDA reservation memory: "
                f"free={round(free_bytes / mb)} MB, "
                f"need_chunk={round(alloc_bytes / mb)} MB, "
                f"target_reserved={round(safe_target_bytes / mb)} MB"
            )
            time.sleep(poll_seconds)
            continue
        chunks.append(torch.empty(int(alloc_bytes), dtype=torch.uint8, device=device))

    del chunks
    torch.cuda.synchronize(device)
    logger.log(
        "CUDA cache reservation ready: "
        f"reserved={round(torch.cuda.memory_reserved(device) / mb)} MB, "
        f"allocated={round(torch.cuda.memory_allocated(device) / mb)} MB"
    )


def reduce_outputs(output: torch.Tensor, target_key: str, mode: str) -> torch.Tensor:
    if output.dim() == 3:
        if target_key == "age":
            output = output.mean(0)
        elif mode == "train":
            output = output.mean(0)
        else:
            output = output.max(0)[0]
    if target_key == "age" and output.dim() > 1:
        output = output.squeeze(-1)
    return output


def get_target_tensor(batch: Any, target_key: str, device: torch.device) -> torch.Tensor:
    if target_key == "age":
        return batch.age.view(-1).float().to(device)
    if target_key == "sex":
        return batch.sex.view(-1).long().to(device)
    if target_key == "y":
        return batch.y.view(-1).long().to(device)
    raise KeyError(f"Unsupported target_key: {target_key}")


def compute_train_target_std(datasets: Dict[str, Any], task_cfg: Dict[str, Any]) -> Optional[float]:
    if task_cfg["task_type"] != "regression":
        return None
    train_dataset = datasets.get("train")
    samples = getattr(train_dataset, "samples", None)
    if not samples:
        return None
    values = [float(sample.target) for sample in samples]
    if len(values) < 2:
        return None
    std = float(np.std(np.asarray(values, dtype=np.float32)))
    if std <= 0:
        return None
    return std


def compute_metrics(task_cfg: Dict[str, Any], y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
    if task_cfg["task_type"] == "classification":
        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_pred_np = np.asarray(y_pred, dtype=np.int64)
        acc = float(accuracy_score(y_true_np, y_pred_np))
        f1_weighted = float(f1_score(y_true_np, y_pred_np, average="weighted"))
        f1_macro = float(f1_score(y_true_np, y_pred_np, average="macro"))
        return {
            "acc": acc,
            "acc_percent": acc * 100.0,
            "f1": f1_weighted,
            "f1_weighted": f1_weighted,
            "f1_weighted_percent": f1_weighted * 100.0,
            "f1_macro": f1_macro,
            "f1_macro_percent": f1_macro * 100.0,
        }

    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_pred_np = np.asarray(y_pred, dtype=np.float32)
    mse_raw = float(np.mean((y_pred_np - y_true_np) ** 2))
    rmse = float(np.sqrt(mse_raw))
    train_target_std = task_cfg.get("train_target_std")
    mse_standardized = None
    if isinstance(train_target_std, (int, float)) and float(train_target_std) > 0:
        mse_standardized = float(mse_raw / float(train_target_std) ** 2)
    return {
        "mse": mse_raw,
        "mse_raw": mse_raw,
        "rmse": rmse,
        "pearson_r": pearson_r(y_true_np, y_pred_np),
        "mse_standardized": mse_standardized,
    }


def _sample_meta_from_index(loader: Any, sample_index: int) -> Dict[str, Any]:
    dataset = getattr(loader, "dataset", None)
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return {"sample_id": None, "subject_id": None, "split": None}
    if 0 <= int(sample_index) < len(samples):
        sample = samples[int(sample_index)]
        return {
            "sample_id": getattr(sample, "sample_id", None),
            "subject_id": getattr(sample, "subject_id", None),
            "split": getattr(sample, "split", None),
        }
    return {"sample_id": None, "subject_id": None, "split": None}


def selection_value(task_cfg: Dict[str, Any], metrics: Dict[str, float]) -> Tuple[float, float]:
    key = task_cfg["model_select_metric"]
    if key == "val_f1":
        return metrics["f1"], metrics.get("acc", -1.0)
    if key == "val_mse":
        return -metrics["mse"], metrics.get("pearson_r", -1.0)
    raise KeyError(f"Unsupported model_select_metric: {key}")


def train_one_epoch(
    model: nn.Module,
    classifier: nn.Module,
    loader: Any,
    optimizer: optim.Optimizer,
    device: torch.device,
    task_cfg: Dict[str, Any],
    *,
    max_batches: Optional[int] = None,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    model.train()
    classifier.train()
    loss_fn = nn.MSELoss() if task_cfg["task_type"] == "regression" else nn.CrossEntropyLoss()
    grad_accum_steps = max(1, int(grad_accum_steps))

    losses: List[float] = []
    optimizer.zero_grad(set_to_none=True)
    accumulated_steps = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch.to(device)
        feat = model(batch)
        edge_index = batch.edge_index
        batch_index = batch.batch
        if isinstance(feat, tuple) or isinstance(feat, list):
            feat = feat[0]
        outputs = classifier(feat, edge_index, batch_index)
        pred = reduce_outputs(outputs[task_cfg["target_key"]], task_cfg["target_key"], mode="train")
        target = get_target_tensor(batch, task_cfg["target_key"], device)
        loss = loss_fn(pred, target)
        if hasattr(model, "loss"):
            loss = loss + model.loss
        losses.append(float(loss.detach().cpu().item()))
        (loss / grad_accum_steps).backward()
        accumulated_steps += 1
        if accumulated_steps % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if accumulated_steps and accumulated_steps % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"train_loss": float(np.mean(losses)) if losses else float("nan")}


def evaluate(
    model: nn.Module,
    classifier: nn.Module,
    loader: Any,
    device: torch.device,
    task_cfg: Dict[str, Any],
    *,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    classifier.eval()

    y_true: List[Any] = []
    y_pred: List[Any] = []
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(device)
            feat = model(batch)
            edge_index = batch.edge_index
            batch_index = batch.batch
            if isinstance(feat, tuple) or isinstance(feat, list):
                feat = feat[0]
            outputs = classifier(feat, edge_index, batch_index)
            reduced = reduce_outputs(outputs[task_cfg["target_key"]], task_cfg["target_key"], mode="eval")
            target = get_target_tensor(batch, task_cfg["target_key"], device)

            if hasattr(batch, "sample_idx"):
                sample_indices = batch.sample_idx.view(-1).detach().cpu().tolist()
            else:
                # Backward compatibility for old cached/batched payloads.
                sample_indices = batch.sample_index.view(-1).detach().cpu().tolist()
            if task_cfg["task_type"] == "classification":
                probs = torch.softmax(reduced, dim=-1)
                pred_labels = probs.argmax(dim=-1)
                y_true.extend(target.detach().cpu().tolist())
                y_pred.extend(pred_labels.detach().cpu().tolist())
                for row_idx, sample_index in enumerate(sample_indices):
                    sample_meta = _sample_meta_from_index(loader, int(sample_index))
                    row = {
                        "sample_id": sample_meta["sample_id"],
                        "subject_id": sample_meta["subject_id"],
                        "split": sample_meta["split"],
                        "y_true": int(target[row_idx].detach().cpu().item()),
                        "y_pred": int(pred_labels[row_idx].detach().cpu().item()),
                    }
                    for class_index in range(probs.shape[1]):
                        row[f"prob_{class_index}"] = float(probs[row_idx, class_index].detach().cpu().item())
                    records.append(row)
            else:
                pred_values = reduced.view(-1)
                y_true.extend(target.detach().cpu().tolist())
                y_pred.extend(pred_values.detach().cpu().tolist())
                for row_idx, sample_index in enumerate(sample_indices):
                    sample_meta = _sample_meta_from_index(loader, int(sample_index))
                    records.append(
                        {
                            "sample_id": sample_meta["sample_id"],
                            "subject_id": sample_meta["subject_id"],
                            "split": sample_meta["split"],
                            "y_true": float(target[row_idx].detach().cpu().item()),
                            "y_pred": float(pred_values[row_idx].detach().cpu().item()),
                        }
                    )

    return compute_metrics(task_cfg, y_true, y_pred), pd.DataFrame.from_records(records)


def build_run_diagnostics(
    task_cfg: Dict[str, Any],
    epoch_history: List[Dict[str, Any]],
    best_epoch: int,
) -> Dict[str, Any]:
    if not epoch_history:
        return {
            "final_epoch": 0,
            "epochs_after_best": 0,
            "train_loss_at_best": None,
            "final_train_loss": None,
            "overfit_flag": False,
        }

    final_epoch = int(epoch_history[-1]["epoch"])
    train_loss_at_best = None
    for item in epoch_history:
        if int(item["epoch"]) == int(best_epoch):
            train_loss_at_best = item["train_loss"]
            break
    final_train_loss = epoch_history[-1]["train_loss"]
    epochs_after_best = max(0, final_epoch - int(best_epoch))
    early_best = int(best_epoch) <= max(1, final_epoch // 3)
    train_kept_improving = (
        isinstance(train_loss_at_best, (int, float))
        and isinstance(final_train_loss, (int, float))
        and final_train_loss <= float(train_loss_at_best) * 0.8
    )
    overfit_flag = bool(early_best and epochs_after_best >= 10 and train_kept_improving)
    return {
        "final_epoch": final_epoch,
        "epochs_after_best": epochs_after_best,
        "train_loss_at_best": train_loss_at_best,
        "final_train_loss": final_train_loss,
        "overfit_flag": overfit_flag,
        "selection_metric": task_cfg["model_select_metric"],
        "selection_source": "val",
        "report_split": "test",
    }


def epoch_history_frame(epoch_history: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in epoch_history:
        row = {
            "epoch": item.get("epoch"),
            "train_loss": item.get("train_loss"),
        }
        for key, value in item.get("val_metrics", {}).items():
            row[f"val_{key}"] = value
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def infer_checkpoint_paths(pretrained_dir: str, load_dname: str = "hcpa") -> Tuple[str, str]:
    pretrained_root = Path(resolve_path(pretrained_dir))
    if not pretrained_root.exists():
        raise FileNotFoundError(f"Pretrained dir not found: {pretrained_dir}")

    backbone_ckpt = None
    head_ckpt = None
    for file_name in os.listdir(pretrained_root):
        if file_name.startswith(f"bb_fold0_{load_dname}Best_"):
            backbone_ckpt = str(pretrained_root / file_name)
        if file_name.startswith(f"head_fold0_{load_dname}Best_"):
            head_ckpt = str(pretrained_root / file_name)
    if backbone_ckpt is None or head_ckpt is None:
        raise FileNotFoundError(f"Cannot locate pretrained weights under {pretrained_root}")
    return backbone_ckpt, head_ckpt


def build_model_and_head(task_cfg: Dict[str, Any], audit: Dict[str, Any]) -> Tuple[nn.Module, nn.Module]:
    from models.heads import BNDecoder

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

    backbone_ckpt, head_ckpt = infer_checkpoint_paths(
        model_cfg["pretrained_dir"], load_dname=model_cfg.get("load_dname", "hcpa")
    )
    backbone_state = torch.load(backbone_ckpt, map_location="cpu")
    head_state = torch.load(head_ckpt, map_location="cpu")
    pretrain_nclass = int(head_state["object_query.weight"].shape[0] - 3)

    if task_cfg["target_key"] == "y":
        task_nclass = len(audit["label_meta"]["class_names"])
        if task_nclass <= 0:
            raise ValueError(
                f"Task {task_cfg['task_id']} resolved zero classes from labels; "
                "please check class_map/label_filters/label_source."
            )
        classifier = BNDecoder(
            model_cfg["hiddim"],
            nclass=pretrain_nclass,
            node_sz=node_sz,
            nlayer=model_cfg["decoder_layer"],
            head_num=8,
            finetune=True,
            finetune_nclass=task_nclass,
            finetune_tokenid=torch.arange(pretrain_nclass, pretrain_nclass + task_nclass).long(),
        )
        classifier.load_state_dict(head_state, strict=False)
    else:
        classifier = BNDecoder(
            model_cfg["hiddim"],
            nclass=pretrain_nclass,
            node_sz=node_sz,
            nlayer=model_cfg["decoder_layer"],
            head_num=8,
        )
        classifier.load_state_dict(head_state, strict=True)

    model.load_state_dict(backbone_state, strict=True)
    return model, classifier


def resolve_repo_relative(path: str, config_path: str) -> str:
    if not path:
        return path
    resolved = resolve_path(path)
    if Path(resolved).exists():
        return resolved
    config_dir = Path(config_path).resolve().parent
    candidate = (config_dir.parent / path).resolve()
    if candidate.exists():
        return str(candidate)
    return path


def resolve_task_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    task_cfg = load_task_config(args.config, args.task_id)
    model_cfg = task_cfg["model"]
    runtime_cfg = task_cfg.get("runtime", {})

    model_cfg["device"] = args.device
    if args.batch_size is not None:
        model_cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        model_cfg["epochs"] = args.epochs
    if args.max_patience is not None:
        model_cfg["max_patience"] = args.max_patience
    if args.num_workers is not None:
        runtime_cfg["num_workers"] = args.num_workers
    model_cfg["pretrained_dir"] = resolve_repo_relative(model_cfg["pretrained_dir"], args.config)
    task_cfg["runtime"] = runtime_cfg
    return task_cfg


def _format_examples(values: Any, limit: int = 8) -> str:
    if not values:
        return "[]"
    if not isinstance(values, list):
        values = [values]
    shown = [str(value) for value in values[:limit]]
    suffix = ", ..." if len(values) > limit else ""
    return "[" + ", ".join(shown) + suffix + "]"


def log_audit_summary(logger: RunLogger, audit: Dict[str, Any]) -> None:
    logger.log(
        "Split summary: "
        + ", ".join(
            f"{split}:{info['samples']} samples/{info['subjects']} subjects"
            for split, info in audit["split_counts"].items()
        )
    )

    split_audit = audit.get("split_audit", {})
    source_members = split_audit.get("source_members", {})
    if source_members:
        logger.log(
            "Source split members: "
            + ", ".join(
                f"{split}:raw={info['raw_members']},unique_subjects={info['unique_subjects']}"
                for split, info in source_members.items()
            )
        )
    source_files = split_audit.get("source_files", {})
    if source_files:
        logger.log(
            "Source split files: "
            + ", ".join(
                f"{split}:raw={info['raw_files']},valid_names={info['valid_file_names']},"
                f"unique_subjects={info['unique_subjects_before_label_join']}"
                for split, info in source_files.items()
            )
        )

    warning_specs = [
        ("missing_files", "missing_files_examples"),
        ("missing_labels", "missing_labels_examples"),
        ("missing_labels", "missing_label_file_examples"),
        ("missing_labels", "missing_label_subject_examples"),
        ("invalid_file_names", "invalid_file_name_examples"),
        ("duplicate_files", "duplicate_file_subject_examples"),
    ]
    emitted = set()
    for count_key, examples_key in warning_specs:
        counts = split_audit.get(count_key, {})
        if examples_key not in split_audit:
            continue
        examples_by_split = split_audit.get(examples_key, {})
        for split, count in counts.items():
            try:
                count_value = int(count)
            except (TypeError, ValueError):
                continue
            if count_value <= 0:
                continue
            dedupe_key = (count_key, examples_key, split)
            if dedupe_key in emitted:
                continue
            emitted.add(dedupe_key)
            logger.log(
                f"Audit warning: {split} {count_key}={count_value}; "
                f"{examples_key}={_format_examples(examples_by_split.get(split, []))}"
            )


def main() -> None:
    args = parse_args()
    task_cfg = resolve_task_cfg(args)
    if args.fold not in task_cfg["fold_ids"]:
        raise ValueError(f"Fold {args.fold} not configured for task {args.task_id}")

    run_dir = task_run_dir(
        args.output_root,
        args.task_id,
        args.fold,
        args.seed,
        task_dir_name_override=args.task_dir_name,
        task_dir_suffix=args.task_dir_suffix,
    )
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir)
    update_run_state(run_dir, "running", task_id=args.task_id, fold=args.fold, seed=args.seed, pid=os.getpid())
    lock_guard = LockGuard(args.gpu_lock_file)
    if args.gpu_lock_file:
        lock_guard.install_signal_handlers()

    try:
        set_seed(args.seed)
        task_cfg["model"]["device"] = args.device
        model_cfg = task_cfg["model"]
        runtime_cfg = task_cfg.get("runtime", {})

        if args.audit_only:
            _, audit = build_task_records(task_cfg, args.fold)
            write_json(str(Path(run_dir) / "audit.json"), audit)
            resolved_payload = {
                "task_cfg": task_cfg,
                "audit": audit,
                "run_dir": run_dir,
            }
            write_json(str(Path(run_dir) / "config.json"), resolved_payload)
            logger.log(f"Prepared audit for task={args.task_id} fold={args.fold} seed={args.seed}")
            log_audit_summary(logger, audit)
            logger.log("Audit only mode completed.")
            update_run_state(run_dir, "completed", audit_only=True)
            return

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        task_cfg["model"]["device"] = str(device)
        reserve_cuda_cache(device, args.reserve_cuda_mem_mb, args.reserve_cuda_poll_seconds, logger)

        transform = get_data_transform(model_cfg["models"])
        datasets, loaders, audit = make_dataloaders(
            task_cfg,
            args.fold,
            batch_size=model_cfg["batch_size"],
            node_attr=model_cfg["node_attr"],
            adj_type=model_cfg["adj_type"],
            fc_th=model_cfg["fc_th"],
            bold_winsize=model_cfg["bold_winsize"],
            transform=transform,
            num_workers=runtime_cfg.get("num_workers", 0),
        )
        split_counts = audit.get("split_counts", {})
        empty_splits = [split for split in ("train", "val", "test") if split_counts.get(split, {}).get("samples", 0) <= 0]
        if empty_splits:
            raise ValueError(
                f"Empty splits for task={args.task_id}, fold={args.fold}: {empty_splits}. "
                "Please check split files and label joins."
            )
        write_json(str(Path(run_dir) / "audit.json"), audit)

        resolved_payload = {
            "task_cfg": task_cfg,
            "audit": audit,
            "run_dir": run_dir,
        }
        write_json(str(Path(run_dir) / "config.json"), resolved_payload)
        logger.log(f"Prepared task={args.task_id} fold={args.fold} seed={args.seed}")
        log_audit_summary(logger, audit)

        model, classifier = build_model_and_head(task_cfg, audit)
        model = model.to(device)
        classifier = classifier.to(device)
        task_cfg["train_target_std"] = compute_train_target_std(datasets, task_cfg)
        optimizer = optim.Adam(
            list(model.parameters()) + list(classifier.parameters()),
            lr=model_cfg["lr"],
            weight_decay=model_cfg["decay"],
            foreach=False,
        )

        max_batches = 1 if args.smoke_test else None
        max_epochs = 1 if args.smoke_test else model_cfg["epochs"]
        patience_limit = 1 if args.smoke_test else model_cfg["max_patience"]
        grad_accum_steps = max(1, int(args.grad_accum_steps))
        if grad_accum_steps > 1:
            logger.log(f"Using gradient accumulation: {grad_accum_steps} steps.")

        best_payload: Optional[Dict[str, Any]] = None
        best_backbone_state = None
        best_head_state = None
        wait_count = 0
        epoch_history: List[Dict[str, Any]] = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(1, max_epochs + 1):
            train_metrics = train_one_epoch(
                model,
                classifier,
                loaders["train"],
                optimizer,
                device,
                task_cfg,
                max_batches=max_batches,
                grad_accum_steps=grad_accum_steps,
            )
            val_metrics, _ = evaluate(
                model,
                classifier,
                loaders["val"],
                device,
                task_cfg,
                max_batches=max_batches,
            )
            logger.log(
                f"Epoch {epoch}: train={json.dumps(train_metrics, ensure_ascii=False)} "
                f"val={json.dumps(val_metrics, ensure_ascii=False)}"
            )
            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics.get("train_loss"),
                    "val_metrics": val_metrics,
                }
            )

            current_score = selection_value(task_cfg, val_metrics)
            best_score = None if best_payload is None else selection_value(task_cfg, best_payload["val_metrics"])
            if best_score is None or current_score > best_score:
                best_payload = {
                    "best_epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
                best_backbone_state = clone_state_dict_to_cpu(model)
                best_head_state = clone_state_dict_to_cpu(classifier)
                wait_count = 0
                torch.save(best_backbone_state, str(Path(run_dir) / "backbone_best.pt"))
                torch.save(best_head_state, str(Path(run_dir) / "head_best.pt"))
                if device.type == "cuda" and args.reserve_cuda_mem_mb <= 0:
                    torch.cuda.empty_cache()
                logger.log(f"New best checkpoint at epoch {epoch}.")
            else:
                wait_count += 1
                if wait_count >= patience_limit:
                    logger.log(f"Early stopping at epoch {epoch} (patience={patience_limit}).")
                    break

        if best_payload is None or best_backbone_state is None or best_head_state is None:
            raise RuntimeError("No valid checkpoint was produced.")

        model.load_state_dict(best_backbone_state)
        classifier.load_state_dict(best_head_state)
        test_metrics, test_predictions = evaluate(
            model,
            classifier,
            loaders["test"],
            device,
            task_cfg,
            max_batches=max_batches,
        )
        peak_gpu_mem_mb = None
        if device.type == "cuda":
            peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

        run_diagnostics = build_run_diagnostics(task_cfg, epoch_history, best_payload["best_epoch"])
        best_payload["test_metrics"] = test_metrics
        best_payload["peak_gpu_mem_mb"] = peak_gpu_mem_mb
        best_payload["class_names"] = audit["label_meta"]["class_names"]
        best_payload["smoke_test"] = bool(args.smoke_test)
        best_payload["train_target_std"] = task_cfg.get("train_target_std")
        best_payload.update(run_diagnostics)
        write_json(str(Path(run_dir) / "best_metrics.json"), best_payload)
        epoch_history_frame(epoch_history).to_csv(str(Path(run_dir) / "epoch_history.csv"), index=False)
        test_predictions.to_csv(str(Path(run_dir) / "test_predictions.csv"), index=False)

        logger.log(f"Best epoch={best_payload['best_epoch']}")
        logger.log(f"Test metrics={json.dumps(test_metrics, ensure_ascii=False)}")
        logger.log(
            "Run diagnostics="
            + json.dumps(
                {
                    "epochs_after_best": best_payload["epochs_after_best"],
                    "train_loss_at_best": best_payload["train_loss_at_best"],
                    "final_train_loss": best_payload["final_train_loss"],
                    "overfit_flag": best_payload["overfit_flag"],
                },
                ensure_ascii=False,
            )
        )
        update_run_state(
            run_dir,
            "completed",
            best_epoch=best_payload["best_epoch"],
            peak_gpu_mem_mb=peak_gpu_mem_mb,
            epochs_after_best=best_payload["epochs_after_best"],
            overfit_flag=best_payload["overfit_flag"],
        )
    except Exception as exc:
        update_run_state(run_dir, "failed", error=str(exc))
        raise
    finally:
        lock_guard.release()


if __name__ == "__main__":
    main()
