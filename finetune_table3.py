import argparse
import copy
import json
import os
import random
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
    parser.add_argument("--output_root", type=str, default="outputs/table3")
    parser.add_argument("--gpu_lock_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
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


def compute_metrics(task_cfg: Dict[str, Any], y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
    if task_cfg["task_type"] == "classification":
        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_pred_np = np.asarray(y_pred, dtype=np.int64)
        return {
            "acc": float(accuracy_score(y_true_np, y_pred_np)),
            "f1": float(f1_score(y_true_np, y_pred_np, average="weighted")),
        }

    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_pred_np = np.asarray(y_pred, dtype=np.float32)
    mse = float(np.mean((y_pred_np - y_true_np) ** 2))
    return {
        "mse": mse,
        "pearson_r": pearson_r(y_true_np, y_pred_np),
    }


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
) -> Dict[str, float]:
    model.train()
    classifier.train()
    loss_fn = nn.MSELoss() if task_cfg["task_type"] == "regression" else nn.CrossEntropyLoss()

    losses: List[float] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        optimizer.zero_grad()
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
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
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

            sample_indices = batch.sample_index.view(-1).detach().cpu().tolist()
            if task_cfg["task_type"] == "classification":
                probs = torch.softmax(reduced, dim=-1)
                pred_labels = probs.argmax(dim=-1)
                y_true.extend(target.detach().cpu().tolist())
                y_pred.extend(pred_labels.detach().cpu().tolist())
                for row_idx, sample_index in enumerate(sample_indices):
                    sample = loader.dataset.samples[int(sample_index)]
                    row = {
                        "sample_id": sample.sample_id,
                        "subject_id": sample.subject_id,
                        "split": sample.split,
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
                    sample = loader.dataset.samples[int(sample_index)]
                    records.append(
                        {
                            "sample_id": sample.sample_id,
                            "subject_id": sample.subject_id,
                            "split": sample.split,
                            "y_true": float(target[row_idx].detach().cpu().item()),
                            "y_pred": float(pred_values[row_idx].detach().cpu().item()),
                        }
                    )

    return compute_metrics(task_cfg, y_true, y_pred), pd.DataFrame.from_records(records)


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
    task_cfg["runtime"] = runtime_cfg
    return task_cfg


def main() -> None:
    args = parse_args()
    task_cfg = resolve_task_cfg(args)
    if args.fold not in task_cfg["fold_ids"]:
        raise ValueError(f"Fold {args.fold} not configured for task {args.task_id}")

    run_dir = task_run_dir(args.output_root, args.task_id, args.fold, args.seed)
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
            logger.log(
                "Split summary: "
                + ", ".join(
                    f"{split}:{info['samples']} samples/{info['subjects']} subjects"
                    for split, info in audit["split_counts"].items()
                )
            )
            logger.log("Audit only mode completed.")
            update_run_state(run_dir, "completed", audit_only=True)
            return

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
        write_json(str(Path(run_dir) / "audit.json"), audit)

        resolved_payload = {
            "task_cfg": task_cfg,
            "audit": audit,
            "run_dir": run_dir,
        }
        write_json(str(Path(run_dir) / "config.json"), resolved_payload)
        logger.log(f"Prepared task={args.task_id} fold={args.fold} seed={args.seed}")
        logger.log(
            "Split summary: "
            + ", ".join(
                f"{split}:{info['samples']} samples/{info['subjects']} subjects"
                for split, info in audit["split_counts"].items()
            )
        )

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        task_cfg["model"]["device"] = str(device)
        model, classifier = build_model_and_head(task_cfg, audit)
        model = model.to(device)
        classifier = classifier.to(device)
        optimizer = optim.Adam(
            list(model.parameters()) + list(classifier.parameters()),
            lr=model_cfg["lr"],
            weight_decay=model_cfg["decay"],
        )

        max_batches = 1 if args.smoke_test else None
        max_epochs = 1 if args.smoke_test else model_cfg["epochs"]
        patience_limit = 1 if args.smoke_test else model_cfg["max_patience"]

        best_payload: Optional[Dict[str, Any]] = None
        best_backbone_state = None
        best_head_state = None
        wait_count = 0

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

            current_score = selection_value(task_cfg, val_metrics)
            best_score = None if best_payload is None else selection_value(task_cfg, best_payload["val_metrics"])
            if best_score is None or current_score > best_score:
                best_payload = {
                    "best_epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
                best_backbone_state = copy.deepcopy(model.state_dict())
                best_head_state = copy.deepcopy(classifier.state_dict())
                wait_count = 0
                torch.save(best_backbone_state, str(Path(run_dir) / "backbone_best.pt"))
                torch.save(best_head_state, str(Path(run_dir) / "head_best.pt"))
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

        best_payload["test_metrics"] = test_metrics
        best_payload["peak_gpu_mem_mb"] = peak_gpu_mem_mb
        best_payload["class_names"] = audit["label_meta"]["class_names"]
        best_payload["smoke_test"] = bool(args.smoke_test)
        write_json(str(Path(run_dir) / "best_metrics.json"), best_payload)
        test_predictions.to_csv(str(Path(run_dir) / "test_predictions.csv"), index=False)

        logger.log(f"Best epoch={best_payload['best_epoch']}")
        logger.log(f"Test metrics={json.dumps(test_metrics, ensure_ascii=False)}")
        update_run_state(
            run_dir,
            "completed",
            best_epoch=best_payload["best_epoch"],
            peak_gpu_mem_mb=peak_gpu_mem_mb,
        )
    except Exception as exc:
        update_run_state(run_dir, "failed", error=str(exc))
        raise
    finally:
        lock_guard.release()


if __name__ == "__main__":
    main()
