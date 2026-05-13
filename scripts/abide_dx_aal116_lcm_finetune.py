import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from abide_lcm_utils import (
    DEFAULT_BB_NAME,
    DEFAULT_HEAD_NAME,
    DEFAULT_WEIGHT_DIR,
    AbideGraphDataset,
    build_lcm_model,
    compute_metrics,
    ensure_dir,
    graph_loader,
    json_dump,
    layer_logits_for_eval,
    layer_logits_for_loss,
    load_abide_records,
    load_pretrained_lcm,
    records_by_split,
    resolve_device,
    save_confusion_matrix,
    set_seed,
    summarize_runs,
    selected_token_logits,
    write_csv_rows,
    write_simple_yaml,
)


def parse_args():
    parser = argparse.ArgumentParser("ABIDE dx AAL116 LCM full finetune")
    parser.add_argument(
        "--roi_dir",
        type=str,
        default="/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL",
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default="/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/data_csv/ABIDE.csv",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL_crop_split",
    )
    parser.add_argument("--out_dir", type=str, default="outputs/abide_dx_aal116_lcm_finetune")
    parser.add_argument("--seeds", type=int, nargs="*", default=[4, 44, 444])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--fc_th", type=float, default=0.5)
    parser.add_argument("--hiddim", type=int, default=2048)
    parser.add_argument("--pretrained_nclass", type=int, default=24)
    parser.add_argument("--decoder_layer", type=int, default=32)
    parser.add_argument("--diagnosis_token_ids", type=int, nargs=2, default=[6, 7])
    parser.add_argument("--weight_dir", type=str, default=str(DEFAULT_WEIGHT_DIR))
    parser.add_argument("--bb_name", type=str, default=DEFAULT_BB_NAME)
    parser.add_argument("--head_name", type=str, default=DEFAULT_HEAD_NAME)
    parser.add_argument("--best_metric", choices=["val_loss", "val_macro_f1"], default="val_loss")
    return parser.parse_args()


def forward_logits(model, classifier, batch, token_ids: Sequence[int]) -> torch.Tensor:
    node_feat = model(batch)
    return selected_token_logits(classifier, node_feat, token_ids)


def train_one_epoch(model, classifier, loader, optimizer, criterion, device, token_ids, epoch: int) -> Dict:
    model.train()
    classifier.train()
    losses = []
    y_true = []
    y_pred = []
    total = 0
    for batch in tqdm(loader, desc=f"Train epoch {epoch}", leave=False, mininterval=10):
        batch = batch.to(device)
        target = batch.y.long().view(-1)
        optimizer.zero_grad()
        logits_all_layers = forward_logits(model, classifier, batch, token_ids)
        logits = layer_logits_for_loss(logits_all_layers, target, epoch)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        batch_n = int(target.numel())
        losses.append(float(loss.detach().cpu().item()) * batch_n)
        total += batch_n
        pred = layer_logits_for_eval(logits_all_layers).argmax(1).detach().cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(target.detach().cpu().numpy().tolist())
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.sum(losses) / max(total, 1))
    return metrics


def evaluate(model, classifier, loader, criterion, device, token_ids) -> Dict:
    model.eval()
    classifier.eval()
    losses = []
    y_true = []
    y_pred = []
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False, mininterval=10):
            batch = batch.to(device)
            target = batch.y.long().view(-1)
            logits_all_layers = forward_logits(model, classifier, batch, token_ids)
            logits = layer_logits_for_eval(logits_all_layers)
            loss = criterion(logits, target)
            batch_n = int(target.numel())
            losses.append(float(loss.detach().cpu().item()) * batch_n)
            total += batch_n
            pred = logits.argmax(1).detach().cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(target.detach().cpu().numpy().tolist())
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.sum(losses) / max(total, 1))
    return metrics


def is_better(current: Dict, best: Dict, metric_name: str) -> bool:
    if best is None:
        return True
    if metric_name == "val_loss":
        return current["loss"] < best["loss"]
    return current["macro_f1"] > best["macro_f1"]


def save_checkpoint(path: Path, model, classifier, config: Dict, epoch: int, val_metrics: Dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
        },
        path,
    )


def load_checkpoint(path: Path, model, classifier, device) -> Dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    return ckpt


def run_one_seed(run_idx: int, seed: int, records_by_name: Dict[str, List], args) -> Dict:
    set_seed(seed)
    device = resolve_device(args.device)
    run_dir = ensure_dir(Path(args.out_dir) / f"run_{run_idx}_seed_{seed}")
    config = {
        "method": "LCM full finetune",
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pretrained_nclass": args.pretrained_nclass,
        "decoder_layer": args.decoder_layer,
        "diagnosis_token_ids": args.diagnosis_token_ids,
        "best_metric": args.best_metric,
        "label_mapping": "Control=0, ASD=1",
    }
    json_dump(config, run_dir / "config.json")
    write_simple_yaml(run_dir / "config.yaml", config)

    train_ds = AbideGraphDataset(records_by_name["train"], fc_th=args.fc_th)
    val_ds = AbideGraphDataset(records_by_name["val"], fc_th=args.fc_th)
    test_ds = AbideGraphDataset(records_by_name["test"], fc_th=args.fc_th)
    train_loader = graph_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = graph_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = graph_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, classifier = build_lcm_model(
        device=device,
        hiddim=args.hiddim,
        pretrained_nclass=args.pretrained_nclass,
        decoder_layer=args.decoder_layer,
    )
    bb_path, head_path = load_pretrained_lcm(model, classifier, args.weight_dir, args.bb_name, args.head_name)
    config["pretrained_bb_path"] = str(bb_path)
    config["pretrained_head_path"] = str(head_path)
    json_dump(config, run_dir / "config.json")

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_val = None
    best_epoch = 0
    best_path = run_dir / "best_checkpoint.pt"
    train_log = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, classifier, train_loader, optimizer, criterion, device, args.diagnosis_token_ids, epoch
        )
        val_metrics = evaluate(model, classifier, val_loader, criterion, device, args.diagnosis_token_ids)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_asd_f1": val_metrics["asd_f1"],
            "val_control_f1": val_metrics["control_f1"],
        }
        train_log.append(row)
        if is_better(val_metrics, best_val, args.best_metric):
            best_val = val_metrics
            best_epoch = epoch
            save_checkpoint(best_path, model, classifier, config, epoch, val_metrics)
        print(
            f"seed={seed} epoch={epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    write_csv_rows(run_dir / "train_log.csv", train_log)
    ckpt = load_checkpoint(best_path, model, classifier, device)
    test_metrics = evaluate(model, classifier, test_loader, criterion, device, args.diagnosis_token_ids)
    metrics = {
        "method": "LCM full finetune",
        "run_idx": run_idx,
        "seed": seed,
        "best_epoch": best_epoch,
        "val": best_val,
        "test": test_metrics,
        "pretrained_bb_path": str(bb_path),
        "pretrained_head_path": str(head_path),
    }
    json_dump(metrics, run_dir / "metrics.json")
    save_confusion_matrix(run_dir / "val_confusion_matrix.csv", best_val["confusion_matrix"])
    save_confusion_matrix(run_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"])
    print(
        f"finetune run={run_idx} seed={seed} best_epoch={best_epoch} "
        f"test_acc={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    return metrics


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    print(f"Finetune seeds: {args.seeds}")
    records, _, _ = load_abide_records(args.roi_dir, args.label_csv, args.split_dir, out_dir)
    by_split = records_by_split(records)
    print(
        "Finetune fixed splits: "
        f"train={len(by_split['train'])}, val={len(by_split['val'])}, test={len(by_split['test'])}"
    )
    runs = []
    for run_idx, seed in enumerate(args.seeds):
        runs.append(run_one_seed(run_idx, seed, by_split, args))
    summary = summarize_runs(runs)
    summary["method"] = "LCM full finetune"
    summary["seeds"] = list(args.seeds)
    json_dump(summary, out_dir / "summary.json")
    rows = []
    for run in runs:
        rows.append(
            {
                "run_idx": run["run_idx"],
                "seed": run["seed"],
                "best_epoch": run["best_epoch"],
                "val_accuracy": run["val"]["accuracy"],
                "val_weighted_f1": run["val"]["weighted_f1"],
                "val_macro_f1": run["val"]["macro_f1"],
                "val_asd_f1": run["val"]["asd_f1"],
                "val_control_f1": run["val"]["control_f1"],
                "test_accuracy": run["test"]["accuracy"],
                "test_weighted_f1": run["test"]["weighted_f1"],
                "test_macro_f1": run["test"]["macro_f1"],
                "test_asd_f1": run["test"]["asd_f1"],
                "test_control_f1": run["test"]["control_f1"],
            }
        )
    write_csv_rows(out_dir / "summary.csv", rows)


if __name__ == "__main__":
    main()
