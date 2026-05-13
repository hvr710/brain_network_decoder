import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from abide_lcm_utils import (
    DEFAULT_BB_NAME,
    DEFAULT_HEAD_NAME,
    DEFAULT_WEIGHT_DIR,
    AbideRecord,
    AbideGraphDataset,
    build_lcm_model,
    compute_fc,
    compute_metrics,
    decoder_token_hidden,
    ensure_dir,
    feature_cache_path,
    graph_loader,
    json_dump,
    load_abide_records,
    load_feature_cache,
    load_pretrained_lcm,
    load_timeseries,
    records_by_split,
    resolve_device,
    save_feature_cache,
    save_run_metrics,
    set_seed,
    split_cache_arrays,
    summarize_runs,
    write_csv_rows,
    write_simple_yaml,
)


def parse_args():
    parser = argparse.ArgumentParser("ABIDE dx AAL116 LCM frozen feature probes")
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
    parser.add_argument("--out_dir", type=str, default="outputs/abide_dx_aal116_lcm_probe")
    parser.add_argument("--feature_source", choices=["lcm_frozen", "fc_vector"], default="lcm_frozen")
    parser.add_argument("--mode", choices=["lp", "mlp", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--base_seed", type=int, default=23)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed_stride", type=int, default=1)
    parser.add_argument("--recompute_features", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--feature_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fc_th", type=float, default=0.5)
    parser.add_argument("--hiddim", type=int, default=2048)
    parser.add_argument("--pretrained_nclass", type=int, default=24)
    parser.add_argument("--decoder_layer", type=int, default=32)
    parser.add_argument("--diagnosis_token_ids", type=int, nargs=2, default=[6, 7])
    parser.add_argument("--weight_dir", type=str, default=str(DEFAULT_WEIGHT_DIR))
    parser.add_argument("--bb_name", type=str, default=DEFAULT_BB_NAME)
    parser.add_argument("--head_name", type=str, default=DEFAULT_HEAD_NAME)
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_num_layers", type=int, default=2)
    parser.add_argument("--mlp_input_norm", type=str, default="layernorm")
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_max_epochs", type=int, default=200)
    parser.add_argument("--mlp_patience", type=int, default=30)
    parser.add_argument("--mlp_batch_size", type=int, default=32)
    return parser.parse_args()


def seeds_from_args(args) -> List[int]:
    if args.seeds:
        return list(args.seeds)
    return [args.base_seed + i * args.seed_stride for i in range(args.num_runs)]


def extract_fc_vector_features(records: Sequence[AbideRecord]) -> np.ndarray:
    idx = np.triu_indices(116, k=1)
    feats = []
    for i, rec in enumerate(records):
        ts = load_timeseries(rec.roi_path)
        fc = compute_fc(ts)
        feats.append(fc[idx])
        if i == 0:
            print(f"FC input shape: timeseries={ts.shape}, fc={fc.shape}, vector_dim={len(idx[0])}")
    X = np.stack(feats).astype(np.float32)
    print(f"Extracted fc_vector feature shape: {X.shape}")
    return X


def extract_lcm_frozen_features(records: Sequence[AbideRecord], args) -> Tuple[np.ndarray, str, str]:
    device = resolve_device(args.device)
    dataset = AbideGraphDataset(records, fc_th=args.fc_th)
    loader = graph_loader(
        dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model, classifier = build_lcm_model(
        device=device,
        hiddim=args.hiddim,
        pretrained_nclass=args.pretrained_nclass,
        decoder_layer=args.decoder_layer,
    )
    bb_path, head_path = load_pretrained_lcm(model, classifier, args.weight_dir, args.bb_name, args.head_name)
    model.eval()
    classifier.eval()
    for param in list(model.parameters()) + list(classifier.parameters()):
        param.requires_grad_(False)

    feats = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Extract lcm_frozen features", mininterval=5)):
            batch = batch.to(device)
            if step == 0:
                print(f"LCM graph batch input x shape: {tuple(batch.x.shape)}", flush=True)
            node_feat = model(batch)
            hidden = decoder_token_hidden(classifier, node_feat, args.diagnosis_token_ids)
            feats.append(hidden.detach().cpu().numpy())
    X = np.concatenate(feats, axis=0).astype(np.float32)
    print(f"Extracted lcm_frozen feature shape: {X.shape}", flush=True)
    return X, str(bb_path), str(head_path)


def get_or_create_feature_cache(records: Sequence[AbideRecord], args) -> Dict[str, np.ndarray]:
    cache_path = feature_cache_path(args.out_dir, args.feature_source)
    if cache_path.exists() and not args.recompute_features:
        print(f"Reuse feature cache: {cache_path}")
        return load_feature_cache(cache_path)

    print(f"Create feature cache: {cache_path}")
    if args.feature_source == "fc_vector":
        X = extract_fc_vector_features(records)
        bb_path = ""
        head_path = ""
    else:
        X, bb_path, head_path = extract_lcm_frozen_features(records, args)
    save_feature_cache(cache_path, X, records, args.feature_source, bb_path, head_path)
    return load_feature_cache(cache_path)


def subject_level_eval_split(
    eval_subjects: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    unique_subjects = sorted(set(eval_subjects.astype(str).tolist()))
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique_subjects, dtype=object)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    val_subjects = set(str(x) for x in shuffled[:half])
    test_subjects = set(str(x) for x in shuffled[half:])
    val_idx = np.asarray([str(s) in val_subjects for s in eval_subjects], dtype=bool)
    test_idx = np.asarray([str(s) in test_subjects for s in eval_subjects], dtype=bool)
    return val_idx, test_idx, sorted(val_subjects), sorted(test_subjects)


def assert_two_classes(name: str, y: np.ndarray) -> None:
    labels = sorted(set(np.asarray(y).astype(int).tolist()))
    if labels != [0, 1]:
        raise ValueError(f"{name} must contain both classes, got {labels}")


def run_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    run_dir: Path,
) -> Dict:
    clf = LogisticRegression(C=1.0, max_iter=8000, random_state=seed)
    clf.fit(X_train, y_train)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    with (ckpt_dir / "logistic_regression.pkl").open("wb") as f:
        pickle.dump(clf, f)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    return {
        "val": compute_metrics(y_val, val_pred),
        "test": compute_metrics(y_test, test_pred),
        "best_epoch": None,
    }


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int, dropout: float, num_layers: int, input_norm: str):
    num_layers = max(int(num_layers), 1)
    input_norm = str(input_norm).lower()
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


def standardize_from_train(X_train: np.ndarray, *arrays: np.ndarray):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    out = [(arr - mean) / std for arr in arrays]
    return out, mean, std


def eval_mlp(model, X: np.ndarray, y: np.ndarray, criterion, device: torch.device) -> Tuple[float, Dict]:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X.astype(np.float32)).to(device)
        yb = torch.from_numpy(y.astype(np.int64)).to(device)
        logits = model(xb)
        loss = criterion(logits, yb).item()
        pred = logits.argmax(1).detach().cpu().numpy()
    return loss, compute_metrics(y, pred)


def run_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    args,
    run_dir: Path,
) -> Dict:
    set_seed(seed)
    device = resolve_device(args.device)
    (X_train_s, X_val_s, X_test_s), mean, std = standardize_from_train(X_train, X_train, X_val, X_test)
    model = build_mlp(
        input_dim=X_train.shape[1],
        output_dim=2,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
        num_layers=args.mlp_num_layers,
        input_norm=args.mlp_input_norm,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    criterion = nn.CrossEntropyLoss()
    ds = TensorDataset(
        torch.from_numpy(X_train_s.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(ds, batch_size=args.mlp_batch_size, shuffle=True, generator=generator)

    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    train_log = []
    best = {"val_loss": float("inf"), "epoch": 0, "val": None, "test": None}
    patience_left = args.mlp_patience

    for epoch in range(1, args.mlp_max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_loss, val_metrics = eval_mlp(model, X_val_s, y_val, criterion, device)
        test_loss, test_metrics = eval_mlp(model, X_test_s, y_test, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "test_loss": test_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
        }
        train_log.append(row)
        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "test_loss": test_loss,
                "epoch": epoch,
                "val": val_metrics,
                "test": test_metrics,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "feature_mean": mean.astype(np.float32),
                    "feature_std": std.astype(np.float32),
                    "config": vars(args),
                },
                ckpt_dir / "best_model.pt",
            )
            patience_left = args.mlp_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    write_csv_rows(run_dir / "train_log.csv", train_log)
    return {
        "val": best["val"],
        "test": best["test"],
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "test_loss": best["test_loss"],
    }


def run_probe_method(method: str, cache: Dict[str, np.ndarray], seeds: Sequence[int], args) -> Dict:
    X_train, y_train, train_subjects = split_cache_arrays(cache, "train")
    X_orig_val, y_orig_val, val_subjects = split_cache_arrays(cache, "val")
    X_orig_test, y_orig_test, test_subjects = split_cache_arrays(cache, "test")
    assert_two_classes("fixed train", y_train)
    eval_X = np.concatenate([X_orig_val, X_orig_test], axis=0)
    eval_y = np.concatenate([y_orig_val, y_orig_test], axis=0)
    eval_subjects = np.concatenate([val_subjects, test_subjects], axis=0)

    method_dir = ensure_dir(Path(args.out_dir) / args.feature_source / method)
    runs = []
    for run_idx, seed in enumerate(seeds):
        run_dir = ensure_dir(method_dir / f"run_{run_idx}_seed_{seed}")
        val_idx, test_idx, run_val_subjects, run_test_subjects = subject_level_eval_split(eval_subjects, seed)
        X_val, y_val = eval_X[val_idx], eval_y[val_idx]
        X_test, y_test = eval_X[test_idx], eval_y[test_idx]
        assert_two_classes(f"{method} run {run_idx} val", y_val)
        assert_two_classes(f"{method} run {run_idx} test", y_test)
        split_payload = {
            "run_idx": run_idx,
            "seed": seed,
            "train_subjects": sorted(set(train_subjects.astype(str).tolist())),
            "val_subjects": run_val_subjects,
            "test_subjects": run_test_subjects,
            "counts": {
                "train": int(len(train_subjects)),
                "val": int(len(run_val_subjects)),
                "test": int(len(run_test_subjects)),
            },
        }
        json_dump(split_payload, run_dir / "val_test_subject_split.json")
        config = {
            "method": method,
            "feature_source": args.feature_source,
            "seed": seed,
            "train_subjects": len(train_subjects),
            "val_subjects": len(run_val_subjects),
            "test_subjects": len(run_test_subjects),
            "feature_dim": int(X_train.shape[1]),
        }
        write_simple_yaml(run_dir / "config.yaml", config)
        if method == "lp":
            result = run_linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, seed, run_dir)
        else:
            result = run_mlp_probe(X_train, y_train, X_val, y_val, X_test, y_test, seed, args, run_dir)
        metrics = {
            "method": "LCM frozen + LP" if method == "lp" else "LCM frozen + MLP",
            "feature_source": args.feature_source,
            "run_idx": run_idx,
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "val": result["val"],
            "test": result["test"],
        }
        save_run_metrics(run_dir, metrics)
        runs.append(metrics)
        print(
            f"{method} run={run_idx} seed={seed} "
            f"val_acc={metrics['val']['accuracy']:.4f} test_acc={metrics['test']['accuracy']:.4f} "
            f"test_macro_f1={metrics['test']['macro_f1']:.4f}"
        )

    summary = summarize_runs(runs)
    summary["method"] = "LCM frozen + LP" if method == "lp" else "LCM frozen + MLP"
    summary["feature_source"] = args.feature_source
    summary["seeds"] = list(seeds)
    json_dump(summary, method_dir / "summary.json")
    flat_rows = []
    for run in runs:
        flat_rows.append(
            {
                "run_idx": run["run_idx"],
                "seed": run["seed"],
                "best_epoch": run["best_epoch"] if run["best_epoch"] is not None else "",
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
    write_csv_rows(method_dir / "summary.csv", flat_rows)
    return summary


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    seeds = seeds_from_args(args)
    print(f"Probe seeds: {seeds}")
    records, _, _ = load_abide_records(args.roi_dir, args.label_csv, args.split_dir, out_dir)
    by_split = records_by_split(records)
    print(
        "Fixed feature extraction splits: "
        f"train={len(by_split['train'])}, orig_val={len(by_split['val'])}, orig_test={len(by_split['test'])}"
    )
    cache = get_or_create_feature_cache(records, args)
    print(f"Feature cache loaded: X={cache['X'].shape}, y={cache['y'].shape}")

    modes = ["lp", "mlp"] if args.mode == "both" else [args.mode]
    all_summaries = []
    for method in modes:
        all_summaries.append(run_probe_method(method, cache, seeds, args))
    json_dump({"summaries": all_summaries}, out_dir / "probe_summary.json")


if __name__ == "__main__":
    main()
