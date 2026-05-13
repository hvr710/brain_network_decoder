import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:
    Data = None
    PyGDataLoader = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_WEIGHT_DIR = (
    REPO_ROOT
    / "model_weights"
    / "none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC"
)
DEFAULT_BB_NAME = "bb_fold0_hcpaBest_2025-01-20-02-09-00-551325.pt"
DEFAULT_HEAD_NAME = "head_fold0_hcpaBest_2025-01-20-02-09-00-551325 .pt"


@dataclass
class AbideRecord:
    subject_id: str
    subject_num: int
    file_id: str
    site_id: str
    age: float
    gender: str
    dx_group: int
    dsm_iv_tr: str
    label: int
    label_name: str
    roi_path: str
    origin_split: str


def ensure_dir(path: os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def json_dump(obj, path: os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv_rows(path: os.PathLike, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_simple_yaml(path: os.PathLike, config: Dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                f.write(f"{key}: [{', '.join(map(str, value))}]\n")
            elif isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")


def subject_num_from_text(text: str) -> Optional[int]:
    matches = re.findall(r"(?<!\d)(\d{7})(?!\d)", str(text))
    if matches:
        return int(matches[-1])
    matches = re.findall(r"(?<!\d)(\d{5})(?!\d)", str(text))
    if matches:
        return int(matches[-1])
    return None


def normalize_subject(value) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return subject_num_from_text(text)


def parse_split_file(path: os.PathLike) -> List[int]:
    ids = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            sid = subject_num_from_text(line.strip())
            if sid is not None:
                ids.append(sid)
    return sorted(set(ids))


def load_fixed_splits(split_dir: os.PathLike) -> Dict[str, List[int]]:
    split_dir = Path(split_dir)
    splits = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[name] = parse_split_file(path)
    return splits


def load_abide_labels(label_csv: os.PathLike) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    label_csv = Path(label_csv)
    if not label_csv.exists():
        raise FileNotFoundError(f"Missing ABIDE label CSV: {label_csv}")
    df = pd.read_csv(label_csv)
    required = [
        "Subject",
        "SITE_ID",
        "FILE_ID",
        "DX_GROUP",
        "DSM_IV_TR",
        "age",
        "Gender",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"ABIDE CSV missing required columns: {missing}")

    rows: Dict[int, Dict] = {}
    for _, row in df.iterrows():
        subject_num = normalize_subject(row["Subject"])
        if subject_num is None:
            continue
        dx = int(row["DX_GROUP"])
        if dx == 0:
            label = 1
            label_name = "ASD"
        elif dx == 3:
            label = 0
            label_name = "Control"
        else:
            continue
        rows[subject_num] = {
            "subject_id": str(subject_num),
            "subject_num": subject_num,
            "file_id": str(row["FILE_ID"]),
            "site_id": str(row["SITE_ID"]),
            "age": float(row["age"]) if not pd.isna(row["age"]) else np.nan,
            "gender": str(row["Gender"]),
            "dx_group": dx,
            "dsm_iv_tr": str(row["DSM_IV_TR"]),
            "label": label,
            "label_name": label_name,
        }
    return df, rows


def index_roi_files(roi_dir: os.PathLike) -> Dict[int, Path]:
    roi_dir = Path(roi_dir)
    if not roi_dir.exists():
        raise FileNotFoundError(f"AAL ROI directory does not exist: {roi_dir}")
    supported = {".npy", ".npz", ".txt", ".csv", ".tsv"}
    paths = [p for p in roi_dir.iterdir() if p.is_file() and p.suffix.lower() in supported]
    index: Dict[int, Path] = {}
    duplicates: Dict[int, List[str]] = {}
    for path in paths:
        sid = subject_num_from_text(path.name)
        if sid is None:
            continue
        if sid in index:
            duplicates.setdefault(sid, [str(index[sid])]).append(str(path))
        else:
            index[sid] = path
    if duplicates:
        examples = list(duplicates.items())[:5]
        raise ValueError(f"Duplicate ROI files by subject id, examples: {examples}")
    return index


def load_abide_records(
    roi_dir: os.PathLike,
    label_csv: os.PathLike,
    split_dir: os.PathLike,
    out_dir: Optional[os.PathLike] = None,
) -> Tuple[List[AbideRecord], Dict[str, List[str]], Dict]:
    label_df, labels = load_abide_labels(label_csv)
    roi_index = index_roi_files(roi_dir)
    split_subjects = load_fixed_splits(split_dir)

    all_split_ids = []
    for ids in split_subjects.values():
        all_split_ids.extend(ids)
    if len(set(all_split_ids)) != len(all_split_ids):
        overlaps = {}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ov = sorted(set(split_subjects[a]).intersection(split_subjects[b]))
            if ov:
                overlaps[f"{a}_{b}"] = ov[:20]
        raise ValueError(f"Split subject overlap detected: {overlaps}")

    records = []
    split_map: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    missing_label = []
    missing_roi = []
    for split_name, ids in split_subjects.items():
        for sid in ids:
            if sid not in labels:
                missing_label.append(sid)
                continue
            if sid not in roi_index:
                missing_roi.append(sid)
                continue
            meta = labels[sid]
            rec = AbideRecord(
                subject_id=str(meta["subject_id"]),
                subject_num=sid,
                file_id=meta["file_id"],
                site_id=meta["site_id"],
                age=meta["age"],
                gender=meta["gender"],
                dx_group=meta["dx_group"],
                dsm_iv_tr=meta["dsm_iv_tr"],
                label=meta["label"],
                label_name=meta["label_name"],
                roi_path=str(roi_index[sid]),
                origin_split=split_name,
            )
            records.append(rec)
            split_map[split_name].append(rec.subject_id)

    if missing_label or missing_roi:
        raise ValueError(
            "Failed to match split subjects. "
            f"missing_label={missing_label[:20]}, missing_roi={missing_roi[:20]}"
        )
    if len(records) < 100:
        raise ValueError(f"Too few matched ABIDE subjects: {len(records)}")
    labels_present = sorted(set(r.label for r in records))
    if labels_present != [0, 1]:
        raise ValueError(f"Need both classes after matching, got labels={labels_present}")
    for split_name in ("train", "val", "test"):
        split_labels = sorted(set(r.label for r in records if r.origin_split == split_name))
        if split_labels != [0, 1]:
            raise ValueError(f"Split {split_name} lost a class: {split_labels}")

    dx_dsm = (
        label_df.groupby(["DX_GROUP", "DSM_IV_TR"]).size().reset_index(name="count").to_dict("records")
    )
    label_dist = {}
    for split_name in ("train", "val", "test"):
        one = [r.label for r in records if r.origin_split == split_name]
        label_dist[split_name] = {
            "Control_0": int(sum(1 for y in one if y == 0)),
            "ASD_1": int(sum(1 for y in one if y == 1)),
            "total": len(one),
        }
    data_check = {
        "roi_dir": str(roi_dir),
        "label_csv": str(label_csv),
        "split_dir": str(split_dir),
        "roi_file_count": len(roi_index),
        "csv_row_count": int(len(label_df)),
        "matched_subjects": len(records),
        "split_subject_counts": {k: len(v) for k, v in split_map.items()},
        "label_distribution": label_dist,
        "dx_group_to_label": {"0": "ASD -> 1", "3": "Control -> 0"},
        "dx_group_dsm_iv_tr_crosstab": dx_dsm,
        "example_records": [asdict(r) for r in records[:5]],
    }
    if out_dir is not None:
        json_dump(data_check, Path(out_dir) / "data_check.json")
        json_dump(split_map, Path(out_dir) / "fixed_crop_split_subjects.json")
    print(
        "Data check: "
        f"ROI files={len(roi_index)}, matched={len(records)}, "
        f"train/val/test={len(split_map['train'])}/{len(split_map['val'])}/{len(split_map['test'])}"
    )
    print(f"Label distribution: {label_dist}")
    return records, split_map, data_check


def load_timeseries(path: os.PathLike) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = data.files[0]
        arr = data[key]
    elif suffix in {".txt", ".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else None
        arr = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported ROI file format: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Time series must be 2D, got shape={arr.shape} for {path}")
    if arr.shape[1] == 116:
        ts = arr
    elif arr.shape[0] == 116:
        ts = arr.T
    else:
        raise ValueError(f"Cannot infer 116 ROI axis, got shape={arr.shape} for {path}")
    return np.nan_to_num(ts.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def compute_fc(ts: np.ndarray) -> np.ndarray:
    if ts.ndim != 2 or ts.shape[1] != 116:
        raise ValueError(f"Expected [T,116] time series, got {ts.shape}")
    fc = np.corrcoef(ts, rowvar=False).astype(np.float32)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    return fc


class AbideGraphDataset(Dataset):
    def __init__(self, records: Sequence[AbideRecord], fc_th: float = 0.5):
        if Data is None:
            raise ImportError("torch_geometric is required for LCM graph datasets.")
        self.records = list(records)
        self.fc_th = fc_th
        self.node_num = 116
        self._cache: List[Optional[Data]] = [None for _ in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Data:
        cached = self._cache[index]
        if cached is not None:
            return cached
        rec = self.records[index]
        ts = load_timeseries(rec.roi_path)
        fc = torch.from_numpy(compute_fc(ts)).float()
        edge_index_fc = torch.stack(torch.where(fc > self.fc_th))
        adj_fc = torch.zeros(self.node_num, self.node_num).bool()
        adj_fc[edge_index_fc[0], edge_index_fc[1]] = True
        adj_fc[torch.arange(self.node_num), torch.arange(self.node_num)] = True
        data = Data.from_dict(
            {
                "x": fc,
                "edge_index": edge_index_fc,
                "edge_index_fc": edge_index_fc,
                "edge_index_sc": edge_index_fc,
                "adj_fc": adj_fc[None],
                "adj_sc": adj_fc[None],
                "y": torch.tensor(rec.label, dtype=torch.long),
                "age": torch.tensor([[rec.age]], dtype=torch.float32),
                "sex": torch.tensor(int(float(rec.gender)) if str(rec.gender).replace(".", "", 1).isdigit() else -1),
                "sample_idx": torch.tensor(index, dtype=torch.long),
            }
        )
        self._cache[index] = data
        return data


def graph_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    if PyGDataLoader is None:
        raise ImportError("torch_geometric is required for LCM graph dataloaders.")
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_lcm_model(
    device: torch.device,
    hiddim: int = 2048,
    pretrained_nclass: int = 24,
    decoder_layer: int = 32,
):
    from models import brain_identity
    from models.heads import BNDecoder

    model = brain_identity.Identity(
        node_sz=116,
        out_channel=hiddim,
        in_channel=116,
        batch_size=1,
        device=str(device),
        nlayer=4,
        heads=8,
    ).to(device)
    classifier = BNDecoder(
        hiddim,
        nclass=pretrained_nclass,
        node_sz=116,
        nlayer=decoder_layer,
        head_num=8,
    ).to(device)
    return model, classifier


def torch_load(path: os.PathLike, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_pretrained_lcm(
    model,
    classifier,
    weight_dir: os.PathLike = DEFAULT_WEIGHT_DIR,
    bb_name: str = DEFAULT_BB_NAME,
    head_name: str = DEFAULT_HEAD_NAME,
) -> Tuple[Path, Path]:
    weight_dir = Path(weight_dir)
    bb_path = weight_dir / bb_name
    head_path = weight_dir / head_name
    print(f"Pretrained bb exists: {bb_path.exists()}  {bb_path}", flush=True)
    print(f"Pretrained head exists: {head_path.exists()}  {head_path}", flush=True)
    if not bb_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"Missing pretrained weights: bb={bb_path}, head={head_path}")
    print(f"Loading backbone checkpoint: {bb_path}", flush=True)
    model.load_state_dict(torch_load(bb_path, map_location="cpu"))
    print("Backbone checkpoint loaded.", flush=True)
    print(f"Loading decoder/head checkpoint: {head_path}", flush=True)
    try:
        classifier.load_state_dict(torch_load(head_path, map_location="cpu"), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load pretrained head strictly. Check --pretrained_nclass and decoder settings. "
            f"Original error: {exc}"
        )
    print("Decoder/head checkpoint loaded.", flush=True)
    return bb_path, head_path


def decoder_token_hidden(classifier, node_feat: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    x = node_feat.reshape(-1, classifier.node_sz, node_feat.shape[1])
    query_embed = classifier.object_query.weight
    if hasattr(classifier, "finetune_query"):
        query_embed = torch.cat([query_embed[:-3], classifier.finetune_query.weight, query_embed[-3:]])
    query_embed = query_embed.unsqueeze(0).repeat(len(x), 1, 1)
    tgt = torch.zeros_like(query_embed)
    hs = classifier.decoder(tgt, x, memory_key_padding_mask=None, query_pos=query_embed)
    if hs.dim() == 4:
        hs = hs[-1]
    token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=hs.device)
    hidden = hs.index_select(1, token_ids)
    return hidden.reshape(hidden.shape[0], -1)


def selected_token_logits(classifier, node_feat: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    x = node_feat.reshape(-1, classifier.node_sz, node_feat.shape[1])
    out = classifier(node_feat, edge_index=None, batch=None)["y"]
    token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=out.device)
    return out.index_select(-1, token_ids)


def layer_logits_for_loss(logits: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
    if logits.dim() == 3:
        if epoch > 5:
            batch_ids = torch.arange(logits.shape[1], device=logits.device)
            layer_ids = logits[:, batch_ids, target].argmax(0)
            return logits[layer_ids, batch_ids]
        return logits.mean(0)
    return logits


def layer_logits_for_eval(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits.max(0)[0]
    return logits


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = [0, 1]
    f1_each = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "control_f1": float(f1_each[0]),
        "asd_f1": float(f1_each[1]),
        "confusion_matrix": cm.tolist(),
    }


def metric_value(metrics: Dict, key: str):
    if key in metrics:
        return metrics[key]
    if key == "weighted_f1" and "confusion_matrix" in metrics:
        cm = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        support = cm.sum(axis=1)
        total = support.sum()
        if total <= 0:
            return None
        control_f1 = float(metrics.get("control_f1", 0.0))
        asd_f1 = float(metrics.get("asd_f1", 0.0))
        return float((support[0] * control_f1 + support[1] * asd_f1) / total)
    return None


def summarize_runs(
    runs: List[Dict],
    metric_keys: Sequence[str] = ("accuracy", "weighted_f1", "macro_f1", "asd_f1", "control_f1"),
) -> Dict:
    summary = {"runs": runs, "val_mean": {}, "val_std": {}, "test_mean": {}, "test_std": {}}
    for split in ("val", "test"):
        for key in metric_keys:
            vals = []
            for run in runs:
                if split not in run:
                    continue
                val = metric_value(run[split], key)
                if val is not None:
                    vals.append(float(val))
            summary[f"{split}_mean"][key] = float(np.mean(vals)) if vals else None
            summary[f"{split}_std"][key] = float(np.std(vals, ddof=0)) if vals else None
    return summary


def save_confusion_matrix(path: os.PathLike, cm: Sequence[Sequence[int]]) -> None:
    rows = [
        {"true_label": "Control", "pred_control": cm[0][0], "pred_asd": cm[0][1]},
        {"true_label": "ASD", "pred_control": cm[1][0], "pred_asd": cm[1][1]},
    ]
    write_csv_rows(path, rows, ["true_label", "pred_control", "pred_asd"])


def records_by_split(records: Sequence[AbideRecord]) -> Dict[str, List[AbideRecord]]:
    out = {"train": [], "val": [], "test": []}
    for rec in records:
        out[rec.origin_split].append(rec)
    return out


def feature_cache_path(out_dir: os.PathLike, feature_source: str) -> Path:
    name = "lcm_frozen_features.npz" if feature_source == "lcm_frozen" else "fc_vector_features.npz"
    return Path(out_dir) / "feature_cache" / name


def save_feature_cache(
    path: os.PathLike,
    X: np.ndarray,
    records: Sequence[AbideRecord],
    feature_source: str,
    pretrained_bb_path: str = "",
    pretrained_head_path: str = "",
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=np.asarray([r.label for r in records], dtype=np.int64),
        subject_id=np.asarray([r.subject_id for r in records]),
        file_id=np.asarray([r.file_id for r in records]),
        site_id=np.asarray([r.site_id for r in records]),
        age=np.asarray([r.age for r in records], dtype=np.float32),
        gender=np.asarray([r.gender for r in records]),
        origin_split=np.asarray([r.origin_split for r in records]),
        feature_source=np.asarray(feature_source),
        pretrained_bb_path=np.asarray(str(pretrained_bb_path)),
        pretrained_head_path=np.asarray(str(pretrained_head_path)),
    )


def load_feature_cache(path: os.PathLike) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def split_cache_arrays(cache: Dict[str, np.ndarray], split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = cache["origin_split"].astype(str) == split_name
    return cache["X"][mask], cache["y"][mask], cache["subject_id"][mask].astype(str)


def save_run_metrics(run_dir: os.PathLike, metrics: Dict) -> None:
    run_dir = ensure_dir(run_dir)
    json_dump(metrics, run_dir / "metrics.json")
    save_confusion_matrix(run_dir / "val_confusion_matrix.csv", metrics["val"]["confusion_matrix"])
    save_confusion_matrix(run_dir / "test_confusion_matrix.csv", metrics["test"]["confusion_matrix"])
