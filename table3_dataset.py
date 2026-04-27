import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError:
    Data = None
    DataLoader = None

from table3_utils import canonical_label_key, resolve_path


AUDIT_DETAIL_LIMIT = 25


@dataclass
class SampleRecord:
    sample_id: str
    subject_id: str
    file_path: str
    split: str
    raw_target: Any
    target: Any


def _safe_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(str(value).strip())


def _preview(values: Iterable[Any], limit: int = AUDIT_DETAIL_LIMIT) -> List[str]:
    return [str(value) for value in list(values)[:limit]]


def normalize_sample_id(name: str, rule: str) -> str:
    basename = Path(str(name).replace("\\", "/")).name
    stem = Path(basename).stem
    if rule == "abide_file_id":
        return stem.split("_func_minimal")[0]
    if rule == "abide_crop_subject":
        stem = re.sub(r"_seg\d+$", "", stem)
        return stem.split("_func_minimal")[0]
    if rule == "nki_subject":
        match = re.search(r"A0*(\d+)", stem)
        if not match:
            raise ValueError(f"Cannot parse NKI subject from: {name}")
        return str(int(match.group(1)))
    if rule == "numeric_stem":
        digits = re.sub(r"\D", "", stem)
        if not digits:
            raise ValueError(f"Cannot parse numeric stem from: {name}")
        return str(int(digits))
    if rule == "sub_numeric":
        match = re.search(r"sub-(\d+)", stem)
        if not match:
            raise ValueError(f"Cannot parse sub numeric id from: {name}")
        return str(int(match.group(1)))
    if rule == "abcd_subject":
        match = re.search(r"(sub-NDARINV[0-9A-Z]+)", stem)
        if not match:
            raise ValueError(f"Cannot parse ABCD subject from: {name}")
        return match.group(1)
    if rule == "hcp_subject":
        return stem.split("__")[0]
    if rule == "ppmi_subject":
        match = re.search(r"sub-(\d+)", stem)
        if not match:
            raise ValueError(f"Cannot parse PPMI subject from: {name}")
        return str(int(match.group(1)))
    if rule == "adni_subject_code":
        match = re.search(r"([0-9]{3}S[0-9]{4}[0-9]?)", stem)
        if not match:
            raise ValueError(f"Cannot parse ADNI subject code from: {name}")
        return match.group(1)
    if rule == "identity":
        return stem
    raise KeyError(f"Unknown id rule: {rule}")


def _aggregate_values(values: Iterable[Any], agg: str) -> Any:
    cleaned = [value for value in values if pd.notna(value)]
    if not cleaned:
        return None
    if agg == "first_non_null":
        return cleaned[0]
    if agg == "last_non_null":
        return cleaned[-1]
    if agg == "min":
        return min(cleaned)
    if agg == "max":
        return max(cleaned)
    if agg == "mean":
        return float(np.mean([_safe_float(value) for value in cleaned]))
    if agg == "mode":
        return pd.Series(cleaned).mode(dropna=True).iloc[0]
    raise KeyError(f"Unknown label aggregation: {agg}")


def _fill_derived_target_values(frame: pd.DataFrame, task_cfg: Dict[str, Any]) -> pd.DataFrame:
    target_column = task_cfg["target_column"]

    if target_column == "education_group" and "participant_education" in frame.columns:
        education_group_map = {
            "Kindergarten": 0,
            "1st Grade": 0,
            "2nd Grade": 0,
            "3rd Grade": 0,
            "4th Grade": 0,
            "5th Grade": 0,
            "6th Grade": 0,
            "7th Grade": 1,
            "8th Grade": 1,
            "9th Grade": 1,
            "10th Grade": 1,
            "11th Grade": 1,
            "12th Grade": 1,
            "High School Diploma": 1,
            "Some College": 1,
            "Bachelor's Degree": 2,
            "Graduate Degree": 2,
        }
        frame = frame.copy()
        frame[target_column] = frame[target_column].fillna(
            frame["participant_education"].map(education_group_map)
        )

    return frame


def _build_label_lookup(task_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    label_path = resolve_path(task_cfg["label_source"])
    label_format = task_cfg.get("label_format", "csv")
    if label_format == "csv":
        frame = pd.read_csv(label_path)
    elif label_format == "xlsx":
        frame = pd.read_excel(label_path)
    else:
        raise KeyError(f"Unsupported label format: {label_format}")

    target_column = task_cfg["target_column"]
    label_subject_column = task_cfg["label_subject_column"]
    label_id_rule = task_cfg.get("label_id_rule", task_cfg["id_rule"])
    label_agg = task_cfg.get("label_agg", "first_non_null")
    label_filters = task_cfg.get("label_filters", {})

    frame = _fill_derived_target_values(frame, task_cfg)

    for column, allowed in label_filters.items():
        frame = frame[frame[column].isin(allowed)]
    frame = frame.dropna(subset=[label_subject_column, target_column]).copy()
    frame["_normalized_subject"] = frame[label_subject_column].apply(
        lambda value: normalize_sample_id(str(value), label_id_rule)
    )

    grouped = defaultdict(list)
    for subject_id, target_value in zip(frame["_normalized_subject"], frame[target_column]):
        grouped[subject_id].append(target_value)

    lookup: Dict[str, Any] = {}
    ambiguous_subjects = 0
    for subject_id, values in grouped.items():
        unique_values = {
            canonical_label_key(value)
            for value in values
            if pd.notna(value)
        }
        if len(unique_values) > 1:
            ambiguous_subjects += 1
        aggregated = _aggregate_values(values, label_agg)
        if aggregated is not None:
            lookup[subject_id] = aggregated

    class_map = task_cfg.get("class_map")
    if task_cfg["task_type"] == "classification":
        if class_map:
            normalized_map = {
                canonical_label_key(key): int(value)
                for key, value in class_map.items()
            }
        else:
            normalized_map = {
                key: index
                for index, key in enumerate(
                    sorted({canonical_label_key(value) for value in lookup.values()})
                )
            }
        encoded_lookup = {}
        for subject_id, raw_value in lookup.items():
            key = canonical_label_key(raw_value)
            if key in normalized_map:
                encoded_lookup[subject_id] = normalized_map[key]
        lookup = encoded_lookup
        class_names = [
            key for key, _ in sorted(normalized_map.items(), key=lambda item: item[1])
        ]
    else:
        class_names = []
        lookup = {subject_id: float(value) for subject_id, value in lookup.items()}

    label_meta = {
        "label_path": label_path,
        "raw_rows": int(len(frame)),
        "subjects_with_label": int(len(grouped)),
        "ambiguous_subjects": ambiguous_subjects,
        "class_names": class_names,
    }
    return lookup, label_meta


def _scan_data_index(task_cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    roots = task_cfg["data_root"]
    if isinstance(roots, str):
        roots = [roots]
    elif isinstance(roots, dict):
        roots = list(roots.values())
    elif not isinstance(roots, list):
        raise TypeError(f"Unsupported data_root type: {type(roots)}")

    index: Dict[str, List[str]] = defaultdict(list)
    for root in roots:
        root_path = resolve_path(root)
        for file_path in sorted(Path(root_path).glob("*.npy")):
            subject_id = normalize_sample_id(file_path.name, task_cfg["id_rule"])
            index[subject_id].append(str(file_path))
    return index


def _read_txt_lines(path: str) -> List[str]:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def _load_split_members(task_cfg: Dict[str, Any], fold: int) -> Dict[str, List[Any]]:
    split_format = task_cfg["split_format"]
    split_source = task_cfg["split_source"]
    split_id_rule = task_cfg.get("split_id_rule", task_cfg["id_rule"])

    if split_format in {"crop_txt_to_subject", "txt_basename", "txt_subject_codes"}:
        fold_source = split_source if "train" in split_source else split_source[f"fold{fold}"]
        output = {}
        for split_name, split_path in fold_source.items():
            lines = _read_txt_lines(split_path)
            if split_format == "txt_subject_codes":
                output[split_name] = [normalize_sample_id(line, split_id_rule) for line in lines]
            else:
                output[split_name] = [
                    normalize_sample_id(Path(line).name, split_id_rule)
                    for line in lines
                ]
        return output

    if split_format in {"directory_files", "ref_dir_subject"}:
        output = {}
        for split_name, split_path in split_source.items():
            directory = Path(resolve_path(split_path))
            files = sorted(directory.glob("*.npy"))
            if split_format == "directory_files":
                output[split_name] = [str(file_path) for file_path in files]
            else:
                output[split_name] = [
                    normalize_sample_id(file_path.name, split_id_rule)
                    for file_path in files
                ]
        return output

    raise KeyError(f"Unsupported split format: {split_format}")


def _build_records_for_subject_task(
    task_cfg: Dict[str, Any],
    fold: int,
    label_lookup: Dict[str, Any],
) -> Tuple[Dict[str, List[SampleRecord]], Dict[str, Any]]:
    split_members = _load_split_members(task_cfg, fold)
    data_index = _scan_data_index(task_cfg)

    records: Dict[str, List[SampleRecord]] = {}
    audit = {
        "source_members": {},
        "missing_files": {},
        "missing_files_examples": {},
        "missing_labels": {},
        "missing_labels_examples": {},
        "duplicate_files": {},
        "duplicate_file_subject_examples": {},
    }
    for split_name, subject_ids in split_members.items():
        records[split_name] = []
        missing_files = []
        missing_labels = []
        duplicate_files = 0
        duplicate_file_subjects = []
        unique_subject_ids = list(dict.fromkeys(subject_ids))
        audit["source_members"][split_name] = {
            "raw_members": len(subject_ids),
            "unique_subjects": len(unique_subject_ids),
            "duplicates_removed": len(subject_ids) - len(unique_subject_ids),
        }
        for subject_id in unique_subject_ids:
            candidates = data_index.get(subject_id, [])
            if not candidates:
                missing_files.append(subject_id)
                continue
            if len(candidates) > 1:
                duplicate_files += len(candidates) - 1
                duplicate_file_subjects.append(subject_id)
            if subject_id not in label_lookup:
                missing_labels.append(subject_id)
                continue
            file_path = sorted(candidates)[0]
            records[split_name].append(
                SampleRecord(
                    sample_id=Path(file_path).stem,
                    subject_id=subject_id,
                    file_path=file_path,
                    split=split_name,
                    raw_target=label_lookup[subject_id],
                    target=label_lookup[subject_id],
                )
            )
        audit["missing_files"][split_name] = len(missing_files)
        audit["missing_files_examples"][split_name] = _preview(missing_files)
        audit["missing_labels"][split_name] = len(missing_labels)
        audit["missing_labels_examples"][split_name] = _preview(missing_labels)
        audit["duplicate_files"][split_name] = duplicate_files
        audit["duplicate_file_subject_examples"][split_name] = _preview(duplicate_file_subjects)
    return records, audit


def _build_records_for_file_task(
    task_cfg: Dict[str, Any],
    fold: int,
    label_lookup: Dict[str, Any],
) -> Tuple[Dict[str, List[SampleRecord]], Dict[str, Any]]:
    split_members = _load_split_members(task_cfg, fold)
    records: Dict[str, List[SampleRecord]] = {}
    audit = {
        "source_files": {},
        "missing_labels": {},
        "missing_label_file_examples": {},
        "missing_label_subject_examples": {},
        "invalid_file_names": {},
        "invalid_file_name_examples": {},
    }
    for split_name, file_paths in split_members.items():
        records[split_name] = []
        missing_labels = []
        missing_label_subjects = []
        invalid_file_names = []
        parsed_subjects = []
        for file_path in file_paths:
            try:
                subject_id = normalize_sample_id(Path(file_path).name, task_cfg["id_rule"])
            except ValueError:
                invalid_file_names.append(Path(file_path).name)
                continue
            parsed_subjects.append(subject_id)
            if subject_id not in label_lookup:
                missing_labels.append(Path(file_path).name)
                missing_label_subjects.append(subject_id)
                continue
            file_path = resolve_path(file_path)
            records[split_name].append(
                SampleRecord(
                    sample_id=Path(file_path).stem,
                    subject_id=subject_id,
                    file_path=file_path,
                    split=split_name,
                    raw_target=label_lookup[subject_id],
                    target=label_lookup[subject_id],
                )
            )
        audit["source_files"][split_name] = {
            "raw_files": len(file_paths),
            "valid_file_names": len(parsed_subjects),
            "unique_subjects_before_label_join": len(set(parsed_subjects)),
            "unique_subjects_after_label_join": len({record.subject_id for record in records[split_name]}),
        }
        audit["missing_labels"][split_name] = len(missing_labels)
        audit["missing_label_file_examples"][split_name] = _preview(missing_labels)
        audit["missing_label_subject_examples"][split_name] = _preview(dict.fromkeys(missing_label_subjects))
        audit["invalid_file_names"][split_name] = len(invalid_file_names)
        audit["invalid_file_name_examples"][split_name] = _preview(invalid_file_names)
    return records, audit


def build_task_records(task_cfg: Dict[str, Any], fold: int) -> Tuple[Dict[str, List[SampleRecord]], Dict[str, Any]]:
    label_lookup, label_meta = _build_label_lookup(task_cfg)
    if task_cfg["sample_level"] == "subject":
        records, split_audit = _build_records_for_subject_task(task_cfg, fold, label_lookup)
    elif task_cfg["sample_level"] == "file":
        records, split_audit = _build_records_for_file_task(task_cfg, fold, label_lookup)
    else:
        raise KeyError(f"Unsupported sample_level: {task_cfg['sample_level']}")

    subject_sets = {
        split_name: {record.subject_id for record in split_records}
        for split_name, split_records in records.items()
    }
    sample_sets = {
        split_name: {record.sample_id for record in split_records}
        for split_name, split_records in records.items()
    }
    splits = list(records.keys())
    disjoint_subjects = True
    disjoint_samples = True
    for index, split_name in enumerate(splits):
        for other in splits[index + 1 :]:
            disjoint_subjects = disjoint_subjects and subject_sets[split_name].isdisjoint(subject_sets[other])
            disjoint_samples = disjoint_samples and sample_sets[split_name].isdisjoint(sample_sets[other])

    target_counter = {
        split_name: dict(Counter(record.target for record in split_records))
        for split_name, split_records in records.items()
        if task_cfg["task_type"] == "classification"
    }
    audit = {
        "task_id": task_cfg["task_id"],
        "fold": fold,
        "label_meta": label_meta,
        "split_counts": {
            split_name: {
                "samples": len(split_records),
                "subjects": len(subject_sets[split_name]),
            }
            for split_name, split_records in records.items()
        },
        "target_counter": target_counter,
        "disjoint_subjects": disjoint_subjects,
        "disjoint_samples": disjoint_samples,
        "split_audit": split_audit,
    }
    return records, audit


class Table3Dataset(Dataset):
    def __init__(
        self,
        samples: List[SampleRecord],
        *,
        task_cfg: Dict[str, Any],
        node_attr: str = "FC",
        adj_type: str = "FC",
        fc_th: float = 0.5,
        bold_winsize: int = 500,
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.task_cfg = task_cfg
        self.node_attr = node_attr
        self.adj_type = adj_type
        self.fc_th = fc_th
        self.bold_winsize = bold_winsize
        self.transform = transform
        self.node_num = 116
        self.cached: List[Optional[Data]] = [None for _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_timeseries(self, file_path: str) -> torch.Tensor:
        timeseries = np.load(file_path).astype(np.float32)
        if timeseries.ndim != 2:
            raise ValueError(f"Expected 2D timeseries, got {timeseries.shape} from {file_path}")
        if timeseries.shape[1] == self.node_num:
            timeseries = timeseries.T
        elif timeseries.shape[0] != self.node_num:
            raise ValueError(f"Expected ROI dim 116 in {file_path}, got {timeseries.shape}")
        tensor = torch.from_numpy(timeseries)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def __getitem__(self, index: int) -> Data:
        if Data is None:
            raise ModuleNotFoundError("torch_geometric is required to materialize training samples.")
        if self.cached[index] is None:
            sample = self.samples[index]
            bold = self._load_timeseries(sample.file_path)
            fc = torch.corrcoef(bold)
            fc = torch.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            edge_index_fc = torch.stack(torch.where(fc > self.fc_th))
            edge_index = edge_index_fc

            if self.node_attr == "FC":
                x = fc
            elif self.node_attr == "BOLD":
                x = bold[:, : self.bold_winsize]
                if x.shape[1] < self.bold_winsize:
                    x = torch.cat(
                        [x, torch.zeros(x.shape[0], self.bold_winsize - x.shape[1])],
                        dim=1,
                    )
            elif self.node_attr == "ID":
                x = torch.arange(self.node_num).float()[:, None]
            else:
                raise KeyError(f"Unsupported node_attr: {self.node_attr}")

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            adj = torch.zeros(self.node_num, self.node_num, dtype=torch.bool)
            adj[edge_index_fc[0], edge_index_fc[1]] = True
            adj[torch.arange(self.node_num), torch.arange(self.node_num)] = True

            raw_target = sample.target
            if self.task_cfg["target_key"] == "y":
                y = torch.tensor(int(raw_target), dtype=torch.long)
                sex = torch.tensor(-1, dtype=torch.long)
                age = torch.tensor(-1.0, dtype=torch.float32)
            elif self.task_cfg["target_key"] == "sex":
                y = torch.tensor(-1, dtype=torch.long)
                sex = torch.tensor(int(raw_target), dtype=torch.long)
                age = torch.tensor(-1.0, dtype=torch.float32)
            elif self.task_cfg["target_key"] == "age":
                y = torch.tensor(-1, dtype=torch.long)
                sex = torch.tensor(-1, dtype=torch.long)
                age = torch.tensor(float(raw_target), dtype=torch.float32)
            else:
                raise KeyError(f"Unsupported target_key: {self.task_cfg['target_key']}")

            payload = {
                "edge_index": edge_index,
                "edge_index_fc": edge_index_fc,
                "edge_index_sc": edge_index_fc,
                "x": x,
                "y": y,
                "sex": sex,
                "age": age,
                "adj_fc": adj[None],
                "adj_sc": adj[None],
                # Avoid PyG auto-increment on batch collate for keys containing "index".
                "sample_idx": torch.tensor(index, dtype=torch.long),
            }
            data = Data.from_dict(payload)
            if self.transform is not None:
                transformed = self.transform(data)
                for key in transformed.keys():
                    data[key] = transformed[key]
            self.cached[index] = data
        return self.cached[index]


def make_dataloaders(
    task_cfg: Dict[str, Any],
    fold: int,
    *,
    batch_size: int,
    node_attr: str,
    adj_type: str,
    fc_th: float,
    bold_winsize: int,
    transform: Optional[Any],
    num_workers: int,
) -> Tuple[Dict[str, Table3Dataset], Dict[str, DataLoader], Dict[str, Any]]:
    if DataLoader is None:
        raise ModuleNotFoundError("torch_geometric is required to create training dataloaders.")
    records, audit = build_task_records(task_cfg, fold)
    datasets = {
        split_name: Table3Dataset(
            split_records,
            task_cfg=task_cfg,
            node_attr=node_attr,
            adj_type=adj_type,
            fc_th=fc_th,
            bold_winsize=bold_winsize,
            transform=transform,
        )
        for split_name, split_records in records.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        ),
    }
    return datasets, loaders, audit
