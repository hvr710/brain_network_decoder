from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from table3_utils import list_task_ids, load_task_config  # noqa: E402
from LP_MLP.probe_utils import (  # noqa: E402
    base_split_tag,
    format_mean_std,
    load_probe_config,
)


TOP_TASK_ORDER = [
    "abide_age",
    "nki_age",
    "sald_age",
    "abcd_sex",
    "hcp_sex",
]

BOTTOM_TASK_ORDER = [
    "bhrc_sex",
    "ppmi_pd_diagnosis",
    "adni_mci",
    "adni_ad",
    "nki_education",
]

TASK_LAYOUT = {
    "abide_age": {
        "dataset": "ABIDE",
        "subtitle": "Age Regression",
        "metrics": [("table3_mse", "MSE"), ("table3_r", "R")],
    },
    "nki_age": {
        "dataset": "NKI",
        "subtitle": "Age Regression",
        "metrics": [("table3_mse", "MSE"), ("table3_r", "R")],
    },
    "sald_age": {
        "dataset": "SALD",
        "subtitle": "Age Regression",
        "metrics": [("table3_mse", "MSE"), ("table3_r", "R")],
    },
    "abcd_sex": {
        "dataset": "ABCD",
        "subtitle": "Sex Classif.",
        "metrics": [("table3_acc", "ACC"), ("table3_f1", "F1")],
    },
    "hcp_sex": {
        "dataset": "HCP",
        "subtitle": "Sex Classif.",
        "metrics": [("table3_acc", "ACC"), ("table3_f1", "F1")],
    },
    "bhrc_sex": {
        "dataset": "BHRC",
        "subtitle": "Sex Classif.",
        "metrics": [("table3_acc", "ACC"), ("table3_f1", "F1")],
    },
    "ppmi_pd_diagnosis": {
        "dataset": "PPMI",
        "subtitle": "PD Diagnosis",
        "metrics": [("table3_acc", "ACC"), ("table3_f1", "F1")],
    },
    "adni_mci": {
        "dataset": "ADNI (MCI)",
        "subtitle": "Diagnosis",
        "metrics": [("table3_acc", "ACC"), ("table3_f1", "F1")],
    },
    "adni_ad": {
        "dataset": "ADNI (AD)",
        "subtitle": "Diagnosis",
        "metrics": [("table3_acc", "ACC"), ("table3_f1", "F1")],
    },
    "nki_education": {
        "dataset": "NKI",
        "subtitle": "Education Classif.",
        "metrics": [("table3_acc", "ACC"), ("table3_f1", "F1")],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LP/MLP probe runs")
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--probe_config", type=str, default="LP_MLP/probe_defaults.yaml")
    parser.add_argument("--output_root", type=str, default="LP_MLP/outputs")
    return parser.parse_args()


def metric_columns(task_cfg: Dict[str, Any]) -> List[str]:
    if task_cfg["task_type"] == "classification":
        return [
            "acc",
            "acc_percent",
            "f1_weighted",
            "f1_weighted_percent",
            "f1_macro",
            "f1_macro_percent",
        ]
    return ["mse_raw", "rmse", "mse_standardized", "pearson_r"]


def primary_metric_spec(task_cfg: Dict[str, Any]) -> Tuple[str, bool]:
    if task_cfg["task_type"] == "classification":
        return "f1_weighted_percent", True
    return "mse_standardized", False


def secondary_metric_spec(task_cfg: Dict[str, Any]) -> str:
    if task_cfg["task_type"] == "classification":
        return "acc_percent"
    return "pearson_r"


def parse_mean_std(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text or text == "NA":
        return None, None
    parts = text.split("+/-")
    if len(parts) != 2:
        return None, None
    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None, None


def pair_decimals(metric_key: str) -> int:
    return 3 if metric_key in {"table3_mse", "table3_r"} else 2


def metric_sort_reverse(metric_key: str) -> bool:
    return metric_key != "table3_mse"


def render_mean_std(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f}\u00b1{std:.{decimals}f}"


def cell_rankings(models: List[Dict[str, Any]], task_order: List[str]) -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    rankings: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    for task_id in task_order:
        for metric_key, _ in TASK_LAYOUT[task_id]["metrics"]:
            scored = []
            for model in models:
                payload = model.get("values", {}).get(task_id, {}).get(metric_key)
                if not payload:
                    continue
                mean_value = payload.get("mean")
                if isinstance(mean_value, (int, float)):
                    scored.append((model["name"], float(mean_value)))
            if not scored:
                rankings[(task_id, metric_key)] = {"best": None, "second": None}
                continue
            scored.sort(key=lambda item: item[1], reverse=metric_sort_reverse(metric_key))
            best_name = scored[0][0]
            second_name = scored[1][0] if len(scored) > 1 else None
            rankings[(task_id, metric_key)] = {"best": best_name, "second": second_name}
    return rankings


def render_table_rows(models: List[Dict[str, Any]], task_order: List[str]) -> str:
    rankings = cell_rankings(models, task_order)
    rows: List[str] = []
    for model in models:
        cells = [f"<th class='model-cell'>{html.escape(model['name'])}</th>"]
        for task_id in task_order:
            metrics = model.get("values", {}).get(task_id, {})
            for metric_key, _ in TASK_LAYOUT[task_id]["metrics"]:
                payload = metrics.get(metric_key)
                css_class = ""
                rank_info = rankings.get((task_id, metric_key), {})
                if rank_info.get("best") == model["name"]:
                    css_class = "best"
                elif rank_info.get("second") == model["name"]:
                    css_class = "second"
                if payload and isinstance(payload.get("mean"), (int, float)) and isinstance(payload.get("std"), (int, float)):
                    text = render_mean_std(float(payload["mean"]), float(payload["std"]), pair_decimals(metric_key))
                else:
                    text = "-"
                class_attr = f" class='{css_class}'" if css_class else ""
                cells.append(f"<td{class_attr}>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def render_table_section(title: str, task_order: List[str], models: List[Dict[str, Any]]) -> str:
    dataset_header = "".join(
        (
            f"<th colspan='2'><div class='dataset'>{html.escape(TASK_LAYOUT[task_id]['dataset'])}</div>"
            f"<div class='subtitle'>{html.escape(TASK_LAYOUT[task_id]['subtitle'])}</div></th>"
        )
        for task_id in task_order
    )
    metric_header = "".join(
        f"<th>{html.escape(metric_label)}</th>"
        for task_id in task_order
        for _, metric_label in TASK_LAYOUT[task_id]["metrics"]
    )
    body_rows = render_table_rows(models, task_order)
    return (
        f"<section class='table-block'><h2>{html.escape(title)}</h2>"
        "<table>"
        "<thead>"
        f"<tr><th rowspan='2' class='model-head'>Model</th>{dataset_header}</tr>"
        f"<tr>{metric_header}</tr>"
        "</thead>"
        f"<tbody>{body_rows}</tbody>"
        "</table></section>"
    )


def render_probe_html(models: List[Dict[str, Any]]) -> str:
    css = """
body { font-family: "Times New Roman", Georgia, serif; margin: 24px; color: #111; }
h1 { text-align: center; margin-bottom: 8px; }
h2 { text-align: center; margin: 20px 0 10px; font-size: 22px; }
p.caption { text-align: center; font-size: 18px; margin: 0 0 18px; }
ul.notes { margin: 0 auto 20px; max-width: 1100px; }
.table-block { margin: 0 auto 22px; max-width: 1220px; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; }
th, td { border: 1px solid #222; padding: 8px 6px; text-align: center; font-size: 16px; }
th.model-head, th.model-cell { width: 170px; }
th.model-cell { text-align: left; padding-left: 12px; }
.dataset { font-weight: 700; }
.subtitle { font-style: italic; font-weight: 400; margin-top: 2px; }
.best { color: #c62828; font-weight: 700; }
.second { text-decoration: underline; text-underline-offset: 2px; }
"""
    notes = [
        "Metrics are aggregated over 5 runs.",
        "The original val/test splits were merged and then re-split at subject level for each run.",
        "Age columns report standardized MSE and Pearson correlation.",
        "Classification columns report ACC and weighted F1 in percent.",
        "Red indicates the better value and underline indicates the other mode.",
    ]
    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    top_section = render_table_section("Part 1", TOP_TASK_ORDER, models)
    bottom_section = render_table_section("Part 2", BOTTOM_TASK_ORDER, models)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>LP/MLP Probe Table 3-style Comparison</title>"
        f"<style>{css}</style></head><body>"
        "<h1>LP/MLP Probe Table 3-style Comparison</h1>"
        "<p class='caption'>Linear probe and MLP probe results aggregated over 5 subject-level re-splits.</p>"
        f"<ul class='notes'>{notes_html}</ul>"
        f"{top_section}{bottom_section}"
        "</body></html>"
    )


def build_probe_html_models(mode_lookup: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    model_specs = [
        ("linear_probe", "Linear Probe"),
        ("mlp_probe", "MLP Probe"),
    ]
    models: List[Dict[str, Any]] = []
    for mode_key, display_name in model_specs:
        values: Dict[str, Dict[str, Dict[str, float]]] = {}
        for task_id in TOP_TASK_ORDER + BOTTOM_TASK_ORDER:
            row = mode_lookup.get((task_id, mode_key), {})
            task_values: Dict[str, Dict[str, float]] = {}
            mapping = (
                [("table3_acc", "test_metrics_acc_percent_mean_std"), ("table3_f1", "test_metrics_f1_weighted_percent_mean_std")]
                if task_id not in {"abide_age", "nki_age", "sald_age"}
                else [("table3_mse", "test_metrics_mse_standardized_mean_std"), ("table3_r", "test_metrics_pearson_r_mean_std")]
            )
            for metric_key, source_key in mapping:
                mean_value, std_value = parse_mean_std(row.get(source_key, "NA"))
                if mean_value is not None and std_value is not None:
                    task_values[metric_key] = {"mean": mean_value, "std": std_value}
            if task_values:
                values[task_id] = task_values
        models.append({"name": display_name, "values": values})
    return models


def choose_better_mode(
    task_cfg: Dict[str, Any],
    lp_value: Optional[float],
    mlp_value: Optional[float],
) -> str:
    if lp_value is None and mlp_value is None:
        return "NA"
    if lp_value is None:
        return "mlp_probe"
    if mlp_value is None:
        return "linear_probe"
    _, higher_is_better = primary_metric_spec(task_cfg)
    if higher_is_better:
        return "mlp_probe" if mlp_value > lp_value else "linear_probe"
    return "mlp_probe" if mlp_value < lp_value else "linear_probe"


def write_shareable_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# LP/MLP Probe Summary",
        "",
        "This file is the short version for sharing with collaborators.",
        "",
        "- Metrics are aggregated over 5 runs.",
        "- `val` and `test` were re-split from the merged original val/test pool at subject level.",
        "- `linear_probe` uses sklearn logistic/ridge; `mlp_probe` uses the configured MLP head.",
        "",
        "| Task | Type | LP primary | LP secondary | MLP primary | MLP secondary | Better mode |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {display_name} | {task_type} | {lp_primary} | {lp_secondary} | {mlp_primary} | {mlp_secondary} | {better_mode} |".format(
                display_name=row["display_name"],
                task_type=row["task_type"],
                lp_primary=row["lp_test_primary_mean_std"],
                lp_secondary=row["lp_test_secondary_mean_std"],
                mlp_primary=row["mlp_test_primary_mean_std"],
                mlp_secondary=row["mlp_test_secondary_mean_std"],
                better_mode=row["better_mode"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_to_csv(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_latest{path.suffix}")
        frame.to_csv(fallback, index=False)
        return fallback


def safe_write_text(path: Path, content: str) -> Path:
    try:
        path.write_text(content, encoding="utf-8")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_latest{path.suffix}")
        fallback.write_text(content, encoding="utf-8")
        return fallback


def main() -> None:
    args = parse_args()
    probe_cfg = load_probe_config(args.probe_config)
    base_split_index = int(probe_cfg.get("base_split_index", 1))
    expected_runs = int(probe_cfg.get("num_runs", 5))
    split_tag = base_split_tag(base_split_index)
    output_root = Path(args.output_root)

    task_rows: List[Dict[str, Any]] = []
    table3_rows: List[Dict[str, Any]] = []
    share_rows: List[Dict[str, Any]] = []
    task_mode_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for task_id in list_task_ids(args.config):
        task_cfg = load_task_config(args.config, task_id)
        for mode in probe_cfg.get("modes", ["linear_probe", "mlp_probe"]):
            mode_dir = output_root / mode / task_id / split_tag
            summary_path = mode_dir / "probe_summary.json"
            manifest_path = mode_dir / "run_manifest.csv"
            if not summary_path.exists():
                continue
            with open(summary_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            row = {
                "task_id": task_id,
                "display_name": task_cfg.get("display_name", task_id),
                "task_type": task_cfg["task_type"],
                "mode": mode,
                "base_split_tag": split_tag,
                "completed_runs": payload.get("completed_runs", 0),
                "expected_runs": expected_runs,
                "mode_dir": str(mode_dir),
                "run_manifest_path": str(manifest_path),
            }
            for split_name in ("val_metrics", "test_metrics"):
                for metric_name, metric_payload in payload.get(split_name, {}).items():
                    row[f"{split_name}_{metric_name}_mean_std"] = metric_payload.get("mean_std", "NA")
            task_rows.append(row)
            task_mode_lookup[(task_id, mode)] = row

            table_row = {
                "task_id": task_id,
                "display_name": task_cfg.get("display_name", task_id),
                "task_type": task_cfg["task_type"],
                "mode": mode,
                "completed_runs": payload.get("completed_runs", 0),
                "expected_runs": expected_runs,
            }
            for metric_name in metric_columns(task_cfg):
                test_payload = payload.get("test_metrics", {}).get(metric_name, {})
                table_row[f"test_{metric_name}_mean_std"] = test_payload.get("mean_std", "NA")
            table3_rows.append(table_row)

    import pandas as pd

    output_root.mkdir(parents=True, exist_ok=True)
    task_summary_path = safe_to_csv(pd.DataFrame(task_rows), output_root / "probe_task_summary.csv")
    table3_summary_path = safe_to_csv(pd.DataFrame(table3_rows), output_root / "probe_table3_style_summary.csv")
    for task_id in list_task_ids(args.config):
        task_cfg = load_task_config(args.config, task_id)
        display_name = task_cfg.get("display_name", task_id)
        primary_metric, _ = primary_metric_spec(task_cfg)
        secondary_metric = secondary_metric_spec(task_cfg)
        lp_row = task_mode_lookup.get((task_id, "linear_probe"), {})
        mlp_row = task_mode_lookup.get((task_id, "mlp_probe"), {})
        lp_primary_text = lp_row.get(f"test_metrics_{primary_metric}_mean_std", "NA")
        lp_secondary_text = lp_row.get(f"test_metrics_{secondary_metric}_mean_std", "NA")
        mlp_primary_text = mlp_row.get(f"test_metrics_{primary_metric}_mean_std", "NA")
        mlp_secondary_text = mlp_row.get(f"test_metrics_{secondary_metric}_mean_std", "NA")
        lp_primary_value, _ = parse_mean_std(lp_primary_text)
        mlp_primary_value, _ = parse_mean_std(mlp_primary_text)
        share_rows.append(
            {
                "task_id": task_id,
                "display_name": display_name,
                "task_type": task_cfg["task_type"],
                "primary_metric": primary_metric,
                "secondary_metric": secondary_metric,
                "lp_test_primary_mean_std": lp_primary_text,
                "lp_test_secondary_mean_std": lp_secondary_text,
                "mlp_test_primary_mean_std": mlp_primary_text,
                "mlp_test_secondary_mean_std": mlp_secondary_text,
                "better_mode": choose_better_mode(task_cfg, lp_primary_value, mlp_primary_value),
            }
        )
    comparison_csv_path = safe_to_csv(pd.DataFrame(share_rows), output_root / "probe_lp_mlp_comparison.csv")
    comparison_md_path = safe_write_text(output_root / "probe_lp_mlp_comparison.md", "\n".join([
        "# LP/MLP Probe Summary",
        "",
        "This file is the short version for sharing with collaborators.",
        "",
        "- Metrics are aggregated over 5 runs.",
        "- `val` and `test` were re-split from the merged original val/test pool at subject level.",
        "- `linear_probe` uses sklearn logistic/ridge; `mlp_probe` uses the configured MLP head.",
        "",
        "| Task | Type | LP primary | LP secondary | MLP primary | MLP secondary | Better mode |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            "| {display_name} | {task_type} | {lp_primary} | {lp_secondary} | {mlp_primary} | {mlp_secondary} | {better_mode} |".format(
                display_name=row["display_name"],
                task_type=row["task_type"],
                lp_primary=row["lp_test_primary_mean_std"],
                lp_secondary=row["lp_test_secondary_mean_std"],
                mlp_primary=row["mlp_test_primary_mean_std"],
                mlp_secondary=row["mlp_test_secondary_mean_std"],
                better_mode=row["better_mode"],
            )
            for row in share_rows
        ],
        "",
    ]) ,)
    probe_html = render_probe_html(build_probe_html_models(task_mode_lookup))
    html_path = safe_write_text(output_root / "probe_table3_comparison.html", probe_html)
    print(f"Saved {task_summary_path}")
    print(f"Saved {table3_summary_path}")
    print(f"Saved {comparison_csv_path}")
    print(f"Saved {comparison_md_path}")
    print(f"Saved {html_path}")


if __name__ == "__main__":
    main()
