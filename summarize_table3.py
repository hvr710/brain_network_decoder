import argparse
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from table3_utils import list_task_ids, load_task_config, load_yaml, read_json


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
    parser = argparse.ArgumentParser(description="Summarize Table 3 runs")
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--paper_baselines", type=str, default="our_plan/table3_paper_baselines.yaml")
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


def table3_metric_columns(task_cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    if task_cfg["task_type"] == "classification":
        return [
            ("table3_acc", "acc_percent"),
            ("table3_f1", "f1_weighted_percent"),
        ]
    return [
        ("table3_mse", "mse_standardized"),
        ("table3_r", "pearson_r"),
    ]


def format_mean_std(values: List[float], decimals: int = 4) -> str:
    if not values:
        return "NA"
    series = pd.Series(values, dtype=float)
    return f"{series.mean():.{decimals}f} +/- {series.std(ddof=0):.{decimals}f}"


def mean_std_pair(values: List[float]) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    series = pd.Series(values, dtype=float)
    return float(series.mean()), float(series.std(ddof=0))


def render_mean_std(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f}\u00b1{std:.{decimals}f}"


def resolve_task_root(output_root: Path, task_id: str) -> Path:
    direct_root = output_root / task_id
    if direct_root.exists():
        return direct_root

    if output_root.name == task_id or output_root.name.startswith(f"{task_id}_"):
        nested_root = output_root / task_id
        return nested_root if nested_root.exists() else output_root

    candidates = [path for path in output_root.glob(f"{task_id}_*") if path.is_dir()]
    if candidates:
        latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
        nested_root = latest / task_id
        return nested_root if nested_root.exists() else latest

    recursive_candidates = [path for path in output_root.rglob(task_id) if path.is_dir() and path.name == task_id]
    if recursive_candidates:
        return max(recursive_candidates, key=lambda path: (path.stat().st_mtime, path.name))

    return direct_root


def build_alignment_row(
    task_cfg: Dict[str, Any],
    metrics_found: Dict[str, List[float]],
    table3_found: Dict[str, List[float]],
) -> Dict[str, Any]:
    row = {
        "task_id": task_cfg["task_id"],
        "display_name": task_cfg.get("display_name", task_cfg["task_id"]),
        "task_type": task_cfg["task_type"],
    }
    for metric_name, values in metrics_found.items():
        row[f"{metric_name}_mean_std"] = format_mean_std(values)
    for metric_name, values in table3_found.items():
        row[f"{metric_name}_mean_std"] = format_mean_std(values)
    return row


def pair_decimals(metric_key: str) -> int:
    return 3 if metric_key in {"table3_mse", "table3_r"} else 2


def metric_sort_reverse(metric_key: str) -> bool:
    return metric_key != "table3_mse"


def build_lcm_html_model(task_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    values: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task_id, stats in task_stats.items():
        table3_values = stats.get("table3_values", {})
        if table3_values:
            values[task_id] = table3_values
    return {"name": "LCM", "values": values}


def load_paper_models(path: str) -> List[Dict[str, Any]]:
    payload = load_yaml(path)
    return payload.get("models", [])


def cell_rankings(models: List[Dict[str, Any]], task_order: List[str]) -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    rankings: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    for task_id in task_order:
        for metric_key, _ in TASK_LAYOUT[task_id]["metrics"]:
            scored = []
            for model in models:
                metrics = model.get("values", {}).get(task_id, {})
                payload = metrics.get(metric_key)
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


def render_table_rows(
    models: List[Dict[str, Any]],
    task_order: List[str],
    *,
    highlight: bool,
) -> str:
    rankings = cell_rankings(models, task_order) if highlight else {}
    rows = []
    for model in models:
        cells = [f"<th class='model-cell'>{html.escape(model['name'])}</th>"]
        for task_id in task_order:
            metrics = model.get("values", {}).get(task_id, {})
            for metric_key, _ in TASK_LAYOUT[task_id]["metrics"]:
                payload = metrics.get(metric_key)
                css_class = ""
                if highlight:
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


def render_table_section(title: str, task_order: List[str], models: List[Dict[str, Any]], *, highlight: bool) -> str:
    dataset_header = "".join(
        [
            (
                f"<th colspan='2'><div class='dataset'>{html.escape(TASK_LAYOUT[task_id]['dataset'])}</div>"
                f"<div class='subtitle'>{html.escape(TASK_LAYOUT[task_id]['subtitle'])}</div></th>"
            )
            for task_id in task_order
        ]
    )
    metric_header = "".join(
        [f"<th>{html.escape(metric_label)}</th>" for task_id in task_order for _, metric_label in TASK_LAYOUT[task_id]["metrics"]]
    )
    body_rows = render_table_rows(models, task_order, highlight=highlight)
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


def render_html_page(
    *,
    title: str,
    subtitle: str,
    models: List[Dict[str, Any]],
    highlight: bool,
) -> str:
    notes = [
        "Age columns report standardized MSE and Pearson correlation to match the Table 3 scale.",
        "Classification columns report ACC and weighted F1 in percent.",
    ]
    if highlight:
        notes.append("Red indicates the best value and underline indicates the second best value within the displayed rows.")
    css = """
body { font-family: "Times New Roman", Georgia, serif; margin: 24px; color: #111; }
h1 { text-align: center; margin-bottom: 8px; }
h2 { text-align: center; margin: 20px 0 10px; font-size: 22px; }
p.caption { text-align: center; font-size: 18px; margin: 0 0 18px; }
ul.notes { margin: 0 auto 20px; max-width: 1100px; }
.table-block { margin: 0 auto 22px; max-width: 1220px; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; }
th, td { border: 1px solid #222; padding: 8px 6px; text-align: center; font-size: 16px; }
th.model-head, th.model-cell { width: 150px; }
th.model-cell { text-align: left; padding-left: 12px; }
.dataset { font-weight: 700; }
.subtitle { font-style: italic; font-weight: 400; margin-top: 2px; }
.best { color: #c62828; font-weight: 700; }
.second { text-decoration: underline; text-underline-offset: 2px; }
"""
    top_section = render_table_section("Part 1", TOP_TASK_ORDER, models, highlight=highlight)
    bottom_section = render_table_section("Part 2", BOTTOM_TASK_ORDER, models, highlight=highlight)
    notes_html = "".join([f"<li>{html.escape(note)}</li>" for note in notes])
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{css}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='caption'>{html.escape(subtitle)}</p>"
        f"<ul class='notes'>{notes_html}</ul>"
        f"{top_section}{bottom_section}"
        "</body></html>"
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    omni_rows: List[Dict[str, Any]] = []
    task_stats: Dict[str, Dict[str, Any]] = {}

    for task_id in list_task_ids(args.config):
        task_cfg = load_task_config(args.config, task_id)
        runtime_cfg = task_cfg.get("runtime", {})
        seeds = runtime_cfg.get("seeds", [1, 2, 3])
        folds = task_cfg["fold_ids"]
        expected_runs = len(seeds) * len(folds)
        task_root = resolve_task_root(output_root, task_id)
        metrics_found = {key: [] for key in metric_columns(task_cfg)}
        table3_found = {key: [] for key, _ in table3_metric_columns(task_cfg)}
        completed_runs = 0

        for fold in folds:
            for seed in seeds:
                run_dir = task_root / f"fold{fold}" / f"seed{seed}"
                metrics_path = run_dir / "best_metrics.json"
                state_path = run_dir / "run_state.json"
                row = {
                    "task_id": task_id,
                    "display_name": task_cfg.get("display_name", task_id),
                    "task_root": str(task_root),
                    "fold": fold,
                    "seed": seed,
                    "run_dir": str(run_dir),
                }
                if state_path.exists():
                    row["run_state"] = read_json(str(state_path)).get("state", "unknown")
                else:
                    row["run_state"] = "missing"
                if metrics_path.exists():
                    payload = read_json(str(metrics_path))
                    completed_runs += 1
                    row["best_epoch"] = payload.get("best_epoch")
                    row["final_epoch"] = payload.get("final_epoch")
                    row["epochs_after_best"] = payload.get("epochs_after_best")
                    row["train_loss_at_best"] = payload.get("train_loss_at_best")
                    row["final_train_loss"] = payload.get("final_train_loss")
                    row["overfit_flag"] = payload.get("overfit_flag")
                    row["selection_metric"] = payload.get("selection_metric")
                    row["peak_gpu_mem_mb"] = payload.get("peak_gpu_mem_mb")
                    test_metrics = payload.get("test_metrics", {})
                    for metric_name in metric_columns(task_cfg):
                        value = test_metrics.get(metric_name)
                        row[metric_name] = value
                        if isinstance(value, (int, float)):
                            metrics_found[metric_name].append(float(value))
                    for table3_name, source_name in table3_metric_columns(task_cfg):
                        value = test_metrics.get(source_name)
                        row[table3_name] = value
                        if isinstance(value, (int, float)):
                            table3_found[table3_name].append(float(value))
                all_rows.append(row)

        summary_row = {
            "task_id": task_id,
            "display_name": task_cfg.get("display_name", task_id),
            "task_type": task_cfg["task_type"],
            "task_root": str(task_root),
            "completed_runs": completed_runs,
            "expected_runs": expected_runs,
        }
        for metric_name, values in metrics_found.items():
            summary_row[f"{metric_name}_mean_std"] = format_mean_std(values)
        for metric_name, values in table3_found.items():
            summary_row[f"{metric_name}_mean_std"] = format_mean_std(values)
        summary_rows.append(summary_row)
        omni_rows.append(build_alignment_row(task_cfg, metrics_found, table3_found))

        table3_values: Dict[str, Dict[str, float]] = {}
        for table3_name, values in table3_found.items():
            pair = mean_std_pair(values)
            if pair is not None:
                table3_values[table3_name] = {"mean": pair[0], "std": pair[1]}
        task_stats[task_id] = {
            "task_id": task_id,
            "display_name": task_cfg.get("display_name", task_id),
            "task_type": task_cfg["task_type"],
            "table3_values": table3_values,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_root / "task_summary.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_root / "table3_style_summary.csv", index=False)
    pd.DataFrame(omni_rows).to_csv(output_root / "omni_alignment_summary.csv", index=False)

    lcm_model = build_lcm_html_model(task_stats)
    lcm_only_html = render_html_page(
        title="LCM Baseline on Omni-fMRI Table 3 Benchmark",
        subtitle="Downstream finetuning summary for the 10 Table 3 tasks.",
        models=[lcm_model],
        highlight=False,
    )
    (output_root / "table3_lcm_only.html").write_text(lcm_only_html, encoding="utf-8")

    paper_models = load_paper_models(args.paper_baselines)
    comparison_html = render_html_page(
        title="Omni-fMRI Table 3-style Comparison with LCM Baseline",
        subtitle="Paper baselines are static references; the LCM row is produced from the current output_root.",
        models=paper_models + [lcm_model],
        highlight=True,
    )
    (output_root / "table3_with_paper_baselines.html").write_text(comparison_html, encoding="utf-8")

    print(f"Saved {output_root / 'task_summary.csv'}")
    print(f"Saved {output_root / 'table3_style_summary.csv'}")
    print(f"Saved {output_root / 'omni_alignment_summary.csv'}")
    print(f"Saved {output_root / 'table3_lcm_only.html'}")
    print(f"Saved {output_root / 'table3_with_paper_baselines.html'}")


if __name__ == "__main__":
    main()
