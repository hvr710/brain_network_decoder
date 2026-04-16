import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from table3_utils import list_task_ids, load_task_config, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Table 3 runs")
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--output_root", type=str, default="outputs/table3")
    return parser.parse_args()


def metric_columns(task_cfg: Dict[str, Any]) -> List[str]:
    return ["acc", "f1"] if task_cfg["task_type"] == "classification" else ["mse", "pearson_r"]


def format_mean_std(values: List[float]) -> str:
    if not values:
        return "NA"
    series = pd.Series(values, dtype=float)
    return f"{series.mean():.4f}±{series.std(ddof=0):.4f}"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for task_id in list_task_ids(args.config):
        task_cfg = load_task_config(args.config, task_id)
        runtime_cfg = task_cfg.get("runtime", {})
        seeds = runtime_cfg.get("seeds", [1, 2, 3])
        folds = task_cfg["fold_ids"]
        expected_runs = len(seeds) * len(folds)
        metrics_found = {key: [] for key in metric_columns(task_cfg)}
        completed_runs = 0

        for fold in folds:
            for seed in seeds:
                run_dir = output_root / task_id / f"fold{fold}" / f"seed{seed}"
                metrics_path = run_dir / "best_metrics.json"
                state_path = run_dir / "run_state.json"
                row = {
                    "task_id": task_id,
                    "display_name": task_cfg.get("display_name", task_id),
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
                    row["peak_gpu_mem_mb"] = payload.get("peak_gpu_mem_mb")
                    for metric_name in metric_columns(task_cfg):
                        value = payload.get("test_metrics", {}).get(metric_name)
                        row[metric_name] = value
                        if isinstance(value, (int, float)):
                            metrics_found[metric_name].append(float(value))
                all_rows.append(row)

        summary_row = {
            "task_id": task_id,
            "display_name": task_cfg.get("display_name", task_id),
            "task_type": task_cfg["task_type"],
            "completed_runs": completed_runs,
            "expected_runs": expected_runs,
        }
        for metric_name, values in metrics_found.items():
            summary_row[f"{metric_name}_mean_std"] = format_mean_std(values)
        summary_rows.append(summary_row)

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_root / "task_summary.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_root / "table3_style_summary.csv", index=False)
    print(f"Saved {output_root / 'task_summary.csv'}")
    print(f"Saved {output_root / 'table3_style_summary.csv'}")


if __name__ == "__main__":
    main()
