import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from table3_utils import load_task_config, now_ts, read_json, task_run_dir


GROUP_TASKS = {
    "age": ["abide_age", "nki_age", "sald_age"],
    "sex": ["abcd_sex", "hcp_sex", "bhrc_sex"],
    "disease_education": ["ppmi_pd_diagnosis", "adni_mci", "adni_ad", "nki_education"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch grouped Table 3 jobs on a fixed GPU")
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--group", choices=sorted(GROUP_TASKS), required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    return parser.parse_args()


def parse_optional_int_list(raw: str) -> Optional[List[int]]:
    if raw == "all":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def expand_group_jobs(args: argparse.Namespace, group_root: Path) -> List[Dict[str, Any]]:
    selected_seeds = parse_optional_int_list(args.seeds)
    jobs: List[Dict[str, Any]] = []
    for task_id in GROUP_TASKS[args.group]:
        task_cfg = load_task_config(args.config, task_id)
        runtime_cfg = task_cfg.get("runtime", {})
        seeds = runtime_cfg.get("seeds", [1, 2, 3]) if selected_seeds is None else selected_seeds
        for fold in task_cfg["fold_ids"]:
            for seed in seeds:
                jobs.append(
                    {
                        "task_id": task_id,
                        "fold": fold,
                        "seed": seed,
                        "run_dir": task_run_dir(str(group_root), task_id, fold, seed),
                    }
                )
    return jobs


def build_run_name(raw_name: Optional[str]) -> str:
    if raw_name:
        return raw_name
    return time.strftime("%Y%m%d_%H%M%S")


def format_seed_summary(task_id: str, fold: int, seed: int, payload: Dict[str, Any]) -> str:
    best_epoch = payload.get("best_epoch")
    metrics = payload.get("test_metrics", {})
    if "mse_raw" in metrics:
        return (
            f"{task_id}/fold{fold}/seed{seed} best_epoch={best_epoch} "
            f"mse_raw={metrics.get('mse_raw'):.4f} rmse={metrics.get('rmse'):.4f} "
            f"pearson_r={metrics.get('pearson_r'):.4f} "
            f"mse_standardized={metrics.get('mse_standardized') if metrics.get('mse_standardized') is not None else 'NA'}"
        )
    return (
        f"{task_id}/fold{fold}/seed{seed} best_epoch={best_epoch} "
        f"acc={metrics.get('acc'):.4f} f1_weighted={metrics.get('f1_weighted'):.4f} "
        f"f1_macro={metrics.get('f1_macro'):.4f}"
    )


def run_job(script_dir: Path, args: argparse.Namespace, group_root: Path, job: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "launcher_stdout.log"
    command = [
        sys.executable,
        "finetune_table3.py",
        "--config",
        args.config,
        "--task_id",
        job["task_id"],
        "--fold",
        str(job["fold"]),
        "--seed",
        str(job["seed"]),
        "--device",
        args.device,
        "--output_root",
        str(group_root),
    ]
    if args.smoke_test:
        command.append("--smoke_test")
    if args.audit_only:
        command.append("--audit_only")
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.max_patience is not None:
        command.extend(["--max_patience", str(args.max_patience)])
    if args.num_workers is not None:
        command.extend(["--num_workers", str(args.num_workers)])

    print(
        f"[{now_ts()}] Starting {job['task_id']}/fold{job['fold']}/seed{job['seed']} "
        f"on {args.device} -> {run_dir}",
        flush=True,
    )
    with open(log_path, "a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Run failed for {job['task_id']}/fold{job['fold']}/seed{job['seed']} with code {return_code}"
        )

    if args.audit_only:
        summary = f"{job['task_id']}/fold{job['fold']}/seed{job['seed']} audit completed"
        print(f"[{now_ts()}] {summary}", flush=True)
        return {"summary": summary}

    metrics_path = run_dir / "best_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    payload = read_json(str(metrics_path))
    summary = format_seed_summary(job["task_id"], job["fold"], job["seed"], payload)
    print(f"[{now_ts()}] {summary}", flush=True)
    return {
        "task_id": job["task_id"],
        "fold": job["fold"],
        "seed": job["seed"],
        "summary": summary,
        "best_metrics": payload,
    }


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    run_name = build_run_name(args.run_name)
    group_root = Path(args.output_root) / run_name / args.group
    jobs = expand_group_jobs(args, group_root)

    print(
        json.dumps(
            {
                "group": args.group,
                "device": args.device,
                "run_name": run_name,
                "group_output_root": str(group_root),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "max_patience": args.max_patience,
                "num_workers": args.num_workers,
                "jobs": [f"{job['task_id']}/fold{job['fold']}/seed{job['seed']}" for job in jobs],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    summaries: List[str] = []
    for job in jobs:
        result = run_job(script_dir, args, group_root, job)
        summaries.append(result["summary"])

    print(f"[{now_ts()}] Group {args.group} finished. Output root: {group_root}", flush=True)
    for summary in summaries:
        print(f"[{now_ts()}] {summary}", flush=True)


if __name__ == "__main__":
    main()
