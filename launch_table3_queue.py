import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from table3_utils import (
    acquire_gpu_lock,
    choose_gpu,
    estimate_required_mem_mb,
    gpu_lock_path,
    list_task_ids,
    load_task_config,
    read_json,
    release_gpu_lock,
    task_run_dir,
    update_run_state,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue Table 3 jobs with auto GPU selection")
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--output_root", type=str, default="outputs/table3")
    parser.add_argument("--task_ids", type=str, default="all")
    parser.add_argument("--folds", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="all")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    return parser.parse_args()


def parse_optional_int_list(raw: str) -> Optional[List[int]]:
    if raw == "all":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_optional_str_list(raw: str, default: List[str]) -> List[str]:
    if raw == "all":
        return default
    return [part.strip() for part in raw.split(",") if part.strip()]


def expand_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    task_ids = parse_optional_str_list(args.task_ids, list_task_ids(args.config))
    selected_folds = parse_optional_int_list(args.folds)
    selected_seeds = parse_optional_int_list(args.seeds)

    jobs: List[Dict[str, Any]] = []
    for task_id in task_ids:
        task_cfg = load_task_config(args.config, task_id)
        runtime_cfg = task_cfg.get("runtime", {})
        folds = task_cfg["fold_ids"] if selected_folds is None else [fold for fold in selected_folds if fold in task_cfg["fold_ids"]]
        seeds = runtime_cfg.get("seeds", [1, 2, 3]) if selected_seeds is None else selected_seeds
        for fold in folds:
            for seed in seeds:
                jobs.append(
                    {
                        "task_id": task_id,
                        "fold": fold,
                        "seed": seed,
                        "required_mem_mb": estimate_required_mem_mb(args.output_root, task_id),
                    }
                )
    return jobs


def active_summary(active_jobs: List[Dict[str, Any]]) -> str:
    if not active_jobs:
        return "none"
    return ", ".join(
        f"{job['task_id']}/fold{job['fold']}/seed{job['seed']}@cuda:{job['gpu_index']}"
        for job in active_jobs
    )


def main() -> None:
    args = parse_args()
    pending_jobs = deque(expand_jobs(args))
    active_jobs: List[Dict[str, Any]] = []

    print(f"Queued {len(pending_jobs)} jobs.", flush=True)
    while pending_jobs or active_jobs:
        next_active: List[Dict[str, Any]] = []
        for active in active_jobs:
            if active["process"].poll() is None:
                next_active.append(active)
                continue
            active["log_handle"].close()
            release_gpu_lock(active.get("lock_path"))
            print(
                f"Finished {active['task_id']}/fold{active['fold']}/seed{active['seed']} "
                f"(code={active['process'].returncode}).",
                flush=True,
            )
        active_jobs = next_active

        if pending_jobs:
            job = pending_jobs[0]
            run_dir = task_run_dir(args.output_root, job["task_id"], job["fold"], job["seed"])
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            update_run_state(run_dir, "queued", task_id=job["task_id"], fold=job["fold"], seed=job["seed"])

            if args.device:
                if active_jobs:
                    time.sleep(5)
                    continue
                gpu_index = int(args.device.split(":")[-1])
                chosen_gpu = {"index": gpu_index, "memory_free_mb": -1, "utilization_gpu": -1}
                lock_path = None
            else:
                chosen_gpu = choose_gpu(args.output_root, job["required_mem_mb"])
                if chosen_gpu is None:
                    print(
                        f"No GPU available for {job['task_id']}/fold{job['fold']}/seed{job['seed']} "
                        f"(need {job['required_mem_mb']} MB). Waiting {args.poll_seconds}s. "
                        f"Active={active_summary(active_jobs)}",
                        flush=True,
                    )
                    time.sleep(args.poll_seconds)
                    continue
                ok, lock_path = acquire_gpu_lock(
                    args.output_root,
                    chosen_gpu["index"],
                    pid=os.getpid(),
                    task_id=job["task_id"],
                    fold=job["fold"],
                    seed=job["seed"],
                )
                if not ok:
                    time.sleep(2)
                    continue

            log_path = Path(run_dir) / "launcher_stdout.log"
            log_handle = open(log_path, "a", encoding="utf-8")
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
                args.device or f"cuda:{chosen_gpu['index']}",
                "--output_root",
                args.output_root,
            ]
            if args.smoke_test:
                command.append("--smoke_test")
            if args.audit_only:
                command.append("--audit_only")
            if not args.device:
                command.extend(["--gpu_lock_file", gpu_lock_path(args.output_root, chosen_gpu["index"])])

            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if not args.device:
                write_json(
                    gpu_lock_path(args.output_root, chosen_gpu["index"]),
                    {
                        "pid": process.pid,
                        "task_id": job["task_id"],
                        "fold": job["fold"],
                        "seed": job["seed"],
                        "gpu_index": chosen_gpu["index"],
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )

            job = pending_jobs.popleft()
            job["gpu_index"] = chosen_gpu["index"]
            job["process"] = process
            job["log_handle"] = log_handle
            job["lock_path"] = None if args.device else gpu_lock_path(args.output_root, chosen_gpu["index"])
            active_jobs.append(job)
            print(
                f"Started {job['task_id']}/fold{job['fold']}/seed{job['seed']} on cuda:{job['gpu_index']} "
                f"(need {job['required_mem_mb']} MB). Active={active_summary(active_jobs)}",
                flush=True,
            )
            continue

        time.sleep(5)

    print("All queued jobs finished.", flush=True)


if __name__ == "__main__":
    main()
