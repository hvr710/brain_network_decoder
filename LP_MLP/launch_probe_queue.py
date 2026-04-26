from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from table3_utils import (  # noqa: E402
    acquire_gpu_lock,
    clear_stale_lock,
    gpu_lock_path,
    list_task_ids,
    release_gpu_lock,
    update_run_state,
    write_json,
    query_gpus,
)
from LP_MLP.probe_utils import (  # noqa: E402
    base_split_tag,
    feature_cache_dir,
    feature_cache_ready,
    load_probe_config,
    probe_mode_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue LP/MLP extraction and probe jobs")
    parser.add_argument("--stage", type=str, choices=["extract", "probe", "all"], required=True)
    parser.add_argument("--config", type=str, default="our_plan/table3_tasks.yaml")
    parser.add_argument("--probe_config", type=str, default="LP_MLP/probe_defaults.yaml")
    parser.add_argument("--task_ids", type=str, default="all")
    parser.add_argument("--modes", type=str, default="all")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--preferred_gpus", type=str, default="2,7,6")
    parser.add_argument("--wait_for_cache", action="store_true")
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--output_root", type=str, default="LP_MLP/outputs")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def parse_optional_list(raw: str, default: List[str]) -> List[str]:
    if raw == "all":
        return default
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def choose_preferred_gpu(output_root: str, preferred_gpus: List[int], required_mem_mb: int) -> Optional[Dict[str, int]]:
    gpu_map = {gpu["index"]: gpu for gpu in query_gpus()}
    for gpu_index in preferred_gpus:
        gpu = gpu_map.get(int(gpu_index))
        if gpu is None:
            continue
        lock_path = gpu_lock_path(output_root, gpu["index"])
        clear_stale_lock(lock_path)
        if os.path.exists(lock_path):
            continue
        if int(gpu["memory_free_mb"]) < int(required_mem_mb):
            continue
        return gpu
    return None


def job_requires_gpu(job: Dict[str, Any], requested_device: str) -> bool:
    if requested_device == "cpu":
        return False
    if job["stage"] == "extract":
        return requested_device.startswith("cuda") or requested_device == "cuda"
    if job["stage"] == "probe" and job["mode"] == "mlp_probe":
        return requested_device.startswith("cuda") or requested_device == "cuda"
    return False


def job_device_string(job: Dict[str, Any], selected_gpu_index: Optional[int], requested_device: str) -> str:
    if not job_requires_gpu(job, requested_device):
        return "cpu"
    if requested_device.startswith("cuda:"):
        return requested_device
    if requested_device == "cuda":
        if selected_gpu_index is None:
            raise ValueError("selected_gpu_index is required when requested_device='cuda'")
        return f"cuda:{selected_gpu_index}"
    return requested_device


def required_mem_mb(job: Dict[str, Any]) -> int:
    if job["stage"] == "extract":
        return 12000
    if job["mode"] == "mlp_probe":
        return 4000
    return 0


def job_output_dir(output_root: str, job: Dict[str, Any], base_split_index: int) -> Path:
    if job["stage"] == "extract":
        return feature_cache_dir(output_root, job["task_id"], base_split_index)
    return probe_mode_dir(output_root, job["mode"], job["task_id"], base_split_index)


def append_launcher_log(output_root: str, job: Dict[str, Any], base_split_index: int, message: str) -> None:
    log_path = job_output_dir(output_root, job, base_split_index) / "launcher_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def expand_jobs(args: argparse.Namespace, probe_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_ids = parse_optional_list(args.task_ids, list_task_ids(args.config))
    default_modes = list(probe_cfg.get("modes", ["linear_probe", "mlp_probe"]))
    modes = parse_optional_list(args.modes, default_modes)
    jobs: List[Dict[str, Any]] = []
    if args.stage in {"extract", "all"}:
        jobs.extend({"stage": "extract", "task_id": task_id, "mode": None} for task_id in task_ids)
    if args.stage in {"probe", "all"}:
        for task_id in task_ids:
            for mode in modes:
                jobs.append({"stage": "probe", "task_id": task_id, "mode": mode})
    return jobs


def build_command(args: argparse.Namespace, job: Dict[str, Any], base_split_index: int, device: str, gpu_lock_file: Optional[str]) -> List[str]:
    if job["stage"] == "extract":
        command = [
            sys.executable,
            "LP_MLP/extract_lcm_features.py",
            "--config",
            args.config,
            "--probe_config",
            args.probe_config,
            "--task_id",
            job["task_id"],
            "--base_split_index",
            str(base_split_index),
            "--device",
            device,
            "--output_root",
            args.output_root,
        ]
        if gpu_lock_file:
            command.extend(["--gpu_lock_file", gpu_lock_file])
    else:
        command = [
            sys.executable,
            "LP_MLP/run_probe_task.py",
            "--config",
            args.config,
            "--probe_config",
            args.probe_config,
            "--task_id",
            job["task_id"],
            "--mode",
            job["mode"],
            "--base_split_index",
            str(base_split_index),
            "--device",
            device,
            "--output_root",
            args.output_root,
        ]
    if args.smoke_test:
        command.append("--smoke_test")
    return command


def feature_ready_for_job(args: argparse.Namespace, job: Dict[str, Any], base_split_index: int) -> bool:
    if job["stage"] != "probe":
        return True
    return feature_cache_ready(args.output_root, job["task_id"], base_split_index)


def main() -> None:
    args = parse_args()
    probe_cfg = load_probe_config(args.probe_config)
    base_split_index = int(probe_cfg.get("base_split_index", 1))
    preferred_gpus = parse_int_list(args.preferred_gpus) if args.preferred_gpus else []
    pending_jobs: Deque[Dict[str, Any]] = deque(expand_jobs(args, probe_cfg))
    active_job: Optional[Dict[str, Any]] = None

    print(
        f"Queued {len(pending_jobs)} jobs for stage={args.stage} base_split={base_split_tag(base_split_index)}.",
        flush=True,
    )

    while pending_jobs or active_job is not None:
        if active_job is not None:
            process = active_job["process"]
            if process.poll() is None:
                time.sleep(5)
                continue
            release_gpu_lock(active_job.get("lock_path"))
            finish_line = (
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished {active_job['stage']} "
                f"task={active_job['task_id']} mode={active_job.get('mode')} code={process.returncode}"
            )
            append_launcher_log(args.output_root, active_job, base_split_index, finish_line)
            print(finish_line, flush=True)
            active_job = None
            continue

        blocked_count = 0
        started_job = False
        for _ in range(len(pending_jobs)):
            job = pending_jobs.popleft()
            if not feature_ready_for_job(args, job, base_split_index):
                if args.wait_for_cache:
                    pending_jobs.append(job)
                    blocked_count += 1
                    continue
                print(
                    f"Skipping probe task={job['task_id']} mode={job['mode']} because feature cache is missing.",
                    flush=True,
                )
                continue

            needs_gpu = job_requires_gpu(job, args.device)
            selected_gpu: Optional[Dict[str, int]] = None
            lock_path: Optional[str] = None
            if needs_gpu:
                if args.device.startswith("cuda:"):
                    selected_index = int(args.device.split(":")[-1])
                    selected_gpu = {"index": selected_index}
                    ok, lock_path = acquire_gpu_lock(
                        args.output_root,
                        selected_index,
                        pid=os.getpid(),
                        task_id=job["task_id"],
                        fold=base_split_index,
                        seed=0,
                    )
                    if not ok:
                        pending_jobs.append(job)
                        blocked_count += 1
                        continue
                else:
                    selected_gpu = choose_preferred_gpu(args.output_root, preferred_gpus, required_mem_mb(job))
                    if selected_gpu is None:
                        pending_jobs.append(job)
                        blocked_count += 1
                        continue
                    ok, lock_path = acquire_gpu_lock(
                        args.output_root,
                        selected_gpu["index"],
                        pid=os.getpid(),
                        task_id=job["task_id"],
                        fold=base_split_index,
                        seed=0,
                    )
                    if not ok:
                        pending_jobs.append(job)
                        blocked_count += 1
                        continue

            run_device = job_device_string(job, None if selected_gpu is None else selected_gpu["index"], args.device)
            command = build_command(args, job, base_split_index, run_device, lock_path if job["stage"] == "extract" else None)
            out_dir = job_output_dir(args.output_root, job, base_split_index)
            out_dir.mkdir(parents=True, exist_ok=True)
            if job["stage"] == "extract":
                update_run_state(
                    str(out_dir),
                    "queued",
                    task_id=job["task_id"],
                    stage="extract",
                    base_split_index=base_split_index,
                )
            start_line = (
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started {job['stage']} "
                f"task={job['task_id']} mode={job.get('mode')} device={run_device}"
            )
            append_launcher_log(args.output_root, job, base_split_index, start_line)
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                text=True,
            )
            if lock_path:
                write_json(
                    gpu_lock_path(args.output_root, int(run_device.split(":")[-1])),
                    {
                        "pid": process.pid,
                        "task_id": job["task_id"],
                        "mode": job.get("mode"),
                        "base_split_index": base_split_index,
                        "gpu_index": int(run_device.split(":")[-1]),
                        "stage": job["stage"],
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
            active_job = dict(job)
            active_job["process"] = process
            active_job["lock_path"] = lock_path
            started_job = True
            print(start_line, flush=True)
            break

        if started_job:
            continue

        if blocked_count and not active_job:
            print(
                f"Waiting {args.poll_seconds}s for cache/GPU availability. pending_jobs={len(pending_jobs)}",
                flush=True,
            )
            time.sleep(args.poll_seconds)
            continue

        if not pending_jobs:
            break

    print("All queued LP/MLP jobs finished.", flush=True)


if __name__ == "__main__":
    main()
