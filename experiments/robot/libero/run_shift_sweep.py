"""
run_shift_sweep.py

Launches deterministic LIBERO shift sweeps by repeatedly invoking run_libero_eval.py.

Example:
python experiments/robot/libero/run_shift_sweep.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 5
"""

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import draccus

from experiments.robot.robot_utils import DATE_TIME

RECOMMENDED_VERSIONS = {
    "python": "3.10.13",
    "torch": "2.2.0",
    "transformers": "4.40.1",
    "flash-attn": "2.5.5",
}


@dataclass
class SweepConfig:
    # fmt: off

    #################################################################################################################
    # Base eval configuration
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    task_suite_name: str = "libero_spatial"
    center_crop: bool = True
    num_trials_per_task: int = 5
    num_steps_wait: int = 10

    #################################################################################################################
    # Sweep axes
    #################################################################################################################
    shift_names: List[str] = field(default_factory=lambda: ["appearance"])
    sweep_severities: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    #################################################################################################################
    # Logging
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"
    sweep_name: str = "shift_sweep"
    manifest_path: Optional[str] = None

    #################################################################################################################
    # Execution
    #################################################################################################################
    fail_fast: bool = False
    dry_run: bool = False

    #################################################################################################################
    # Optional eval passthrough
    #################################################################################################################
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # fmt: on


def _get_pkg_version(pkg_names: List[str]) -> Optional[str]:
    for pkg in pkg_names:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def print_version_guardrails() -> None:
    detected = {
        "python": platform.python_version(),
        "torch": _get_pkg_version(["torch"]),
        "transformers": _get_pkg_version(["transformers"]),
        "flash-attn": _get_pkg_version(["flash-attn", "flash_attn"]),
    }
    print("Detected versions:")
    for key, value in detected.items():
        print(f"  - {key}: {value}")
    print("Recommended versions (from OpenVLA README):")
    for key, value in RECOMMENDED_VERSIONS.items():
        detected_value = detected.get(key)
        mismatch = detected_value is not None and detected_value != value
        suffix = " [WARNING: mismatch]" if mismatch else ""
        print(f"  - {key}: {value}{suffix}")


def _validate_sweep_cfg(cfg: SweepConfig) -> None:
    if cfg.pretrained_checkpoint == "":
        raise ValueError("Expected pretrained_checkpoint to be set.")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Cannot use both load_in_8bit and load_in_4bit.")
    if len(cfg.shift_names) == 0:
        raise ValueError("Expected at least one shift_name.")
    if len(cfg.sweep_severities) == 0:
        raise ValueError("Expected at least one sweep severity.")
    if len(cfg.seeds) == 0:
        raise ValueError("Expected at least one seed.")
    for severity in cfg.sweep_severities:
        if severity < 0 or severity > 4:
            raise ValueError(f"Expected sweep severity in [0, 4], got {severity}.")


def _map_eval_shift(shift_name: str, sweep_severity: int) -> Tuple[str, str, int]:
    """
    Maps external sweep severity (0..4) to eval config:
      - 0 -> baseline (shift_name=none, severity=1 unused)
      - 1..4 -> perturbed run
    """
    if sweep_severity == 0:
        return "none", "noise_blur_gamma", 1

    if shift_name == "appearance":
        return "appearance", "noise_blur_gamma", sweep_severity

    raise NotImplementedError(
        f"Shift '{shift_name}' is not implemented yet in run_libero_eval.py. "
        "Currently supported for non-zero severity: appearance."
    )


def _extract_metrics_path(stdout: str, stderr: str) -> Optional[str]:
    marker = "Saved metrics JSON at path "
    for stream in (stdout, stderr):
        for line in stream.splitlines():
            if marker in line:
                return line.split(marker, 1)[1].strip()
    return None


@draccus.wrap()
def run_shift_sweep(cfg: SweepConfig) -> None:
    _validate_sweep_cfg(cfg)
    print_version_guardrails()

    repo_root = Path(__file__).resolve().parents[3]
    eval_script = repo_root / "experiments/robot/libero/run_libero_eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Could not find eval script at {eval_script}")

    if cfg.manifest_path is None:
        manifest_path = Path(cfg.local_log_dir) / "sweeps" / f"{DATE_TIME}-manifest.jsonl"
    else:
        manifest_path = Path(cfg.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_runs = len(cfg.shift_names) * len(cfg.sweep_severities) * len(cfg.seeds)
    print(f"Planned runs: {total_runs}")

    run_idx = 0
    for shift_name in cfg.shift_names:
        for sweep_severity in cfg.sweep_severities:
            for seed in cfg.seeds:
                run_idx += 1
                effective_shift_name, shift_mode, severity = _map_eval_shift(shift_name, sweep_severity)
                run_id_note = f"{cfg.sweep_name}__{shift_name}_s{sweep_severity}_seed{seed}"

                command = [
                    sys.executable,
                    str(eval_script.relative_to(repo_root)),
                    "--model_family",
                    cfg.model_family,
                    "--pretrained_checkpoint",
                    cfg.pretrained_checkpoint,
                    "--task_suite_name",
                    cfg.task_suite_name,
                    "--center_crop",
                    str(cfg.center_crop),
                    "--num_trials_per_task",
                    str(cfg.num_trials_per_task),
                    "--num_steps_wait",
                    str(cfg.num_steps_wait),
                    "--shift_name",
                    effective_shift_name,
                    "--shift_mode",
                    shift_mode,
                    "--severity",
                    str(severity),
                    "--sweep_severity",
                    str(sweep_severity),
                    "--seed",
                    str(seed),
                    "--run_id_note",
                    run_id_note,
                    "--local_log_dir",
                    cfg.local_log_dir,
                    "--use_wandb",
                    str(cfg.use_wandb),
                    "--wandb_project",
                    cfg.wandb_project,
                    "--wandb_entity",
                    cfg.wandb_entity,
                    "--load_in_8bit",
                    str(cfg.load_in_8bit),
                    "--load_in_4bit",
                    str(cfg.load_in_4bit),
                ]

                print(f"\n[{run_idx}/{total_runs}] Launching run: {run_id_note}")
                print("Command:", " ".join(command))

                run_record = {
                    "run_index": run_idx,
                    "total_runs": total_runs,
                    "run_id_note": run_id_note,
                    "shift_name": shift_name,
                    "effective_shift_name": effective_shift_name,
                    "shift_mode": shift_mode,
                    "severity": severity,
                    "sweep_severity": sweep_severity,
                    "seed": seed,
                    "command": command,
                }
                start_time_utc = datetime.now(timezone.utc).isoformat()
                run_record["start_time_utc"] = start_time_utc
                start_monotonic = time.monotonic()

                if cfg.dry_run:
                    run_record.update(
                        {
                            "status": "dry_run",
                            "return_code": None,
                            "metrics_path": None,
                            "end_time_utc": datetime.now(timezone.utc).isoformat(),
                            "duration_seconds": max(time.monotonic() - start_monotonic, 0.0),
                        }
                    )
                    with open(manifest_path, "a") as manifest_file:
                        manifest_file.write(json.dumps(run_record) + "\n")
                    continue

                process = subprocess.run(
                    command,
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                )
                end_time_utc = datetime.now(timezone.utc).isoformat()
                duration_seconds = max(time.monotonic() - start_monotonic, 0.0)
                metrics_path = _extract_metrics_path(process.stdout, process.stderr)
                run_record.update(
                    {
                        "status": "ok" if process.returncode == 0 else "failed",
                        "return_code": process.returncode,
                        "metrics_path": metrics_path,
                        "end_time_utc": end_time_utc,
                        "duration_seconds": duration_seconds,
                    }
                )

                with open(manifest_path, "a") as manifest_file:
                    manifest_file.write(json.dumps(run_record) + "\n")

                if process.returncode != 0:
                    print("Run failed.")
                    print(process.stderr)
                    if cfg.fail_fast:
                        raise RuntimeError(f"Run failed with return code {process.returncode}: {run_id_note}")
                else:
                    print(f"Run completed. metrics_path={metrics_path}")

    print(f"\nSweep complete. Manifest written to: {manifest_path}")


if __name__ == "__main__":
    run_shift_sweep()
