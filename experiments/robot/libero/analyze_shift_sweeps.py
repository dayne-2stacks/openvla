"""
analyze_shift_sweeps.py

Aggregates metrics.json outputs from run_libero_eval.py / run_shift_sweep.py and generates:
  - per-shift severity curves with error bars,
  - per-shift CSV tables,
  - Markdown and JSON summaries including failure threshold extraction.
"""

import csv
import glob
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import draccus
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.robot.robot_utils import DATE_TIME


@dataclass
class AnalyzeConfig:
    # fmt: off
    metrics_glob: str = "./experiments/logs/*.metrics.json"
    task_suite_name: str = "libero_spatial"
    shift_names: List[str] = field(default_factory=list)   # Optional filter; empty = no filter
    checkpoint_substring: Optional[str] = None
    output_dir: Optional[str] = None
    # fmt: on


def _infer_default_output_dir(metrics_glob_pattern: str) -> Path:
    base_prefix = metrics_glob_pattern.split("*", 1)[0]
    base_dir = Path(base_prefix).parent if base_prefix != "" else Path("./experiments/logs")
    return base_dir / "analysis" / DATE_TIME


def _resolve_sweep_severity(record: Dict) -> Optional[int]:
    if record.get("sweep_severity") is not None:
        return int(record["sweep_severity"])
    if record.get("shift_name") == "none":
        return 0
    if record.get("severity") is not None:
        return int(record["severity"])
    return None


def _parse_shift_name_from_run_id_note(run_id_note: Optional[str]) -> Optional[str]:
    if run_id_note is None or "__" not in run_id_note:
        return None
    # Expected format: <sweep_name>__<shift_name>_s<sweep_severity>_seed<seed>
    suffix = run_id_note.split("__", 1)[1]
    if "_s" not in suffix:
        return None
    return suffix.split("_s", 1)[0]


def _resolve_group_shift_name(record: Dict) -> str:
    parsed_shift = _parse_shift_name_from_run_id_note(record.get("run_id_note"))
    if parsed_shift is not None:
        return parsed_shift
    shift_name = record.get("shift_name", "unknown")
    if shift_name == "none":
        return "baseline"
    return shift_name


def _load_metric_records(pattern: str) -> List[Dict]:
    files = sorted(glob.glob(pattern))
    records = []
    for filepath in files:
        with open(filepath, "r") as metrics_file:
            record = json.load(metrics_file)
        record["_metrics_path"] = filepath
        records.append(record)
    return records


def _filter_records(records: List[Dict], cfg: AnalyzeConfig) -> List[Dict]:
    filtered = []
    for record in records:
        if record.get("task_suite_name") != cfg.task_suite_name:
            continue
        if cfg.checkpoint_substring is not None:
            checkpoint = record.get("pretrained_checkpoint", "")
            if cfg.checkpoint_substring not in checkpoint:
                continue
        group_shift_name = _resolve_group_shift_name(record)
        if len(cfg.shift_names) > 0 and group_shift_name not in cfg.shift_names:
            continue
        sweep_severity = _resolve_sweep_severity(record)
        if sweep_severity is None:
            continue
        record["_group_shift_name"] = group_shift_name
        record["_resolved_sweep_severity"] = sweep_severity
        filtered.append(record)
    return filtered


def _compute_group_stats(records: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for record in records:
        key = (record["_group_shift_name"], record["_resolved_sweep_severity"])
        grouped[key].append(float(record["total_success_rate"]))

    shift_to_rows = defaultdict(list)
    for (shift_name, severity), values in grouped.items():
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        stderr = std / math.sqrt(n) if n > 0 else 0.0
        shift_to_rows[shift_name].append(
            {
                "severity": int(severity),
                "n": n,
                "mean_success_rate": mean,
                "std_success_rate": std,
                "stderr_success_rate": stderr,
            }
        )

    for shift_name in shift_to_rows:
        shift_to_rows[shift_name] = sorted(shift_to_rows[shift_name], key=lambda row: row["severity"])
    return dict(shift_to_rows)


def _write_shift_csv(output_dir: Path, shift_name: str, rows: List[Dict]) -> Path:
    csv_path = output_dir / f"{shift_name}_summary.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["severity", "n", "mean_success_rate", "std_success_rate", "stderr_success_rate"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _plot_shift_curve(output_dir: Path, shift_name: str, rows: List[Dict]) -> Path:
    severities = [row["severity"] for row in rows]
    means = [row["mean_success_rate"] for row in rows]
    stderrs = [row["stderr_success_rate"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(severities, means, yerr=stderrs, fmt="-o", capsize=4, linewidth=2)
    ax.set_title(f"{shift_name}: Success Rate vs Severity")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Mean Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(sorted(set(severities)))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / f"{shift_name}_severity_curve.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def _extract_thresholds(rows: List[Dict], failure_cutoff: float = 0.30) -> Dict:
    first_crossing = None
    for row in rows:
        if row["mean_success_rate"] < failure_cutoff:
            first_crossing = row["severity"]
            break

    worst_row = min(rows, key=lambda row: row["mean_success_rate"])
    return {
        "failure_cutoff": failure_cutoff,
        "first_crossing_threshold": first_crossing,
        "worst_case_severity": worst_row["severity"],
        "worst_case_success_rate": worst_row["mean_success_rate"],
    }


def _write_markdown_summary(output_dir: Path, shift_to_rows: Dict[str, List[Dict]], threshold_summary: Dict[str, Dict]) -> Path:
    summary_path = output_dir / "summary.md"
    lines = ["# Shift Sweep Summary", ""]
    for shift_name, rows in sorted(shift_to_rows.items()):
        lines.append(f"## {shift_name}")
        lines.append("")
        lines.append("| Severity | n | Mean ± Std | StdErr |")
        lines.append("|---:|---:|---:|---:|")
        for row in rows:
            mean_std = f"{row['mean_success_rate']:.4f} ± {row['std_success_rate']:.4f}"
            lines.append(f"| {row['severity']} | {row['n']} | {mean_std} | {row['stderr_success_rate']:.4f} |")
        th = threshold_summary[shift_name]
        lines.append("")
        lines.append(f"- First crossing (< {th['failure_cutoff']:.2f}): {th['first_crossing_threshold']}")
        lines.append(f"- Worst-case severity: {th['worst_case_severity']}")
        lines.append(f"- Worst-case success rate: {th['worst_case_success_rate']:.4f}")
        lines.append("")

    with open(summary_path, "w") as summary_file:
        summary_file.write("\n".join(lines))
    return summary_path


@draccus.wrap()
def analyze_shift_sweeps(cfg: AnalyzeConfig) -> None:
    records = _load_metric_records(cfg.metrics_glob)
    if len(records) == 0:
        raise ValueError(f"No metrics files matched: {cfg.metrics_glob}")

    filtered = _filter_records(records, cfg)
    if len(filtered) == 0:
        raise ValueError("No metrics records remained after filtering.")

    shift_to_rows = _compute_group_stats(filtered)
    if len(shift_to_rows) == 0:
        raise ValueError("No grouped rows found after aggregation.")

    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = _infer_default_output_dir(cfg.metrics_glob)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_summary = {}
    artifacts = {}
    for shift_name, rows in sorted(shift_to_rows.items()):
        csv_path = _write_shift_csv(output_dir, shift_name, rows)
        plot_path = _plot_shift_curve(output_dir, shift_name, rows)
        threshold_summary[shift_name] = _extract_thresholds(rows)
        artifacts[shift_name] = {
            "csv_path": str(csv_path),
            "plot_path": str(plot_path),
        }

    summary_md_path = _write_markdown_summary(output_dir, shift_to_rows, threshold_summary)
    summary_json_path = output_dir / "summary.json"
    summary_payload = {
        "task_suite_name": cfg.task_suite_name,
        "metrics_glob": cfg.metrics_glob,
        "num_runs_filtered": len(filtered),
        "threshold_summary": threshold_summary,
        "artifacts": artifacts,
    }
    with open(summary_json_path, "w") as summary_json_file:
        json.dump(summary_payload, summary_json_file, indent=2, sort_keys=True)

    print(f"Analysis complete. Output dir: {output_dir}")
    for shift_name, values in threshold_summary.items():
        print(
            f"{shift_name}: first_crossing={values['first_crossing_threshold']}, "
            f"worst_case_severity={values['worst_case_severity']}, "
            f"worst_case_success_rate={values['worst_case_success_rate']:.4f}"
        )
    print(f"Summary markdown: {summary_md_path}")
    print(f"Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    analyze_shift_sweeps()
