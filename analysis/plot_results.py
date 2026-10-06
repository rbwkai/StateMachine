"""Create publication-style plots from DWS-Bench audit CSV files.

Example:
    python analysis/plot_results.py --input results --output results/plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    # Run as a script, sys.path[0] is analysis/, whose statistics.py shadows
    # the stdlib module.
    _SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != _SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# One resolver for the run_eval output layout (AGENTS.md §5); eval-free (§4).
from analysis.run_layout import infer_condition  # noqa: E402


def load_rows(input_path: Path) -> List[Dict[str, Any]]:
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*_audit.csv"))
    rows: List[Dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                # The parent directory is "<cot|no_cot>_<tokens>", not the model.
                condition = infer_condition(path, row)
                row["model"] = condition["model"]
                row["cot"] = condition["cot"]
                row["max_new_tokens"] = condition["max_new_tokens"]
                row["condition"] = condition["condition"]
                row["requested"] = json.loads(row.get("requested_factors", "{}") or "{}")
                row["measured"] = json.loads(row.get("measured_factors", "{}") or "{}")
                row["correct"] = row.get("is_correct", "").lower() == "true"
                raw_step_accuracy = row.get("step_accuracy", "")
                row["step_accuracy_value"] = float(raw_step_accuracy) if raw_step_accuracy else None
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No *_audit.csv files found under {input_path}")
    return rows


def grouped_accuracy(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, tuple[int, int]]:
    grouped: Dict[str, tuple[int, int]] = {}
    for row in rows:
        value = str(row["requested"].get(key, row["measured"].get(key, "unknown")))
        correct, total = grouped.get(value, (0, 0))
        grouped[value] = (correct + int(row["correct"]), total + 1)
    return grouped


def add_accuracy_labels(ax: Any, bars: Any) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def make_plots(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#0F766E", "#E07A5F", "#3D405B", "#81B29A", "#F2CC8F", "#6D597A"]

    # Series are run conditions (model, CoT mode, budget) so CoT and non-CoT
    # runs of one model are not pooled into a single bar or curve.
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["condition"]].append(row)

    # Overall and step-wise accuracy by model.
    models = sorted(by_model)
    final_values = [sum(r["correct"] for r in by_model[m]) / len(by_model[m]) for m in models]
    step_values = [
        np.mean([r["step_accuracy_value"] for r in by_model[m] if r["step_accuracy_value"] is not None])
        if any(r["step_accuracy_value"] is not None for r in by_model[m]) else np.nan
        for m in models
    ]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.36
    bars = ax.bar(x - width / 2, final_values, width, label="Final answer", color=colors[0])
    ax.bar(x + width / 2, step_values, width, label="Step-wise", color=colors[1])
    add_accuracy_labels(ax, bars)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("DWS-Bench Evaluation Overview", loc="left", weight="bold")
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "overview_accuracy.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    for factor, title, filename in [
        ("T", "RQ1: Accuracy by Target Update Depth", "rq1_depth_accuracy.png"),
        ("D", "RQ3: Accuracy by State-Changing Distractors", "rq3_distractor_accuracy.png"),
        ("E", "RQ4: Accuracy by Entity Load", "rq4_entity_load_accuracy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for index, model in enumerate(models):
            grouped = grouped_accuracy(by_model[model], factor)
            values = sorted((float(k), correct / total) for k, (correct, total) in grouped.items() if k != "unknown")
            if not values:
                continue
            ax.plot(
                [value for value, _ in values],
                [accuracy for _, accuracy in values],
                marker="o",
                linewidth=2.2,
                label=model,
                color=colors[index % len(colors)],
            )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(factor)
        ax.set_ylabel("Final-answer accuracy")
        ax.set_title(title, loc="left", weight="bold")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=240, bbox_inches="tight")
        plt.close(fig)

    error_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("step_first_error"):
            error_counts["Step-wise error"] += 1
        elif not row["correct"]:
            error_counts["Final-answer error"] += 1
        else:
            error_counts["Correct"] += 1
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(error_counts)
    values = [error_counts[label] for label in labels]
    ax.bar(labels, values, color=[colors[2], colors[1], colors[0]])
    ax.set_ylabel("Queries")
    ax.set_title("Outcome and First-Error Audit", loc="left", weight="bold")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "error_audit.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DWS-Bench evaluation audit CSVs.")
    parser.add_argument("--input", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    make_plots(load_rows(args.input), args.output)
    print(f"Wrote plots to {args.output}")


if __name__ == "__main__":
    main()