from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.analyze_outliers_baseline import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current and weighted calibrated baseline runs."
    )
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--weighted-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected(summary: dict[str, Any]) -> dict[str, Any]:
    if "selected_calibrated_baseline" in summary:
        return summary["selected_calibrated_baseline"]
    if "selected_baseline" in summary:
        return summary["selected_baseline"]
    raise KeyError("summary has no selected baseline block")


def metric_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    block = selected(summary)
    test = block["test"]
    return {
        "model": name,
        "target_fp_rate": block.get("target_fp_rate"),
        "threshold": block["threshold"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "test_normal_false_positive_rate": test["normal_false_positive_rate"],
    }


def load_sweep(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "model", model_name)
    return df


def class_recall_rows(name: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in selected(summary).get("test_class_recall", []):
        rows.append(
            {
                "model": name,
                "class_name": row["class_name"],
                "support": row["support"],
                "detected_by_global": row["detected_by_global"],
                "recall_after_global_detection": row["recall_after_global_detection"],
            }
        )
    return rows


def choose_winner(rows: pd.DataFrame) -> dict[str, Any]:
    current = rows[rows["model"] == "current_calibrated"].iloc[0]
    weighted = rows[rows["model"] == "weighted_calibrated"].iloc[0]
    improved_f1 = float(weighted["test_f1"]) > float(current["test_f1"])
    improved_recall = float(weighted["test_recall"]) > float(current["test_recall"])
    fp_not_worse = (
        float(weighted["test_normal_false_positive_rate"])
        <= float(current["test_normal_false_positive_rate"])
    )
    winner = (
        "weighted_calibrated"
        if (improved_f1 or improved_recall) and fp_not_worse
        else "current_calibrated"
    )
    return {
        "winner": winner,
        "weighted_improved_f1": improved_f1,
        "weighted_improved_recall": improved_recall,
        "weighted_fp_not_worse": fp_not_worse,
    }


def write_report(
    output_dir: Path,
    metric_rows: pd.DataFrame,
    class_rows: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Calibrated Baseline Comparison",
        "",
        f"- Winner: {decision['winner']}",
        f"- Weighted improved F1: {decision['weighted_improved_f1']}",
        f"- Weighted improved recall: {decision['weighted_improved_recall']}",
        f"- Weighted FP not worse: {decision['weighted_fp_not_worse']}",
        "",
        "## Test Metrics",
        "",
    ]
    lines.extend(markdown_table(metric_rows))
    lines.extend(["", "## Class Recall", ""])
    lines.extend(markdown_table(class_rows))
    lines.append("")
    (output_dir / "comparison_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    current_summary = load_json(args.current_dir / "calibrated_summary.json")
    weighted_summary = load_json(args.weighted_dir / "calibrated_summary.json")
    residual_summary = load_json(args.residual_dir / "baseline_summary.json")

    metrics = pd.DataFrame(
        [
            metric_row("current_calibrated", current_summary),
            metric_row("weighted_calibrated", weighted_summary),
            metric_row("residual_fp_constrained", residual_summary),
        ]
    )
    metrics.to_csv(args.output_dir / "comparison_metrics.csv", index=False)

    sweeps = pd.concat(
        [
            load_sweep(args.current_dir / "calibrated_threshold_sweep.csv", "current_calibrated"),
            load_sweep(args.weighted_dir / "calibrated_threshold_sweep.csv", "weighted_calibrated"),
            load_sweep(args.residual_dir / "baseline_threshold_sweep.csv", "residual_fp_constrained"),
        ],
        ignore_index=True,
    )
    sweeps.to_csv(args.output_dir / "comparison_threshold_tradeoff.csv", index=False)

    class_recall = pd.DataFrame(
        class_recall_rows("current_calibrated", current_summary)
        + class_recall_rows("weighted_calibrated", weighted_summary)
        + class_recall_rows("residual_fp_constrained", residual_summary)
    )
    class_recall.to_csv(args.output_dir / "comparison_class_recall.csv", index=False)

    decision = choose_winner(metrics)
    payload = {
        "decision": decision,
        "metrics": metrics.to_dict(orient="records"),
        "outputs": {
            "comparison_metrics": "comparison_metrics.csv",
            "comparison_threshold_tradeoff": "comparison_threshold_tradeoff.csv",
            "comparison_class_recall": "comparison_class_recall.csv",
            "comparison_report": "comparison_report.md",
        },
    }
    (args.output_dir / "comparison_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir, metrics, class_recall, decision)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
