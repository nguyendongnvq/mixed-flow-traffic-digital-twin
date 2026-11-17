from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.analyze_global_threshold import binary_metrics


DEFAULT_FP_TARGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CLASS_NAMES = ["accident", "congestion_spike", "zigzag", "sudden_stop"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze residual outliers and build an FP-constrained global anomaly baseline."
    )
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fp-rate", type=float, default=0.25)
    parser.add_argument(
        "--fp-targets",
        default=",".join(str(value) for value in DEFAULT_FP_TARGETS),
        help="Comma-separated validation normal FP-rate targets.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def parse_fp_targets(raw: str) -> list[float]:
    targets = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not targets:
        raise ValueError("--fp-targets must include at least one value")
    for target in targets:
        if target < 0.0 or target > 1.0:
            raise ValueError(f"FP target must be between 0 and 1, got {target}")
    return sorted(set(targets))


def choose_fp_constrained_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fp_rate: float,
) -> tuple[float, dict[str, float | int]]:
    if target_fp_rate < 0.0 or target_fp_rate > 1.0:
        raise ValueError("target_fp_rate must be between 0 and 1")
    labels_bool = labels.astype(bool)
    scores_float = scores.astype(float)
    candidates = np.unique(scores_float)
    candidates = np.concatenate(
        [
            [np.nextafter(float(candidates.min()), -np.inf)],
            candidates,
            [np.nextafter(float(candidates.max()), np.inf)],
        ]
    )
    feasible: list[tuple[tuple[float, float, float], float, dict[str, float | int]]] = []
    fallback: list[tuple[tuple[float, float, float], float, dict[str, float | int]]] = []
    for threshold in candidates:
        metrics = binary_metrics(labels_bool, scores_float >= threshold)
        fp_rate = float(metrics["normal_false_positive_rate"])
        threshold_float = float(threshold)
        if fp_rate <= target_fp_rate:
            rank = (
                float(metrics["f1"]),
                float(metrics["recall"]),
                -fp_rate,
            )
            feasible.append((rank, threshold_float, metrics))
        else:
            rank = (
                -fp_rate,
                float(metrics["f1"]),
                float(metrics["recall"]),
            )
            fallback.append((rank, threshold_float, metrics))
    rows = feasible if feasible else fallback
    _, threshold, metrics = max(rows, key=lambda item: item[0])
    return threshold, metrics


def metrics_at_threshold(df: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = df["is_anomaly_true"].to_numpy(dtype=bool)
    scores = df["global_anomaly_score"].to_numpy(dtype=float)
    return binary_metrics(labels, scores >= threshold)


def class_recall_rows(df: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    predictions = df["global_anomaly_score"].to_numpy(dtype=float) >= threshold
    rows: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        if class_name not in df.columns:
            continue
        truth = df[class_name].to_numpy(dtype=bool)
        support = int(truth.sum())
        detected = int(np.sum(truth & predictions))
        rows.append(
            {
                "class_name": class_name,
                "support": support,
                "detected_by_global": detected,
                "recall_after_global_detection": detected / support if support else 0.0,
            }
        )
    return rows


def build_threshold_sweep(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fp_targets: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = val_df["is_anomaly_true"].to_numpy(dtype=bool)
    scores = val_df["global_anomaly_score"].to_numpy(dtype=float)
    for target in fp_targets:
        threshold, val_metrics = choose_fp_constrained_threshold(
            labels,
            scores,
            target,
        )
        test_metrics = metrics_at_threshold(test_df, threshold)
        row: dict[str, Any] = {
            "target_fp_rate": target,
            "threshold": threshold,
        }
        for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def score_summary_rows(df: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    groups: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=df.index)),
        ("true_normal", df["is_anomaly_true"] == 0),
        ("true_anomaly", df["is_anomaly_true"] == 1),
    ]
    for class_name in CLASS_NAMES:
        if class_name in df.columns:
            groups.append((f"true_{class_name}", df[class_name] == 1))
    if "primary_label" in df.columns:
        for label in sorted(df["primary_label"].dropna().astype(str).unique()):
            groups.append((f"primary_{label}", df["primary_label"].astype(str) == label))

    rows: list[dict[str, Any]] = []
    for group, mask in groups:
        scores = df.loc[mask, "global_anomaly_score"].astype(float)
        if scores.empty:
            continue
        row: dict[str, Any] = {
            "split": split,
            "group": group,
            "count": int(scores.size),
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=0)),
            "min": float(scores.min()),
            "p25": float(scores.quantile(0.25)),
            "p50": float(scores.quantile(0.50)),
            "p75": float(scores.quantile(0.75)),
            "p90": float(scores.quantile(0.90)),
            "p95": float(scores.quantile(0.95)),
            "p99": float(scores.quantile(0.99)),
            "max": float(scores.max()),
        }
        rows.append(row)
    return rows


def apply_baseline_predictions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = df.copy()
    predictions = output["global_anomaly_score"].to_numpy(dtype=float) >= threshold
    output["is_anomaly_pred"] = predictions
    output["status"] = np.where(
        ~predictions,
        "normal",
        np.where(output["primary_label"].astype(str) == "unknown_anomaly", "unknown_anomaly", "known_anomaly"),
    )
    return output


def load_reference_summary(pred_dir: Path) -> dict[str, Any] | None:
    path = pred_dir / "global_threshold_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    score_summary: pd.DataFrame,
) -> None:
    selected = summary["selected_baseline"]
    reference = summary.get("max_f1_reference")
    lines = [
        "# Outlier Baseline Report",
        "",
        "## Selected Baseline",
        "",
        f"- Target validation normal FP rate: {selected['target_fp_rate']:.3f}",
        f"- Threshold: {selected['threshold']:.6f}",
        f"- Test precision: {selected['test']['precision']:.4f}",
        f"- Test recall: {selected['test']['recall']:.4f}",
        f"- Test F1: {selected['test']['f1']:.4f}",
        f"- Test normal FP rate: {selected['test']['normal_false_positive_rate']:.4f}",
        "",
    ]
    if reference is not None:
        lines.extend(
            [
                "## Max-F1 Reference",
                "",
                f"- Threshold: {reference['threshold']:.6f}",
                f"- Test precision: {reference['test']['precision']:.4f}",
                f"- Test recall: {reference['test']['recall']:.4f}",
                f"- Test F1: {reference['test']['f1']:.4f}",
                f"- Test normal FP rate: {reference['test']['normal_false_positive_rate']:.4f}",
                "",
            ]
        )
    lines.extend(["## Score Summary", ""])
    compact = score_summary[
        score_summary["group"].isin(["true_normal", "true_anomaly"])
    ][["split", "group", "count", "p50", "p75", "p90", "p95", "max"]]
    lines.extend(markdown_table(compact))
    lines.append("")
    lines.append(
        "Baseline detection uses one global residual threshold; class labels remain auxiliary classification output."
    )
    (output_dir / "outlier_baseline_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def markdown_table(df: pd.DataFrame) -> list[str]:
    headers = [str(column) for column in df.columns]
    rows = [headers, ["---"] * len(headers)]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append(values)
    return ["| " + " | ".join(row) + " |" for row in rows]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    fp_targets = parse_fp_targets(args.fp_targets)
    val_df = pd.read_csv(args.pred_dir / "val_global_threshold_predictions.csv")
    test_df = pd.read_csv(args.pred_dir / "test_global_threshold_predictions.csv")
    inference_df = pd.read_csv(args.inference_csv)

    threshold_sweep = build_threshold_sweep(val_df, test_df, fp_targets)
    threshold_sweep.to_csv(args.output_dir / "baseline_threshold_sweep.csv", index=False)

    labels = val_df["is_anomaly_true"].to_numpy(dtype=bool)
    scores = val_df["global_anomaly_score"].to_numpy(dtype=float)
    selected_threshold, selected_val_metrics = choose_fp_constrained_threshold(
        labels,
        scores,
        args.target_fp_rate,
    )
    selected_test_metrics = metrics_at_threshold(test_df, selected_threshold)

    baseline_test = apply_baseline_predictions(test_df, selected_threshold)
    baseline_test.to_csv(args.output_dir / "baseline_test_predictions.csv", index=False)

    false_positives = baseline_test[
        (baseline_test["is_anomaly_true"] == 0) & baseline_test["is_anomaly_pred"]
    ].sort_values("global_anomaly_score", ascending=False)
    false_positives.head(args.top_k).to_csv(
        args.output_dir / "top_false_positives.csv",
        index=False,
    )
    false_negatives = baseline_test[
        (baseline_test["is_anomaly_true"] == 1) & ~baseline_test["is_anomaly_pred"]
    ].sort_values("global_anomaly_score", ascending=True)
    false_negatives.head(args.top_k).to_csv(
        args.output_dir / "top_false_negatives.csv",
        index=False,
    )

    score_summary = pd.DataFrame(
        score_summary_rows(val_df, "val") + score_summary_rows(test_df, "test")
    )
    score_summary.to_csv(args.output_dir / "outlier_score_summary.csv", index=False)

    reference_summary = load_reference_summary(args.pred_dir)
    max_f1_reference = None
    if reference_summary is not None:
        max_f1_reference = {
            "threshold": float(reference_summary["global_threshold"]),
            "selection_metric": reference_summary.get("selection_metric", "unknown"),
            "validation": reference_summary.get("validation"),
            "test": reference_summary.get("splits", {}).get("test"),
        }

    status_counts = baseline_test["status"].value_counts().to_dict()
    inference_status_counts = (
        inference_df["status"].value_counts().to_dict()
        if "status" in inference_df.columns
        else {}
    )
    summary: dict[str, Any] = {
        "pred_dir": str(args.pred_dir),
        "inference_csv": str(args.inference_csv),
        "selected_baseline": {
            "target_fp_rate": float(args.target_fp_rate),
            "threshold": float(selected_threshold),
            "validation": selected_val_metrics,
            "test": selected_test_metrics,
            "test_class_recall": class_recall_rows(test_df, selected_threshold),
            "test_status_counts": status_counts,
        },
        "max_f1_reference": max_f1_reference,
        "input_inference_status_counts": inference_status_counts,
        "outputs": {
            "baseline_threshold_sweep": "baseline_threshold_sweep.csv",
            "outlier_score_summary": "outlier_score_summary.csv",
            "top_false_positives": "top_false_positives.csv",
            "top_false_negatives": "top_false_negatives.csv",
            "baseline_test_predictions": "baseline_test_predictions.csv",
            "report": "outlier_baseline_report.md",
        },
    }
    (args.output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir, summary, score_summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
