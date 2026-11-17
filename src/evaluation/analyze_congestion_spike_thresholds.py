"""Analyze congestion-spike anomaly thresholds and feature importance.

This script compares:
- congestion_spike_only dataset
- normal_only dataset

It infers positive/negative labels, finds single-feature thresholds based on an
optimization objective, and estimates multifeature linear weights via
standardized logistic regression.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPIKE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "congestion"
    / "forecast_congestion_spike_node_level_congestion_spike_only.csv"
)
DEFAULT_NORMAL_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "congestion"
    / "forecast_congestion_spike_node_level_normal_only.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "congestion"
DEFAULT_FEATURES = [
    "flow_count_5m",
    "speed",
    "e1_occupancy",
    "e2_meanOccupancy",
    "jam_veh",
]


@dataclass
class ThresholdResult:
    feature: str
    operator: str
    threshold: float
    tpr: float
    tnr: float
    precision: float
    recall: float
    fpr: float
    f1: float
    balanced_accuracy: float
    score: float
    objective: str
    tp: int
    tn: int
    fp: int
    fn: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze congestion-spike anomaly thresholds and feature importance."
    )
    parser.add_argument(
        "--spike-csv",
        type=Path,
        default=DEFAULT_SPIKE_PATH,
        help="Path to congestion_spike_only CSV.",
    )
    parser.add_argument(
        "--normal-csv",
        type=Path,
        default=DEFAULT_NORMAL_PATH,
        help="Path to normal_only CSV.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature list to evaluate.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="high_recall",
        choices=["high_recall", "balanced", "high_precision"],
        help="Objective to optimize single-feature threshold.",
    )
    parser.add_argument(
        "--quantile-grid-size",
        type=int,
        default=400,
        help="Grid size for quantile-based threshold search.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write analysis artifacts.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="congestion_spike_anomaly_analysis",
        help="Prefix for output files.",
    )
    return parser.parse_args()


def parse_features(raw: str) -> list[str]:
    out = [x.strip() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("No features supplied.")
    return out


def load_and_label(spike_csv: Path, normal_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spike_df = pd.read_csv(spike_csv)
    normal_df = pd.read_csv(normal_csv)

    has_is_anomaly = "is_anomaly" in spike_df.columns
    has_positive = False
    if has_is_anomaly:
        values = pd.to_numeric(spike_df["is_anomaly"], errors="coerce").fillna(0)
        has_positive = bool((values > 0).any())

    if has_positive:
        spike_numeric = pd.to_numeric(spike_df["is_anomaly"], errors="coerce").fillna(0)
        pos_df = spike_df[spike_numeric > 0].copy()
        neg_from_spike = spike_df[spike_numeric <= 0].copy()
        neg_df = pd.concat([neg_from_spike, normal_df.copy()], ignore_index=True)
        label_strategy = "spike_csv.is_anomaly==1 as positive; (spike<=0 + normal_csv) as negative"
    else:
        pos_df = spike_df.copy()
        neg_df = normal_df.copy()
        label_strategy = "all rows in spike_csv as positive; all rows in normal_csv as negative"

    if len(pos_df) == 0 or len(neg_df) == 0:
        raise ValueError(
            f"Invalid class split. positives={len(pos_df)}, negatives={len(neg_df)}."
        )

    meta = {
        "label_strategy": label_strategy,
        "spike_rows": int(len(spike_df)),
        "normal_rows": int(len(normal_df)),
        "positive_rows": int(len(pos_df)),
        "negative_rows": int(len(neg_df)),
        "spike_has_is_anomaly": bool(has_is_anomaly),
        "spike_has_positive_is_anomaly": bool(has_positive),
    }
    return pos_df, neg_df, meta


def safe_float(x: Any) -> float:
    value = float(x)
    if math.isnan(value) or math.isinf(value):
        return float("nan")
    return value


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    ba = (tpr + tnr) / 2.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "tnr": tnr,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "balanced_accuracy": ba,
    }


def objective_score(metrics: dict[str, float | int], objective: str) -> float:
    recall = float(metrics["recall"])
    precision = float(metrics["precision"])
    ba = float(metrics["balanced_accuracy"])
    fpr = float(metrics["fpr"])

    if objective == "high_recall":
        return 0.7 * recall + 0.25 * precision + 0.05 * (1.0 - fpr)
    if objective == "high_precision":
        return 0.7 * precision + 0.25 * recall + 0.05 * (1.0 - fpr)
    return ba


def find_best_threshold(
    feature: str,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    objective: str,
    quantile_grid_size: int,
) -> ThresholdResult:
    pos_values = pd.to_numeric(pos_df[feature], errors="coerce").dropna().to_numpy(dtype=float)
    neg_values = pd.to_numeric(neg_df[feature], errors="coerce").dropna().to_numpy(dtype=float)
    x = np.concatenate([pos_values, neg_values])
    y = np.concatenate([np.ones(len(pos_values), dtype=int), np.zeros(len(neg_values), dtype=int)])

    if len(x) == 0:
        raise ValueError(f"No valid numeric values for feature '{feature}'.")

    if quantile_grid_size < 10:
        raise ValueError("--quantile-grid-size must be >= 10.")
    quantiles = np.linspace(0.01, 0.99, quantile_grid_size)
    candidates = np.unique(np.quantile(x, quantiles))
    if len(candidates) == 0:
        candidates = np.unique(x)

    best: ThresholdResult | None = None
    for threshold in candidates:
        for op in [">=", "<="]:
            pred = (x >= threshold) if op == ">=" else (x <= threshold)
            m = compute_metrics(y, pred.astype(int))
            score = objective_score(m, objective=objective)
            current = ThresholdResult(
                feature=feature,
                operator=op,
                threshold=float(threshold),
                tpr=float(m["tpr"]),
                tnr=float(m["tnr"]),
                precision=float(m["precision"]),
                recall=float(m["recall"]),
                fpr=float(m["fpr"]),
                f1=float(m["f1"]),
                balanced_accuracy=float(m["balanced_accuracy"]),
                score=float(score),
                objective=objective,
                tp=int(m["tp"]),
                tn=int(m["tn"]),
                fp=int(m["fp"]),
                fn=int(m["fn"]),
            )
            if best is None or current.score > best.score:
                best = current

    if best is None:
        raise RuntimeError(f"Threshold search failed for feature '{feature}'.")
    return best


def compute_feature_stats(feature: str, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> dict[str, Any]:
    pos_values = pd.to_numeric(pos_df[feature], errors="coerce").dropna().to_numpy(dtype=float)
    neg_values = pd.to_numeric(neg_df[feature], errors="coerce").dropna().to_numpy(dtype=float)
    p_mean = float(np.mean(pos_values)) if len(pos_values) > 0 else float("nan")
    n_mean = float(np.mean(neg_values)) if len(neg_values) > 0 else float("nan")
    p_std = float(np.std(pos_values, ddof=1)) if len(pos_values) > 1 else float("nan")
    n_std = float(np.std(neg_values, ddof=1)) if len(neg_values) > 1 else float("nan")
    pooled = math.sqrt((p_std * p_std + n_std * n_std) / 2.0) if (p_std > 0 and n_std > 0) else float("nan")
    cohens_d = (p_mean - n_mean) / pooled if np.isfinite(pooled) and pooled > 0 else float("nan")

    return {
        "feature": feature,
        "positive_count": int(len(pos_values)),
        "negative_count": int(len(neg_values)),
        "positive_mean": safe_float(p_mean),
        "negative_mean": safe_float(n_mean),
        "positive_std": safe_float(p_std),
        "negative_std": safe_float(n_std),
        "positive_median": safe_float(np.median(pos_values) if len(pos_values) else float("nan")),
        "negative_median": safe_float(np.median(neg_values) if len(neg_values) else float("nan")),
        "cohens_d": safe_float(cohens_d),
    }


def fit_logistic_weights(
    features: list[str],
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for logistic weight analysis.") from exc

    pos_x = pos_df[features].copy()
    neg_x = neg_df[features].copy()
    x_df = pd.concat([pos_x, neg_x], ignore_index=True)
    y = np.concatenate([np.ones(len(pos_x), dtype=int), np.zeros(len(neg_x), dtype=int)])
    x_df = x_df.apply(pd.to_numeric, errors="coerce")
    x_df = x_df.fillna(x_df.median(numeric_only=True))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    )
    model.fit(x_df, y)
    probs = model.predict_proba(x_df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    lr = model.named_steps["logisticregression"]
    coeffs = lr.coef_[0]

    weights = []
    for feature, weight in sorted(zip(features, coeffs), key=lambda item: abs(item[1]), reverse=True):
        weights.append(
            {
                "feature": feature,
                "weight": float(weight),
                "abs_weight": float(abs(weight)),
                "direction": "increase=>anomaly" if weight > 0 else "decrease=>anomaly",
            }
        )

    metrics = {
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "intercept": float(lr.intercept_[0]),
    }
    return weights, metrics


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_markdown_report(
    output_path: Path,
    objective: str,
    label_meta: dict[str, Any],
    best_overall: ThresholdResult,
    threshold_rows: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    model_metrics: dict[str, Any],
) -> None:
    top_weight = weight_rows[0] if weight_rows else None
    lines: list[str] = []
    lines.append("# Congestion Spike Anomaly Analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Objective: `{objective}`")
    lines.append(f"- Label strategy: `{label_meta['label_strategy']}`")
    lines.append(
        f"- Samples: positive={label_meta['positive_rows']}, negative={label_meta['negative_rows']}"
    )
    lines.append(
        f"- Best single-feature rule: `{best_overall.feature} {best_overall.operator} {best_overall.threshold:.6f}`"
    )
    lines.append(
        f"- Rule quality: recall={best_overall.recall:.4f}, precision={best_overall.precision:.4f}, "
        f"balanced_accuracy={best_overall.balanced_accuracy:.4f}, score={best_overall.score:.4f}"
    )
    if top_weight is not None:
        lines.append(
            f"- Strongest linear weight: `{top_weight['feature']}` ({top_weight['weight']:.4f}, {top_weight['direction']})"
        )
    lines.append("")
    lines.append("## Single-feature Thresholds")
    lines.append("| feature | operator | threshold | recall | precision | balanced_accuracy | objective_score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(threshold_rows, key=lambda r: float(r["score"]), reverse=True):
        lines.append(
            f"| {row['feature']} | {row['operator']} | {float(row['threshold']):.6f} | "
            f"{float(row['recall']):.4f} | {float(row['precision']):.4f} | "
            f"{float(row['balanced_accuracy']):.4f} | {float(row['score']):.4f} |"
        )
    lines.append("")
    lines.append("## Standardized Logistic Weights")
    lines.append("| rank | feature | weight | abs_weight | direction |")
    lines.append("|---:|---|---:|---:|---|")
    for idx, row in enumerate(weight_rows, start=1):
        lines.append(
            f"| {idx} | {row['feature']} | {float(row['weight']):.4f} | "
            f"{float(row['abs_weight']):.4f} | {row['direction']} |"
        )
    lines.append("")
    lines.append("## Model Metrics")
    lines.append(
        f"- logistic balanced_accuracy@0.5: {float(model_metrics['balanced_accuracy_at_0_5']):.4f}"
    )
    lines.append(f"- logistic roc_auc: {float(model_metrics['roc_auc']):.4f}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    features = parse_features(args.features)

    pos_df, neg_df, label_meta = load_and_label(args.spike_csv, args.normal_csv)

    valid_features = [f for f in features if f in pos_df.columns and f in neg_df.columns]
    if not valid_features:
        raise ValueError(
            f"No requested features found in both files. requested={features}, "
            f"spike_columns={list(pos_df.columns)}"
        )

    feature_stats = [compute_feature_stats(f, pos_df, neg_df) for f in valid_features]
    threshold_results = [
        find_best_threshold(
            feature=f,
            pos_df=pos_df,
            neg_df=neg_df,
            objective=args.objective,
            quantile_grid_size=args.quantile_grid_size,
        )
        for f in valid_features
    ]
    best_overall = sorted(threshold_results, key=lambda r: r.score, reverse=True)[0]

    weight_rows, model_metrics = fit_logistic_weights(valid_features, pos_df, neg_df)

    threshold_rows = [r.__dict__ for r in threshold_results]
    report = {
        "inputs": {
            "spike_csv": str(args.spike_csv),
            "normal_csv": str(args.normal_csv),
            "features": valid_features,
            "objective": args.objective,
            "quantile_grid_size": int(args.quantile_grid_size),
        },
        "labeling": label_meta,
        "best_single_feature_rule": best_overall.__dict__,
        "single_feature_thresholds": threshold_rows,
        "feature_stats": feature_stats,
        "standardized_logistic_weights": weight_rows,
        "logistic_metrics": model_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    json_path = args.output_dir / f"{prefix}.json"
    thresholds_csv = args.output_dir / f"{prefix}_thresholds.csv"
    weights_csv = args.output_dir / f"{prefix}_weights.csv"
    stats_csv = args.output_dir / f"{prefix}_feature_stats.csv"
    md_path = args.output_dir / f"{prefix}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, ensure_ascii=False, indent=2)
    pd.DataFrame(threshold_rows).to_csv(thresholds_csv, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_csv, index=False)
    pd.DataFrame(feature_stats).to_csv(stats_csv, index=False)
    write_markdown_report(
        output_path=md_path,
        objective=args.objective,
        label_meta=label_meta,
        best_overall=best_overall,
        threshold_rows=threshold_rows,
        weight_rows=weight_rows,
        model_metrics=model_metrics,
    )

    print("Analysis complete.")
    print(f"Label strategy: {label_meta['label_strategy']}")
    print(
        "Best rule:",
        f"{best_overall.feature} {best_overall.operator} {best_overall.threshold:.6f}",
    )
    print(
        "Best rule metrics:",
        f"recall={best_overall.recall:.4f}",
        f"precision={best_overall.precision:.4f}",
        f"balanced_accuracy={best_overall.balanced_accuracy:.4f}",
        f"score={best_overall.score:.4f}",
    )
    if weight_rows:
        top = weight_rows[0]
        print(
            "Strongest weight:",
            f"{top['feature']} {top['weight']:.4f} ({top['direction']})",
        )
    print(f"Output JSON: {json_path}")
    print(f"Output report: {md_path}")


if __name__ == "__main__":
    main()
