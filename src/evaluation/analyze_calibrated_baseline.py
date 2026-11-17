from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.analyze_global_threshold import (
    binary_metrics,
    predict_split,
)
from src.evaluation.analyze_outliers_baseline import (
    DEFAULT_FP_TARGETS,
    choose_fp_constrained_threshold,
    class_recall_rows,
    markdown_table,
    metrics_at_threshold,
    parse_fp_targets,
)
from src.evaluation.single_label import select_primary_labels
from src.models.dmfgcrn_multilabel import DMFGCRNMultiLabel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a calibrated any-anomaly baseline from residual and class probabilities."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--residual-pred-dir", type=Path)
    parser.add_argument("--target-fp-rate", type=float, default=0.25)
    parser.add_argument(
        "--fp-targets",
        default=",".join(str(value) for value in DEFAULT_FP_TARGETS),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def feature_columns(class_names: list[str]) -> list[str]:
    return (
        ["global_anomaly_score"]
        + [f"prob_{class_name}" for class_name in class_names]
        + ["max_class_prob", "sum_class_prob", "top_class_margin"]
    )


def build_prediction_frame(
    split_payload: dict[str, np.ndarray],
    class_names: list[str],
) -> pd.DataFrame:
    global_prob = split_payload["global_prob"]
    residual = split_payload["residual"]
    labels = split_payload["labels"]
    sorted_prob = np.sort(global_prob, axis=1)
    top_prob = sorted_prob[:, -1]
    second_prob = sorted_prob[:, -2] if global_prob.shape[1] > 1 else np.zeros(len(global_prob))
    rows: list[dict[str, Any]] = []
    for idx in range(len(residual)):
        row: dict[str, Any] = {
            "begin": float(split_payload["begin"][idx]),
            "source": int(split_payload["source"][idx]),
            "is_anomaly_true": int(labels[idx].sum() > 0),
            "global_anomaly_score": float(residual[idx]),
            "max_class_prob": float(top_prob[idx]),
            "sum_class_prob": float(global_prob[idx].sum()),
            "top_class_margin": float(top_prob[idx] - second_prob[idx]),
        }
        for class_idx, class_name in enumerate(class_names):
            row[class_name] = int(labels[idx, class_idx])
            row[f"prob_{class_name}"] = float(global_prob[idx, class_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def train_calibrator(
    train_df: pd.DataFrame,
    columns: list[str],
    random_state: int,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(
        train_df[columns].to_numpy(dtype=float),
        train_df["is_anomaly_true"].to_numpy(dtype=int),
    )
    return model


def add_calibrated_scores(
    df: pd.DataFrame,
    model: Pipeline,
    columns: list[str],
) -> pd.DataFrame:
    output = df.copy()
    output["calibrated_anomaly_score"] = model.predict_proba(
        output[columns].to_numpy(dtype=float)
    )[:, 1]
    return output


def calibrated_metrics_at_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int]:
    labels = df["is_anomaly_true"].to_numpy(dtype=bool)
    predictions = df["calibrated_anomaly_score"].to_numpy(dtype=float) >= threshold
    return binary_metrics(labels, predictions)


def build_calibrated_sweep(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fp_targets: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = val_df["is_anomaly_true"].to_numpy(dtype=bool)
    scores = val_df["calibrated_anomaly_score"].to_numpy(dtype=float)
    for target in fp_targets:
        threshold, val_metrics = choose_fp_constrained_threshold(
            labels,
            scores,
            target,
        )
        test_metrics = calibrated_metrics_at_threshold(test_df, threshold)
        row: dict[str, Any] = {
            "target_fp_rate": target,
            "threshold": threshold,
        }
        for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def apply_calibrated_predictions(
    df: pd.DataFrame,
    class_names: list[str],
    class_thresholds: np.ndarray,
    threshold: float,
    single_label_policy: dict[str, Any] | None,
) -> pd.DataFrame:
    output = df.copy()
    probabilities = output[[f"prob_{class_name}" for class_name in class_names]].to_numpy(
        dtype=float
    )
    global_predictions = output["calibrated_anomaly_score"].to_numpy(dtype=float) >= threshold
    class_predictions = probabilities >= class_thresholds.reshape(1, -1)
    class_predictions = class_predictions & global_predictions.reshape(-1, 1)
    unknown_predictions = global_predictions & ~class_predictions.any(axis=1)
    output["is_anomaly_pred"] = global_predictions
    output["primary_label"] = select_primary_labels(
        class_names,
        probabilities,
        class_thresholds,
        class_predictions,
        output["global_anomaly_score"].to_numpy(dtype=float),
        generic_threshold=float("inf"),
        unknown_predictions=unknown_predictions,
        policy=single_label_policy,
    )
    output["status"] = np.where(
        ~global_predictions,
        "normal",
        np.where(output["primary_label"].astype(str) == "unknown_anomaly", "unknown_anomaly", "known_anomaly"),
    )
    for class_idx, class_name in enumerate(class_names):
        output[f"pred_{class_name}"] = class_predictions[:, class_idx]
    return output


def coefficient_rows(
    model: Pipeline,
    columns: list[str],
) -> list[dict[str, float | str]]:
    classifier = model.named_steps["logreg"]
    coefficients = classifier.coef_[0]
    return [
        {"feature": column, "coefficient": float(coefficient)}
        for column, coefficient in sorted(
            zip(columns, coefficients),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
    ]


def load_residual_reference(residual_pred_dir: Path | None) -> dict[str, Any] | None:
    if residual_pred_dir is None:
        return None
    baseline_summary = residual_pred_dir.parent / "outlier_baseline" / "baseline_summary.json"
    global_summary = residual_pred_dir / "global_threshold_summary.json"
    reference: dict[str, Any] = {}
    if baseline_summary.exists():
        payload = json.loads(baseline_summary.read_text(encoding="utf-8"))
        reference["residual_fp_constrained"] = payload.get("selected_baseline")
    if global_summary.exists():
        payload = json.loads(global_summary.read_text(encoding="utf-8"))
        reference["residual_max_f1"] = {
            "threshold": float(payload["global_threshold"]),
            "validation": payload.get("validation"),
            "test": payload.get("splits", {}).get("test"),
        }
    return reference or None


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    selected = summary["selected_calibrated_baseline"]
    lines = [
        "# Calibrated Baseline Report",
        "",
        "## Selected Calibrated Baseline",
        "",
        f"- Target validation normal FP rate: {selected['target_fp_rate']:.3f}",
        f"- Threshold: {selected['threshold']:.6f}",
        f"- Test precision: {selected['test']['precision']:.4f}",
        f"- Test recall: {selected['test']['recall']:.4f}",
        f"- Test F1: {selected['test']['f1']:.4f}",
        f"- Test normal FP rate: {selected['test']['normal_false_positive_rate']:.4f}",
        "",
    ]
    references = summary.get("references") or {}
    residual = references.get("residual_fp_constrained")
    if residual is not None:
        lines.extend(
            [
                "## Residual FP-Constrained Reference",
                "",
                f"- Threshold: {residual['threshold']:.6f}",
                f"- Test precision: {residual['test']['precision']:.4f}",
                f"- Test recall: {residual['test']['recall']:.4f}",
                f"- Test F1: {residual['test']['f1']:.4f}",
                f"- Test normal FP rate: {residual['test']['normal_false_positive_rate']:.4f}",
                "",
            ]
        )
    max_f1 = references.get("residual_max_f1")
    if max_f1 is not None:
        lines.extend(
            [
                "## Residual Max-F1 Reference",
                "",
                f"- Threshold: {max_f1['threshold']:.6f}",
                f"- Test precision: {max_f1['test']['precision']:.4f}",
                f"- Test recall: {max_f1['test']['recall']:.4f}",
                f"- Test F1: {max_f1['test']['f1']:.4f}",
                f"- Test normal FP rate: {max_f1['test']['normal_false_positive_rate']:.4f}",
                "",
            ]
        )
    lines.extend(["## Top Coefficients", ""])
    coefficient_df = pd.DataFrame(summary["feature_coefficients"]).head(10)
    lines.extend(markdown_table(coefficient_df))
    lines.append("")
    lines.append(
        "The calibrated detector decides only any-anomaly; primary labels remain auxiliary classification."
    )
    (output_dir / "calibrated_baseline_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    fp_targets = parse_fp_targets(args.fp_targets)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = np.load(args.dataset, allow_pickle=False)
    class_names = list(payload["class_names"])
    nodes = list(payload["nodes"])
    model = DMFGCRNMultiLabel(
        project_root=Path.cwd(),
        conf_path=Path(payload["dmfgcrn_conf_path"]),
        forecast_checkpoint_path=Path(payload["dmfgcrn_checkpoint_path"]),
        num_nodes=len(nodes),
        lag=int(payload["lag"]),
        num_classes=len(class_names),
        device=torch.device("cpu"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    frames: dict[str, pd.DataFrame] = {}
    pooling_config = payload.get("pooling_config")
    for split in ("train", "val", "test"):
        split_payload = predict_split(
            model,
            data,
            split,
            class_names,
            pooling_config,
            args.batch_size,
        )
        frames[split] = build_prediction_frame(split_payload, class_names)

    columns = feature_columns(class_names)
    calibrator = train_calibrator(frames["train"], columns, args.random_state)
    for split, frame in list(frames.items()):
        scored = add_calibrated_scores(frame, calibrator, columns)
        frames[split] = scored
        scored.to_csv(args.output_dir / f"{split}_prediction_cache.csv", index=False)

    sweep = build_calibrated_sweep(frames["val"], frames["test"], fp_targets)
    sweep.to_csv(args.output_dir / "calibrated_threshold_sweep.csv", index=False)

    labels = frames["val"]["is_anomaly_true"].to_numpy(dtype=bool)
    scores = frames["val"]["calibrated_anomaly_score"].to_numpy(dtype=float)
    threshold, val_metrics = choose_fp_constrained_threshold(
        labels,
        scores,
        args.target_fp_rate,
    )
    test_metrics = calibrated_metrics_at_threshold(frames["test"], threshold)
    class_thresholds = np.asarray(payload["class_thresholds"], dtype=float)
    test_predictions = apply_calibrated_predictions(
        frames["test"],
        class_names,
        class_thresholds,
        threshold,
        payload.get("single_label_policy"),
    )
    test_predictions.to_csv(
        args.output_dir / "calibrated_test_predictions.csv",
        index=False,
    )

    class_recall_frame = test_predictions.copy()
    class_recall_frame["global_anomaly_score"] = class_recall_frame[
        "calibrated_anomaly_score"
    ]
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "target_fp_rate": float(args.target_fp_rate),
        "feature_columns": columns,
        "selected_calibrated_baseline": {
            "target_fp_rate": float(args.target_fp_rate),
            "threshold": float(threshold),
            "validation": val_metrics,
            "test": test_metrics,
            "test_class_recall": class_recall_rows(class_recall_frame, threshold),
            "test_status_counts": test_predictions["status"].value_counts().to_dict(),
        },
        "references": load_residual_reference(args.residual_pred_dir),
        "feature_coefficients": coefficient_rows(calibrator, columns),
        "outputs": {
            "train_prediction_cache": "train_prediction_cache.csv",
            "val_prediction_cache": "val_prediction_cache.csv",
            "test_prediction_cache": "test_prediction_cache.csv",
            "calibrated_threshold_sweep": "calibrated_threshold_sweep.csv",
            "calibrated_test_predictions": "calibrated_test_predictions.csv",
            "report": "calibrated_baseline_report.md",
        },
    }
    (args.output_dir / "calibrated_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
