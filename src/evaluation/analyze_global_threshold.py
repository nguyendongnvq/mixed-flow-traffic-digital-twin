from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.single_label import select_primary_labels
from src.models.dmfgcrn_multilabel import DMFGCRNMultiLabel
from src.models.multilabel_pooling import aggregate_node_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep one global residual threshold for any-anomaly detection."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    y_true = labels.astype(bool)
    y_pred = predictions.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fp_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normal_false_positive_rate": fp_rate,
    }


def choose_global_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float | int]]:
    candidates = np.unique(scores.astype(float))
    candidates = np.concatenate(
        [
            [np.nextafter(float(candidates.min()), -np.inf)],
            candidates,
            [np.nextafter(float(candidates.max()), np.inf)],
        ]
    )
    rows: list[tuple[tuple[float, float, float], float, dict[str, float | int]]] = []
    for threshold in candidates:
        metrics = binary_metrics(labels, scores >= threshold)
        rank = (
            float(metrics["f1"]),
            float(metrics["recall"]),
            -float(metrics["normal_false_positive_rate"]),
        )
        rows.append((rank, float(threshold), metrics))
    _, threshold, metrics = max(rows, key=lambda item: item[0])
    return threshold, metrics


def class_recall_after_global_detection(
    class_labels: np.ndarray,
    global_predictions: np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for class_idx, class_name in enumerate(class_names):
        truth = class_labels[:, class_idx].astype(bool)
        support = int(truth.sum())
        detected = int(np.sum(truth & global_predictions))
        rows.append(
            {
                "class_name": class_name,
                "support": support,
                "detected_by_global": detected,
                "recall_after_global_detection": detected / support if support else 0.0,
            }
        )
    return rows


def predict_split(
    model: DMFGCRNMultiLabel,
    data: np.lib.npyio.NpzFile,
    split: str,
    class_names: list[str],
    pooling_config: dict[str, Any] | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    x = data[f"{split}_x"]
    target = data[f"{split}_target"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(target).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    node_probabilities = []
    forecasts = []
    with torch.no_grad():
        for x_batch, _ in loader:
            node_logits, forecast = model(x_batch)
            node_probabilities.append(torch.sigmoid(node_logits).numpy())
            forecasts.append(forecast[:, 0, :, :].numpy())
    node_prob = np.concatenate(node_probabilities, axis=0)
    global_prob = aggregate_node_probabilities(node_prob, class_names, pooling_config)
    forecast = np.concatenate(forecasts, axis=0)
    residual = np.abs(forecast - target).mean(axis=2).max(axis=1)
    return {
        "node_prob": node_prob,
        "global_prob": global_prob,
        "residual": residual,
        "labels": data[f"{split}_global_y"],
        "begin": data[f"{split}_begin"],
        "source": data[f"{split}_source"],
    }


def make_prediction_rows(
    split_payload: dict[str, np.ndarray],
    class_names: list[str],
    thresholds: np.ndarray,
    global_threshold: float,
    single_label_policy: dict[str, Any] | None,
) -> pd.DataFrame:
    global_prob = split_payload["global_prob"]
    residual = split_payload["residual"]
    class_predictions = global_prob >= thresholds.reshape(1, -1)
    global_predictions = residual >= global_threshold
    class_predictions = class_predictions & global_predictions.reshape(-1, 1)
    unknown_predictions = global_predictions & ~class_predictions.any(axis=1)
    primary_labels = select_primary_labels(
        class_names,
        global_prob,
        thresholds,
        class_predictions,
        residual,
        generic_threshold=global_threshold,
        unknown_predictions=unknown_predictions,
        policy=single_label_policy,
    )

    rows = []
    for idx in range(len(residual)):
        row: dict[str, object] = {
            "begin": float(split_payload["begin"][idx]),
            "source": int(split_payload["source"][idx]),
            "is_anomaly_true": int(split_payload["labels"][idx].sum() > 0),
            "is_anomaly_pred": bool(global_predictions[idx]),
            "primary_label": str(primary_labels[idx]),
            "global_anomaly_score": float(residual[idx]),
        }
        for class_idx, class_name in enumerate(class_names):
            row[class_name] = int(split_payload["labels"][idx, class_idx])
            row[f"prob_{class_name}"] = float(global_prob[idx, class_idx])
            row[f"pred_{class_name}"] = bool(class_predictions[idx, class_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

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

    pooling_config = payload.get("pooling_config")
    val_payload = predict_split(
        model,
        data,
        "val",
        class_names,
        pooling_config,
        args.batch_size,
    )
    val_any = val_payload["labels"].sum(axis=1) > 0
    threshold, val_metrics = choose_global_threshold(val_any, val_payload["residual"])

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "global_threshold": threshold,
        "selection_metric": "validation_binary_f1",
        "validation": val_metrics,
        "splits": {},
    }
    thresholds = np.asarray(payload["class_thresholds"], dtype=np.float32)
    single_label_policy = payload.get("single_label_policy")
    for split in ("val", "test"):
        split_payload = val_payload if split == "val" else predict_split(
            model,
            data,
            split,
            class_names,
            pooling_config,
            args.batch_size,
        )
        true_any = split_payload["labels"].sum(axis=1) > 0
        pred_any = split_payload["residual"] >= threshold
        split_metrics = binary_metrics(true_any, pred_any)
        split_metrics["class_recall"] = class_recall_after_global_detection(
            split_payload["labels"],
            pred_any,
            class_names,
        )
        summary["splits"][split] = split_metrics
        make_prediction_rows(
            split_payload,
            class_names,
            thresholds,
            threshold,
            single_label_policy,
        ).to_csv(args.output_dir / f"{split}_global_threshold_predictions.csv", index=False)

    (args.output_dir / "global_threshold_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
