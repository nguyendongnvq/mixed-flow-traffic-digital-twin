from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.accident_ensemble import aggregate_ensemble_probabilities
from src.evaluation.accident_event_detection import (
    apply_active_state_policy,
    apply_temporal_policy,
    binary_window_metrics,
    build_events,
    calibrate_active_state_policy,
    calibrate_temporal_policy,
    daily_event_metrics,
    event_metrics,
)
from src.evaluation.single_label import (
    apply_unknown_temporal_policy,
    calibrate_secondary_thresholds,
    calibrate_single_label_policy,
    calibrate_unknown_temporal_policy,
    primary_label_class_metrics,
    primary_label_metrics,
    select_primary_labels,
)
from src.models.dmfgcrn_accident_binary import DMFGCRNAccidentBinary
from src.models.dmfgcrn_multilabel import DMFGCRNMultiLabel
from src.models.multilabel_pooling import aggregate_node_probabilities


DEFAULT_CHECKPOINTS = [
    Path(
        "models/outputs_dmfgcrn_accident_event_ablation/"
        f"source_weighted_hard_normal_seed_{seed}/dmfgcrn_multilabel.pt"
    )
    for seed in (42, 43, 44)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate accident ensemble.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/processed/multilabel/"
            "dmfgcrn_multilabel_windows_with_mixed_14d.npz"
        ),
    )
    parser.add_argument("--checkpoints", type=Path, nargs=3)
    parser.add_argument(
        "--model-type",
        choices=["multilabel", "binary"],
        default="multilabel",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/outputs_dmfgcrn_accident_ensemble"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def predict_accident(
    checkpoint_path: Path,
    x: np.ndarray,
    *,
    batch_size: int,
    model_type: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if model_type == "binary":
        model = DMFGCRNAccidentBinary(
            project_root=Path.cwd(),
            conf_path=Path(payload["dmfgcrn_conf_path"]),
            forecast_checkpoint_path=Path(payload["dmfgcrn_checkpoint_path"]),
            num_nodes=len(payload["nodes"]),
            lag=int(payload["lag"]),
            device=torch.device("cpu"),
        )
        accident_idx = 0
        class_names = ["accident"]
    else:
        class_names = list(payload["class_names"])
        accident_idx = class_names.index("accident")
        model = DMFGCRNMultiLabel(
            project_root=Path.cwd(),
            conf_path=Path(payload["dmfgcrn_conf_path"]),
            forecast_checkpoint_path=Path(payload["dmfgcrn_checkpoint_path"]),
            num_nodes=len(payload["nodes"]),
            lag=int(payload["lag"]),
            num_classes=len(class_names),
            device=torch.device("cpu"),
        )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities = []
    with torch.no_grad():
        for (x_batch,) in loader:
            node_probability = torch.sigmoid(model.classify(x_batch)).numpy()
            global_probability = aggregate_node_probabilities(
                node_probability,
                class_names,
                payload.get("pooling_config"),
            )
            probabilities.append(global_probability[:, accident_idx])
    return np.concatenate(probabilities), payload


def predict_fallback(
    checkpoint_path: Path,
    x: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = list(payload["class_names"])
    model = DMFGCRNMultiLabel(
        project_root=Path.cwd(),
        conf_path=Path(payload["dmfgcrn_conf_path"]),
        forecast_checkpoint_path=Path(payload["dmfgcrn_checkpoint_path"]),
        num_nodes=len(payload["nodes"]),
        lag=int(payload["lag"]),
        num_classes=len(class_names),
        device=torch.device("cpu"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    node_probabilities = []
    forecasts = []
    with torch.no_grad():
        for (x_batch,) in loader:
            node_logits, forecast = model(x_batch)
            node_probabilities.append(torch.sigmoid(node_logits).numpy())
            forecasts.append(forecast[:, 0, :, :].numpy())
    node_probability = np.concatenate(node_probabilities)
    global_probability = aggregate_node_probabilities(
        node_probability,
        class_names,
        payload.get("pooling_config"),
    )
    forecast = np.concatenate(forecasts)
    residual = np.abs(forecast - target).mean(axis=2).max(axis=1)
    return global_probability, residual, payload


def known_status_metrics(
    labels: np.ndarray,
    primary_labels: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    truth = labels.astype(bool).any(axis=1)
    prediction = np.isin(primary_labels, class_names)
    metrics = binary_window_metrics(truth, prediction)
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }


def event_diagnostics(
    labels: np.ndarray,
    alerts: np.ndarray,
    begins: np.ndarray,
    probabilities: np.ndarray,
    *,
    max_latency_seconds: float,
) -> pd.DataFrame:
    events = build_events(labels, begins)
    alert_times = begins[alerts].astype(float)
    rows = []
    for event_index, (start, end) in enumerate(events):
        event_mask = (begins >= start) & (begins <= end)
        candidate_alerts = alert_times[
            (alert_times >= start)
            & (alert_times <= end)
            & (alert_times - start <= max_latency_seconds)
        ]
        detected = len(candidate_alerts) > 0
        rows.append(
            {
                "event_index": event_index,
                "start": start,
                "end": end,
                "detected": detected,
                "alert_time": float(candidate_alerts[0]) if detected else np.nan,
                "latency_seconds": (
                    float(candidate_alerts[0] - start) if detected else np.nan
                ),
                "max_probability": float(probabilities[event_mask].max()),
                "mean_probability": float(probabilities[event_mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    checkpoint_paths = args.checkpoints
    if checkpoint_paths is None:
        if args.model_type == "binary":
            checkpoint_paths = [
                Path(
                    "models/outputs_dmfgcrn_accident_binary/"
                    f"seed_{seed}/dmfgcrn_accident_binary.pt"
                )
                for seed in (42, 43, 44)
            ]
        else:
            checkpoint_paths = DEFAULT_CHECKPOINTS

    data = np.load(args.dataset, allow_pickle=False)
    class_names = data["class_names"].astype(str).tolist()
    accident_idx = class_names.index("accident")
    mixed_source_id = 5
    base_policy = {
        "mixed_source_id": mixed_source_id,
        "rolling_windows_options": [1, 2, 3],
        "confirmation_windows_options": [1, 2],
        "cooldown_seconds_options": [900, 1800],
        "step_seconds": 300,
        "max_latency_seconds": 600,
        "max_false_alerts_per_day": 2.0,
    }

    split_payload: dict[str, dict[str, np.ndarray]] = {}
    model_payloads = []
    for split in ("val", "test"):
        mixed_mask = data[f"{split}_source"] == mixed_source_id
        probabilities = []
        for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
            probability, payload = predict_accident(
                checkpoint_path,
                data[f"{split}_x"][mixed_mask],
                batch_size=args.batch_size,
                model_type=args.model_type,
            )
            probabilities.append(probability)
            if split == "val":
                model_payloads.append(payload)
        split_payload[split] = {
            "probabilities": np.stack(probabilities),
            "labels": data[f"{split}_global_y"][mixed_mask, accident_idx],
            "begins": data[f"{split}_begin"][mixed_mask],
        }

    val = split_payload["val"]
    individual_thresholds = []
    for probability in val["probabilities"]:
        policy, _ = calibrate_temporal_policy(
            val["labels"],
            probability,
            val["begins"],
            base_policy,
        )
        individual_thresholds.append(float(policy["threshold"]))

    candidates = []
    for method in ("mean", "max", "vote"):
        probability = aggregate_ensemble_probabilities(
            val["probabilities"],
            method,
            vote_thresholds=np.asarray(individual_thresholds),
        )
        policy, metrics = calibrate_temporal_policy(
            val["labels"],
            probability,
            val["begins"],
            base_policy,
        )
        if method == "vote":
            policy["model_thresholds"] = individual_thresholds
        feasible = float(metrics["max_daily_false_alerts"]) <= 2.0
        rank = (
            float(feasible),
            float(metrics["event_recall"]),
            float(metrics["event_f1"]),
            float(metrics["event_precision"]),
            -float(metrics["max_daily_false_alerts"]),
        )
        candidates.append(
            {
                "method": method,
                "rank": rank,
                "policy": policy,
                "validation_metrics": metrics,
            }
        )
    selected = max(candidates, key=lambda row: row["rank"])

    test = split_payload["test"]
    test_probability = aggregate_ensemble_probabilities(
        test["probabilities"],
        selected["method"],
        vote_thresholds=np.asarray(individual_thresholds),
    )
    test_alerts = apply_temporal_policy(
        test_probability,
        test["begins"],
        selected["policy"],
    )
    test_metrics = event_metrics(
        test["labels"],
        test_alerts,
        test["begins"],
        max_latency_seconds=600,
    )
    test_daily = daily_event_metrics(
        test["labels"],
        test_alerts,
        test["begins"],
        max_latency_seconds=600,
    )
    threshold = float(selected["policy"]["threshold"])
    window_fp = float(
        (test_probability[test["labels"] == 0] >= threshold).mean()
    )

    fallback_checkpoint = checkpoint_paths[0]
    if args.model_type == "binary":
        fallback_checkpoint = Path(model_payloads[0]["base_checkpoint"])
    fallback_split = {}
    for split in ("val", "test"):
        mixed_mask = data[f"{split}_source"] == mixed_source_id
        global_probability, residual, fallback_payload = predict_fallback(
            fallback_checkpoint,
            data[f"{split}_x"][mixed_mask],
            data[f"{split}_target"][mixed_mask],
            batch_size=args.batch_size,
        )
        fallback_split[split] = {
            "probabilities": global_probability,
            "residual": residual,
            "labels": data[f"{split}_global_y"][mixed_mask],
            "begins": data[f"{split}_begin"][mixed_mask],
        }
    fallback_thresholds = np.asarray(
        fallback_payload["class_thresholds"],
        dtype=np.float32,
    )

    val_probability = aggregate_ensemble_probabilities(
        val["probabilities"],
        selected["method"],
        vote_thresholds=np.asarray(individual_thresholds),
    )
    active_policy, active_validation_metrics = calibrate_active_state_policy(
        val["labels"],
        val_probability,
        val["begins"],
        {
            "step_seconds": 300,
            "rolling_windows_options": [1, 2],
            "enter_confirmation_windows_options": [1, 2],
            "release_confirmation_windows_options": [1, 2, 3],
        },
    )
    val_active = apply_active_state_policy(
        val_probability,
        val["begins"],
        active_policy,
    )
    val_fallback = fallback_split["val"]
    effective_thresholds, threshold_multipliers, threshold_validation_metrics = (
        calibrate_secondary_thresholds(
            val_fallback["labels"],
            class_names,
            val_fallback["probabilities"],
            fallback_thresholds,
            val_active,
        )
    )
    val_predictions = (
        val_fallback["probabilities"]
        >= effective_thresholds.reshape(1, -1)
    )
    val_predictions[:, accident_idx] = val_active
    unknown_policy, unknown_validation_metrics = calibrate_unknown_temporal_policy(
        val_fallback["residual"],
        val_fallback["begins"],
        ~val_fallback["labels"].astype(bool).any(axis=1),
        {
            "step_seconds": 300,
            "target_normal_fp_rate": 0.05,
            "rolling_windows_options": [1, 2, 3],
            "confirmation_windows_options": [1, 2],
            "threshold_percentiles": [90.0, 92.5, 95.0, 97.5, 99.0],
        },
    )
    val_unknown = apply_unknown_temporal_policy(
        val_fallback["residual"],
        val_fallback["begins"],
        unknown_policy,
    )
    single_label_policy, single_label_validation_metrics = (
        calibrate_single_label_policy(
            val_fallback["labels"],
            class_names,
            val_fallback["probabilities"],
            effective_thresholds,
            val_predictions,
            val_fallback["residual"],
            val_unknown,
            {
                "priority_label": "accident",
                "selection": "threshold_ratio",
                "tie_break_order": [
                    "congestion_spike",
                    "zigzag",
                    "sudden_stop",
                ],
                "weight_values": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            },
        )
    )

    test_fallback = fallback_split["test"]
    test_active = apply_active_state_policy(
        test_probability,
        test["begins"],
        active_policy,
    )
    test_predictions = (
        test_fallback["probabilities"]
        >= effective_thresholds.reshape(1, -1)
    )
    test_predictions[:, accident_idx] = test_active
    test_unknown = apply_unknown_temporal_policy(
        test_fallback["residual"],
        test_fallback["begins"],
        unknown_policy,
    )
    test_primary_labels = select_primary_labels(
        class_names,
        test_fallback["probabilities"],
        effective_thresholds,
        test_predictions,
        test_fallback["residual"],
        generic_threshold=float(unknown_policy["threshold"]),
        unknown_predictions=test_unknown,
        policy=single_label_policy,
    )
    active_test_metrics = binary_window_metrics(test["labels"], test_active)
    primary_test_metrics = primary_label_metrics(
        test_fallback["labels"],
        test_primary_labels,
        class_names,
        policy=single_label_policy,
    )
    primary_test_metrics["class_metrics"] = primary_label_class_metrics(
        test_fallback["labels"],
        test_primary_labels,
        class_names,
    )
    primary_test_metrics["known_status"] = known_status_metrics(
        test_fallback["labels"],
        test_primary_labels,
        class_names,
    )
    test_normal_mask = ~test_fallback["labels"].astype(bool).any(axis=1)
    primary_test_metrics["normal_to_unknown_fp_rate"] = float(
        (test_primary_labels[test_normal_mask] == "unknown_anomaly").mean()
    )
    primary_test_metrics["starvation_free"] = all(
        row["support"] == 0 or row["selected_count"] > 0
        for row in primary_test_metrics["class_metrics"].values()
    )
    fallback_metrics = pd.read_csv(
        fallback_checkpoint.parent / "test_metrics_by_class.csv"
    ).set_index("class_name")
    floors_pass = (
        float(fallback_metrics.loc["congestion_spike", "f1"]) >= 0.20
        and float(fallback_metrics.loc["zigzag", "f1"]) >= 0.20
        and float(fallback_metrics.loc["sudden_stop", "f1"]) >= 0.65
    )
    max_daily_false_alerts = max(
        (float(row["false_alerts"]) for row in test_daily),
        default=0.0,
    )
    improvement_targets = {
        "accident_event_f1_min": 0.9375,
        "accident_active_f1_min": 0.90,
        "known_status_f1_min": 0.80,
        "primary_label_hit_rate_min": 0.75,
        "normal_to_unknown_fp_rate_max": 0.05,
    }
    single_label_targets_pass = (
        float(test_metrics["event_f1"])
        >= improvement_targets["accident_event_f1_min"]
        and float(active_test_metrics["f1"])
        >= improvement_targets["accident_active_f1_min"]
        and float(primary_test_metrics["known_status"]["f1"])
        >= improvement_targets["known_status_f1_min"]
        and float(primary_test_metrics["primary_label_hit_rate"])
        >= improvement_targets["primary_label_hit_rate_min"]
        and float(primary_test_metrics["normal_to_unknown_fp_rate"])
        <= improvement_targets["normal_to_unknown_fp_rate_max"]
        and bool(primary_test_metrics["starvation_free"])
    )
    promote = (
        float(test_metrics["event_recall"]) >= 0.85
        and float(test_metrics["p90_latency_seconds"]) <= 600
        and max_daily_false_alerts <= 2.0
        and window_fp <= 0.15
        and floors_pass
        and single_label_targets_pass
    )

    manifest = {
        "model_type": args.model_type,
        "checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
        "fallback_checkpoint": str(fallback_checkpoint.resolve()),
        "aggregation": selected["method"],
        "model_thresholds": individual_thresholds,
        "temporal_policy": selected["policy"],
        "accident_active_policy": active_policy,
        "unknown_temporal_policy": unknown_policy,
        "class_threshold_multipliers": threshold_multipliers,
        "mixed_source_id": mixed_source_id,
        "single_label_policy": single_label_policy,
    }
    (args.output_dir / "ensemble_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "ensemble_summary.json").write_text(
        json.dumps(
            {
                "validation_candidates": [
                    {
                        "method": row["method"],
                        "policy": row["policy"],
                        "metrics": row["validation_metrics"],
                    }
                    for row in candidates
                ],
                "selected_method": selected["method"],
                "test_metrics": test_metrics,
                "accident_active_validation_metrics": active_validation_metrics,
                "accident_active_test_metrics": active_test_metrics,
                "unknown_validation_metrics": unknown_validation_metrics,
                "threshold_validation_metrics": threshold_validation_metrics,
                "single_label_validation_metrics": single_label_validation_metrics,
                "single_label_test_metrics": primary_test_metrics,
                "improvement_targets": improvement_targets,
                "single_label_targets_pass": single_label_targets_pass,
                "test_max_daily_false_alerts": max_daily_false_alerts,
                "test_window_fp": window_fp,
                "secondary_floors_pass": floors_pass,
                "promote": promote,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(test_daily).to_csv(
        args.output_dir / "daily_event_metrics.csv",
        index=False,
    )
    event_diagnostics(
        test["labels"],
        test_alerts,
        test["begins"],
        test_probability,
        max_latency_seconds=600,
    ).to_csv(args.output_dir / "event_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "begin": test["begins"],
            "label_accident": test["labels"],
            "accident_prob_mean": test["probabilities"].mean(axis=0),
            "accident_prob_max": test["probabilities"].max(axis=0),
            "accident_votes": (
                test["probabilities"]
                >= np.asarray(individual_thresholds).reshape(-1, 1)
            ).sum(axis=0),
            "accident_event_alert": test_alerts,
            "accident_active": test_active,
            "primary_label": test_primary_labels,
            "generic_anomaly_score": test_fallback["residual"],
        }
    ).to_csv(args.output_dir / "test_inference.csv", index=False)
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "promote": promote,
                "selected": manifest if promote else None,
                "fallback": None if promote else manifest["fallback_checkpoint"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"promote": promote, **test_metrics}, indent=2))


if __name__ == "__main__":
    main()
