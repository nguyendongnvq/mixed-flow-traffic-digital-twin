from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.evaluation.accident_event_detection import (
    apply_active_state_policy,
    apply_temporal_policy,
    binary_window_metrics,
    build_events,
    calibrate_temporal_policy,
    daily_event_metrics,
    event_metrics,
)
from src.evaluation.accident_ensemble import aggregate_ensemble_probabilities
from src.evaluation.single_label import (
    apply_unknown_temporal_policy,
    calibrate_secondary_thresholds,
    primary_label_metrics,
    select_primary_label,
    select_primary_labels,
)
from src.evaluation.analyze_global_threshold import (
    binary_metrics,
    choose_global_threshold,
)
from src.evaluation.analyze_calibrated_baseline import (
    build_prediction_frame,
    feature_columns,
)
from src.evaluation.analyze_outliers_baseline import choose_fp_constrained_threshold
from src.models.multilabel_pooling import aggregate_node_probabilities
from src.train.train_dmfgcrn_multilabel import (
    ClassWeightedBCEWithLogitsLoss,
    accident_event_metrics,
    build_sample_weights,
    build_loss_sample_weights,
    checkpoint_rank,
    choose_thresholds,
)
from scripts.evaluate_two_stage_ensemble import accident_override_mask
from scripts.evaluate_domain_aware_two_stage import combine_predictions
from src.evaluation.domain_gate import extract_domain_features, train_domain_gate
from src.train.train_dmfgcrn_two_stage import (
    build_group_sampling_weights,
    build_primary_targets,
    coherent_probabilities,
    focal_router_loss,
)


CLASS_NAMES = ["accident", "congestion_spike", "zigzag", "sudden_stop"]


class AccidentFirstTrainingTests(unittest.TestCase):
    def test_two_stage_primary_targets_follow_priority(self) -> None:
        labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            build_primary_targets(labels, CLASS_NAMES),
            [0, 4, 3, 2, 1],
        )

    def test_group_sampling_balances_class_source_and_scale(self) -> None:
        weights = build_group_sampling_weights(
            np.array([0, 0, 1, 1, 1], dtype=np.int64),
            np.array([0, 0, 5, 5, 6], dtype=np.int32),
            np.array([1.0, 1.0, 0.2, 0.2, 0.6], dtype=np.float32),
            {"sampling_weights": {}},
        )
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2:4].sum()))
        self.assertAlmostEqual(float(weights[2:4].sum()), float(weights[4:].sum()))

    def test_coherent_probabilities_sum_to_one(self) -> None:
        probabilities = coherent_probabilities(
            np.array([0.2, 0.8], dtype=np.float32),
            np.array(
                [
                    [0.5, 0.2, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
        np.testing.assert_allclose(probabilities[:, 0], [0.8, 0.2], atol=1e-7)

    def test_router_loss_accepts_anomaly_targets_only(self) -> None:
        loss = focal_router_loss(
            torch.zeros((2, 4)),
            torch.tensor([0, 3]),
            gamma=2.0,
            class_weights=torch.ones(4),
        )
        self.assertTrue(torch.isfinite(loss))

    def test_accident_override_requires_consensus_or_emergency(self) -> None:
        override = accident_override_mask(
            np.array([True, True, True]),
            np.array([0.6, 0.6, 0.8], dtype=np.float32),
            np.array(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                ],
                dtype=np.float32,
            ),
            emergency_threshold=0.75,
        )
        np.testing.assert_array_equal(override, [True, False, True])

    def test_domain_features_do_not_need_source_id(self) -> None:
        x = np.zeros((3, 12, 2, 6), dtype=np.float32)
        x[1, :, :, 0] = 1.0
        features = extract_domain_features(x)
        self.assertEqual(features.shape[0], 3)
        self.assertGreater(features.shape[1], 0)

    def test_domain_gate_learns_from_window_statistics(self) -> None:
        train_x = np.zeros((8, 12, 2, 6), dtype=np.float32)
        train_x[4:, :, :, 0] = -2.0
        val_x = train_x.copy()
        labels = np.array([False, False, False, False, True, True, True, True])
        gate, metrics = train_domain_gate(train_x, labels, val_x, labels)
        predictions = gate.predict_is_mixed(val_x)
        np.testing.assert_array_equal(predictions, labels)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_domain_fallback_marks_conflicting_known_predictions_unknown(self) -> None:
        result = combine_predictions(
            np.array([0, 1, 2]),
            np.array([0, 0, 3]),
            np.array([0.50, 0.49, 0.51]),
            gate_threshold=0.50,
            fallback_margin=0.10,
        )
        np.testing.assert_array_equal(result, [0, 1, -1])

    def test_accident_pooling_uses_weighted_top_k(self) -> None:
        node_probabilities = np.array(
            [
                [
                    [0.9, 0.2, 0.1, 0.3],
                    [0.6, 0.4, 0.2, 0.1],
                    [0.3, 0.1, 0.8, 0.2],
                ]
            ],
            dtype=np.float32,
        )
        result = aggregate_node_probabilities(
            node_probabilities,
            CLASS_NAMES,
            {
                "enabled": True,
                "class_name": "accident",
                "top_k": 2,
                "max_weight": 0.7,
                "top_k_weight": 0.3,
            },
        )
        self.assertAlmostEqual(result[0, 0], 0.7 * 0.9 + 0.3 * 0.75, places=6)
        self.assertAlmostEqual(result[0, 2], 0.8, places=6)

    def test_accident_windows_receive_highest_sampling_weight(self) -> None:
        labels = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        weights = build_sample_weights(
            labels,
            CLASS_NAMES,
            {
                "normal": 1.0,
                "accident": 4.0,
                "congestion_spike": 1.5,
                "zigzag": 1.5,
                "sudden_stop": 1.5,
            },
        )
        np.testing.assert_array_equal(weights, [1.0, 4.0, 1.5, 4.0])

    def test_class_weight_increases_accident_loss(self) -> None:
        logits = torch.zeros((1, 1, 4))
        targets = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        baseline = ClassWeightedBCEWithLogitsLoss(
            torch.ones((1, 1, 4)),
            torch.ones((1, 1, 4)),
        )(logits, targets)
        weighted = ClassWeightedBCEWithLogitsLoss(
            torch.ones((1, 1, 4)),
            torch.tensor([[[3.0, 1.0, 1.0, 1.0]]]),
        )(logits, targets)
        self.assertGreater(float(weighted), float(baseline))

    def test_checkpoint_rank_prefers_valid_secondary_floors(self) -> None:
        valid = pd.DataFrame(
            [
                {"class_name": "accident", "f1": 0.55, "recall": 0.60, "fp_rate": 0.10},
                {"class_name": "congestion_spike", "f1": 0.20, "recall": 0.30, "fp_rate": 0.10},
                {"class_name": "zigzag", "f1": 0.21, "recall": 0.30, "fp_rate": 0.10},
                {"class_name": "sudden_stop", "f1": 0.65, "recall": 0.70, "fp_rate": 0.02},
            ]
        )
        invalid = valid.copy()
        invalid.loc[invalid["class_name"] == "accident", "f1"] = 0.70
        invalid.loc[invalid["class_name"] == "zigzag", "f1"] = 0.10
        floors = {"congestion_spike": 0.20, "zigzag": 0.20, "sudden_stop": 0.65}
        self.assertGreater(
            checkpoint_rank(valid, target_fp_max=0.15, class_f1_floors=floors),
            checkpoint_rank(invalid, target_fp_max=0.15, class_f1_floors=floors),
        )

    def test_threshold_selection_respects_fp_limit(self) -> None:
        labels = np.array([[1], [1], [0], [0]], dtype=np.float32)
        probabilities = np.array([[0.9], [0.7], [0.8], [0.1]], dtype=np.float32)
        threshold = choose_thresholds(labels, probabilities, target_fp_max=0.0)
        self.assertGreater(float(threshold[0]), 0.8)

    def test_event_recall_detects_any_window_in_event(self) -> None:
        labels = np.zeros((5, 4), dtype=np.float32)
        labels[0:2, 0] = 1
        labels[3:5, 0] = 1
        predictions = np.zeros_like(labels, dtype=bool)
        predictions[1, 0] = True
        metrics = accident_event_metrics(
            labels,
            predictions,
            np.array([0, 300, 900, 1200, 1500], dtype=np.float64),
            np.zeros(5, dtype=np.int32),
            CLASS_NAMES,
        )
        self.assertEqual(metrics["accident_event_count"], 2)
        self.assertEqual(metrics["accident_event_detected"], 1)
        self.assertAlmostEqual(metrics["accident_event_recall"], 0.5)

    def test_event_grouping_splits_on_time_gap(self) -> None:
        events = build_events(
            np.array([1, 1, 0, 1], dtype=np.float32),
            np.array([0, 300, 600, 1200], dtype=np.float64),
        )
        self.assertEqual(events, [(0.0, 300.0), (1200.0, 1200.0)])

    def test_temporal_policy_confirms_and_applies_cooldown(self) -> None:
        alerts = apply_temporal_policy(
            np.array([0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            np.array([0, 300, 600, 900], dtype=np.float64),
            {
                "threshold": 0.5,
                "emergency_threshold": 0.95,
                "rolling_windows": 2,
                "confirmation_windows": 2,
                "cooldown_seconds": 900,
                "step_seconds": 300,
            },
        )
        np.testing.assert_array_equal(alerts, [False, True, False, False])

    def test_accident_active_state_uses_hysteresis(self) -> None:
        active = apply_active_state_policy(
            np.array([0.2, 0.7, 0.8, 0.4, 0.3], dtype=np.float32),
            np.array([0, 300, 600, 900, 1200], dtype=np.float64),
            {
                "enter_threshold": 0.65,
                "exit_threshold": 0.45,
                "enter_confirmation_windows": 1,
                "release_confirmation_windows": 2,
                "rolling_windows": 1,
                "step_seconds": 300,
            },
        )
        np.testing.assert_array_equal(active, [False, True, True, True, False])
        metrics = binary_window_metrics(
            np.array([0, 1, 1, 1, 0]),
            active,
        )
        self.assertEqual(metrics["f1"], 1.0)

    def test_unknown_temporal_policy_requires_confirmation(self) -> None:
        predictions = apply_unknown_temporal_policy(
            np.array([0.5, 1.2, 1.3, 0.4], dtype=np.float32),
            np.array([0, 300, 600, 900], dtype=np.float64),
            {
                "threshold": 1.0,
                "rolling_windows": 1,
                "confirmation_windows": 2,
                "step_seconds": 300,
            },
        )
        np.testing.assert_array_equal(predictions, [False, False, True, False])

    def test_event_metrics_enforces_latency_and_counts_false_alerts(self) -> None:
        metrics = event_metrics(
            np.array([1, 1, 1, 0, 0], dtype=np.float32),
            np.array([False, False, False, True, True]),
            np.array([0, 300, 600, 900, 86400], dtype=np.float64),
            max_latency_seconds=600,
        )
        self.assertEqual(metrics["detected_events"], 0)
        self.assertEqual(metrics["false_alerts"], 2)
        self.assertEqual(metrics["false_alerts_per_day"], 1.0)

    def test_calibration_respects_false_alert_budget(self) -> None:
        policy, metrics = calibrate_temporal_policy(
            np.array([0, 0, 1, 1, 0], dtype=np.float32),
            np.array([0.1, 0.2, 0.8, 0.9, 0.1], dtype=np.float32),
            np.array([0, 300, 600, 900, 1200], dtype=np.float64),
            {
                "rolling_windows": 2,
                "confirmation_windows": 2,
                "cooldown_seconds": 1800,
                "step_seconds": 300,
                "max_latency_seconds": 600,
                "max_false_alerts_per_day": 0,
            },
        )
        self.assertLessEqual(metrics["false_alerts_per_day"], 0)
        self.assertGreaterEqual(policy["emergency_threshold"], policy["threshold"])

    def test_source_loss_weights_prioritize_mixed_accident_and_hard_normal(self) -> None:
        labels = np.zeros((5, 4), dtype=np.float32)
        labels[1, 0] = 1
        labels[4, 0] = 1
        weights = build_loss_sample_weights(
            labels,
            np.array([5, 5, 5, 0, 1], dtype=np.int32),
            np.array([0, 300, 600, 900, 1200], dtype=np.float64),
            CLASS_NAMES,
            {
                "mixed_source_id": 5,
                "accident_only_source_id": 1,
                "mixed_accident": 3.0,
                "accident_only": 1.0,
                "mixed_hard_normal": 1.5,
                "hard_normal_radius_seconds": 300,
            },
        )
        np.testing.assert_array_equal(weights, [1.5, 3.0, 1.5, 1.0, 1.0])

    def test_ensemble_aggregation_mean_max_and_vote(self) -> None:
        probabilities = np.array(
            [[0.2, 0.9], [0.4, 0.7], [0.8, 0.3]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            aggregate_ensemble_probabilities(probabilities, "mean"),
            [0.46666667, 0.63333333],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            aggregate_ensemble_probabilities(probabilities, "max"),
            [0.8, 0.9],
        )
        np.testing.assert_allclose(
            aggregate_ensemble_probabilities(
                probabilities,
                "vote",
                vote_thresholds=np.array([0.5, 0.5, 0.5]),
            ),
            [1 / 3, 2 / 3],
        )

    def test_daily_metrics_keep_day_boundaries_separate(self) -> None:
        rows = daily_event_metrics(
            np.array([1, 0, 1, 0], dtype=np.float32),
            np.array([True, True, True, False]),
            np.array([0, 300, 86400, 86700], dtype=np.float64),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["false_alerts"], 1)
        self.assertEqual(rows[1]["false_alerts"], 0)

    def test_single_label_prioritizes_accident(self) -> None:
        primary_label = select_primary_label(
            CLASS_NAMES,
            np.array([0.66, 0.90, 0.80, 0.70], dtype=np.float32),
            np.array([0.65, 0.25, 0.20, 0.50], dtype=np.float32),
            np.array([True, True, True, True]),
            residual=0.0,
            generic_threshold=1.0,
        )
        self.assertEqual(primary_label, "accident")

    def test_single_label_uses_threshold_ratio(self) -> None:
        primary_label = select_primary_label(
            CLASS_NAMES,
            np.array([0.10, 0.40, 0.30, 0.80], dtype=np.float32),
            np.array([0.65, 0.25, 0.20, 0.50], dtype=np.float32),
            np.array([False, True, True, True]),
            residual=0.0,
            generic_threshold=1.0,
        )
        self.assertEqual(primary_label, "sudden_stop")

    def test_single_label_applies_class_weight(self) -> None:
        primary_label = select_primary_label(
            CLASS_NAMES,
            np.array([0.10, 0.40, 0.30, 0.20], dtype=np.float32),
            np.array([0.65, 0.25, 0.20, 0.50], dtype=np.float32),
            np.array([False, True, True, False]),
            residual=0.0,
            generic_threshold=1.0,
            policy={
                "class_weights": {
                    "congestion_spike": 0.5,
                    "zigzag": 2.0,
                }
            },
        )
        self.assertEqual(primary_label, "zigzag")

    def test_single_label_returns_unknown_then_normal(self) -> None:
        probabilities = np.zeros((2, 4), dtype=np.float32)
        predictions = np.zeros((2, 4), dtype=bool)
        primary_labels = select_primary_labels(
            CLASS_NAMES,
            probabilities,
            np.ones(4, dtype=np.float32),
            predictions,
            np.array([1.1, 0.9], dtype=np.float32),
            generic_threshold=1.0,
        )
        np.testing.assert_array_equal(
            primary_labels,
            ["unknown_anomaly", "normal"],
        )

    def test_primary_label_metrics_accepts_any_true_anomaly(self) -> None:
        metrics = primary_label_metrics(
            np.array(
                [
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                ],
                dtype=np.float32,
            ),
            np.array(["accident", "normal", "sudden_stop"], dtype=object),
            CLASS_NAMES,
        )
        self.assertEqual(metrics["primary_label_hit_rate"], 1.0)
        self.assertEqual(
            metrics["primary_label_confusion_matrix"]["zigzag"]["sudden_stop"],
            1,
        )

    def test_secondary_threshold_calibration_preserves_detected_classes(self) -> None:
        labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        probabilities = np.array(
            [
                [0.0, 0.30, 0.25, 0.55],
                [0.0, 0.80, 0.10, 0.10],
                [0.0, 0.10, 0.80, 0.10],
                [0.0, 0.10, 0.10, 0.90],
            ],
            dtype=np.float32,
        )
        thresholds, multipliers, metrics = calibrate_secondary_thresholds(
            labels,
            CLASS_NAMES,
            probabilities,
            np.array([0.5, 0.25, 0.20, 0.50], dtype=np.float32),
            np.zeros(4, dtype=bool),
            [1.0, 1.5],
        )
        self.assertGreater(metrics["f1"], 0.8)
        self.assertEqual(set(multipliers), set(CLASS_NAMES[1:]))
        self.assertEqual(len(thresholds), len(CLASS_NAMES))

    def test_global_threshold_selection_uses_binary_f1(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        scores = np.array([0.1, 0.4, 0.5, 0.9], dtype=np.float32)
        threshold, metrics = choose_global_threshold(labels, scores)
        self.assertAlmostEqual(threshold, 0.5)
        self.assertEqual(metrics["f1"], 1.0)

    def test_binary_metrics_reports_normal_fp_rate(self) -> None:
        metrics = binary_metrics(
            np.array([0, 0, 1, 1], dtype=np.float32),
            np.array([0, 1, 1, 0], dtype=bool),
        )
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["normal_false_positive_rate"], 0.5)

    def test_fp_constrained_threshold_respects_normal_fp_limit(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        scores = np.array([0.10, 0.30, 0.80, 0.20, 0.70, 0.90], dtype=np.float32)
        threshold, metrics = choose_fp_constrained_threshold(
            labels,
            scores,
            target_fp_rate=1.0 / 3.0,
        )
        self.assertAlmostEqual(threshold, 0.70)
        self.assertLessEqual(metrics["normal_false_positive_rate"], 1.0 / 3.0)
        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["false_positive"], 1)

    def test_fp_constrained_threshold_prefers_f1_then_recall(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        scores = np.array([0.10, 0.20, 0.30, 0.90], dtype=np.float32)
        threshold, metrics = choose_fp_constrained_threshold(
            labels,
            scores,
            target_fp_rate=0.0,
        )
        self.assertAlmostEqual(threshold, 0.30)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["normal_false_positive_rate"], 0.0)

    def test_calibrated_baseline_features_include_residual_and_probabilities(self) -> None:
        class_names = ["accident", "congestion_spike"]
        frame = build_prediction_frame(
            {
                "global_prob": np.array([[0.2, 0.7], [0.8, 0.1]], dtype=np.float32),
                "residual": np.array([1.5, 0.4], dtype=np.float32),
                "labels": np.array([[0, 1], [0, 0]], dtype=np.float32),
                "begin": np.array([0.0, 300.0], dtype=np.float32),
                "source": np.array([1, 0], dtype=np.int32),
            },
            class_names,
        )
        self.assertEqual(frame["is_anomaly_true"].tolist(), [1, 0])
        self.assertAlmostEqual(float(frame.loc[0, "max_class_prob"]), 0.7, places=6)
        self.assertAlmostEqual(float(frame.loc[0, "top_class_margin"]), 0.5, places=6)
        self.assertEqual(
            feature_columns(class_names),
            [
                "global_anomaly_score",
                "prob_accident",
                "prob_congestion_spike",
                "max_class_prob",
                "sum_class_prob",
                "top_class_margin",
            ],
        )


if __name__ == "__main__":
    unittest.main()
