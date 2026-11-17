from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from src.configs.config import load_config
from src.evaluation.event_eval import evaluate_event_level, infer_step_seconds, summarize_metrics
from src.evaluation.scoring import (
    apply_alert_logic,
    compute_weighted_node_max_scores,
    predict_reconstruction,
    select_threshold,
)
from src.models.model import GRUAutoencoder, LSTMAutoencoder
from src.preprocessing.data import (
    df_to_tensor,
    ensure_same_nodes,
    load_dataframes,
    make_time_labels,
    to_feature_weights_array,
)
from src.preprocessing.preprocess import (
    fit_scaler,
    flatten_node_feature,
    make_loader,
    make_windows,
    make_windows_with_labels,
    scale_tensor,
    split_normal_data,
)
from src.train.trainer import train_model
from src.utils.io_utils import ensure_output_dir, save_model_artifact, save_summary_files
from src.utils.plotting import (
    plot_delay_charts,
    plot_score_distribution,
    plot_score_timeline,
    plot_summary_rates,
    plot_top_alert_nodes,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LSTM/GRU autoencoders, Isolation Forest, and One-Class SVM on accident data.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lstm_autoencoder_accident.json",
        help="Base config JSON (dataset paths, features, thresholds, and training settings).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="models/benchmark_accident_models",
        help="Output root folder for benchmark artifacts.",
    )
    parser.add_argument(
        "--iforest-n-estimators",
        type=int,
        default=300,
        help="Number of trees for Isolation Forest.",
    )
    parser.add_argument(
        "--iforest-max-samples",
        type=str,
        default="auto",
        help="max_samples for Isolation Forest (int or 'auto').",
    )
    parser.add_argument(
        "--iforest-n-jobs",
        type=int,
        default=1,
        help="n_jobs for Isolation Forest. Keep 1 on restricted Windows environments.",
    )
    parser.add_argument(
        "--ocsvm-kernel",
        type=str,
        default="rbf",
        help="Kernel for One-Class SVM.",
    )
    parser.add_argument(
        "--ocsvm-gamma",
        type=str,
        default="scale",
        help="Gamma for One-Class SVM.",
    )
    parser.add_argument(
        "--ocsvm-nu",
        type=float,
        default=0.05,
        help="nu for One-Class SVM.",
    )
    parser.add_argument(
        "--ocsvm-max-train",
        type=int,
        default=3000,
        help="Maximum train windows used for One-Class SVM fit (subsample if exceeded).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_iforest_max_samples(raw_value: str) -> str | int:
    if raw_value == "auto":
        return "auto"
    return int(raw_value)


def _compute_top_node_ids(
    windows: np.ndarray,
    baseline_mean: np.ndarray,
    n_nodes: int,
    n_features: int,
) -> list[str]:
    windows_4d = windows.reshape(windows.shape[0], windows.shape[1], n_nodes, n_features)
    baseline = baseline_mean.reshape(1, 1, n_nodes, n_features)
    node_deviation = np.abs(windows_4d - baseline).mean(axis=(1, 3))
    top_node_idx = node_deviation.argmax(axis=1)
    return top_node_idx.tolist()


def _evaluate_model_scores(
    *,
    model_name: str,
    output_dir: Path,
    threshold_percentile: float,
    min_consecutive: int,
    cooldown_steps: int,
    seq_size: int,
    test_normal_window_end_begin: np.ndarray,
    accident_timesteps: list[float],
    accident_window_labels: np.ndarray,
    val_scores: np.ndarray,
    test_normal_scores: np.ndarray,
    accident_scores: np.ndarray,
    top_node_ids: list[str],
    step_seconds: float,
    event_gap_seconds: float,
) -> dict[str, Any]:
    selected_threshold = select_threshold(val_scores, threshold_percentile)
    normal_pred_window = test_normal_scores > selected_threshold
    accident_pred_window = accident_scores > selected_threshold

    normal_alerts = apply_alert_logic(
        test_normal_scores,
        selected_threshold,
        min_consecutive=min_consecutive,
        cooldown_steps=cooldown_steps,
    )
    accident_alerts = apply_alert_logic(
        accident_scores,
        selected_threshold,
        min_consecutive=min_consecutive,
        cooldown_steps=cooldown_steps,
    )

    accident_window_end_begin = np.array(accident_timesteps[seq_size - 1 :])
    result_df = pd.DataFrame(
        {
            "window_idx": np.arange(len(accident_scores)),
            "begin": accident_window_end_begin,
            "anomaly_score": accident_scores,
            "threshold": selected_threshold,
            "pred_window": accident_pred_window,
            "alert": accident_alerts,
            "true_window_label": accident_window_labels,
            "top_node_id": top_node_ids,
        }
    )
    event_eval_alert = evaluate_event_level(
        result_df,
        alert_col="alert",
        gap_threshold=event_gap_seconds,
    )
    summary, summary_df = summarize_metrics(
        score_name=f"{model_name}_score",
        threshold_percentile=threshold_percentile,
        selected_threshold=selected_threshold,
        min_consecutive=min_consecutive,
        cooldown_steps=cooldown_steps,
        step_seconds=step_seconds,
        normal_pred_window=normal_pred_window,
        accident_pred_window=accident_pred_window,
        normal_alerts=normal_alerts,
        test_normal_window_end_begin=test_normal_window_end_begin,
        event_eval_alert=event_eval_alert,
    )

    save_summary_files(output_dir, summary, summary_df, result_df, event_eval_alert)

    plot_score_distribution(
        test_normal_weighted=test_normal_scores,
        accident_weighted=accident_scores,
        threshold=selected_threshold,
        threshold_percentile=threshold_percentile,
        output_dir=output_dir,
    )
    plot_score_timeline(
        scores=test_normal_scores,
        threshold=selected_threshold,
        alerts=normal_alerts,
        title=f"{model_name}: normal test score timeline",
        output_name="normal_score_timeline.png",
        output_dir=output_dir,
    )
    plot_score_timeline(
        scores=accident_scores,
        threshold=selected_threshold,
        alerts=accident_alerts,
        title=f"{model_name}: accident score timeline",
        output_name="accident_score_timeline.png",
        output_dir=output_dir,
    )
    plot_summary_rates(
        window_detect_ratio=summary["window_detect_ratio"],
        event_recall=summary["event_recall"],
        normal_fp_rate=summary["normal_fp_rate"],
        output_dir=output_dir,
    )
    plot_delay_charts(
        mean_delay_seconds=summary["mean_delay_seconds"],
        median_delay_seconds=summary["median_delay_seconds"],
        event_eval_alert=event_eval_alert,
        output_dir=output_dir,
    )
    top_nodes_df = plot_top_alert_nodes(result_df, output_dir)
    top_nodes_df.to_csv(output_dir / "top_alert_nodes.csv", index=False)

    summary["model_name"] = model_name
    summary["selected_threshold"] = selected_threshold
    summary["output_dir"] = str(output_dir)
    return summary


def _run_autoencoder_model(
    *,
    model_name: str,
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    test_normal_loader: Any,
    accident_loader: Any,
    config: dict[str, Any],
    device: torch.device,
    feature_weights: np.ndarray,
    n_nodes: int,
    n_features: int,
    seq_size: int,
    test_normal_window_end_begin: np.ndarray,
    accident_timesteps: list[float],
    accident_window_labels: np.ndarray,
    step_seconds: float,
    event_gap_seconds: float,
    idx_to_node: dict[int, str],
    output_dir: Path,
) -> dict[str, Any]:
    model, history, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        patience=config["patience"],
        device=device,
    )
    plot_training_history(history, output_dir)

    val_batch_x, val_recon = predict_reconstruction(model, val_loader, device)
    test_normal_batch_x, test_normal_recon = predict_reconstruction(model, test_normal_loader, device)
    accident_batch_x, accident_recon = predict_reconstruction(model, accident_loader, device)

    val_scores, _ = compute_weighted_node_max_scores(
        val_batch_x,
        val_recon,
        n_nodes,
        n_features,
        seq_size,
        feature_weights,
    )
    test_normal_scores, _ = compute_weighted_node_max_scores(
        test_normal_batch_x,
        test_normal_recon,
        n_nodes,
        n_features,
        seq_size,
        feature_weights,
    )
    accident_scores, accident_node_scores = compute_weighted_node_max_scores(
        accident_batch_x,
        accident_recon,
        n_nodes,
        n_features,
        seq_size,
        feature_weights,
    )

    top_node_ids = [idx_to_node[i] for i in accident_node_scores.argmax(axis=1)]
    summary = _evaluate_model_scores(
        model_name=model_name,
        output_dir=output_dir,
        threshold_percentile=config["threshold_percentile"],
        min_consecutive=config["min_consecutive"],
        cooldown_steps=config["cooldown_steps"],
        seq_size=seq_size,
        test_normal_window_end_begin=test_normal_window_end_begin,
        accident_timesteps=accident_timesteps,
        accident_window_labels=accident_window_labels,
        val_scores=val_scores,
        test_normal_scores=test_normal_scores,
        accident_scores=accident_scores,
        top_node_ids=top_node_ids,
        step_seconds=step_seconds,
        event_gap_seconds=event_gap_seconds,
    )

    save_model_artifact(
        output_dir=output_dir,
        model_output_name=f"{model_name}.pt",
        payload={
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "best_epoch": best_state["epoch"],
            "best_val_loss": best_state["val_loss"],
            "input_dim": val_batch_x.shape[2],
            "seq_size": seq_size,
            "n_nodes": n_nodes,
            "n_features": n_features,
            "summary": summary,
        },
    )
    return summary


def _run_iforest(
    *,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    test_normal_windows: np.ndarray,
    accident_windows: np.ndarray,
    top_node_ids: list[str],
    config: dict[str, Any],
    args: argparse.Namespace,
    step_seconds: float,
    event_gap_seconds: float,
    test_normal_window_end_begin: np.ndarray,
    accident_timesteps: list[float],
    accident_window_labels: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    train_2d = train_windows.reshape(train_windows.shape[0], -1)
    val_2d = val_windows.reshape(val_windows.shape[0], -1)
    test_normal_2d = test_normal_windows.reshape(test_normal_windows.shape[0], -1)
    accident_2d = accident_windows.reshape(accident_windows.shape[0], -1)

    model = IsolationForest(
        n_estimators=args.iforest_n_estimators,
        max_samples=_parse_iforest_max_samples(args.iforest_max_samples),
        random_state=config["seed"],
        n_jobs=args.iforest_n_jobs,
    )
    model.fit(train_2d)

    val_scores = -model.decision_function(val_2d)
    test_normal_scores = -model.decision_function(test_normal_2d)
    accident_scores = -model.decision_function(accident_2d)

    summary = _evaluate_model_scores(
        model_name="isolation_forest",
        output_dir=output_dir,
        threshold_percentile=config["threshold_percentile"],
        min_consecutive=config["min_consecutive"],
        cooldown_steps=config["cooldown_steps"],
        seq_size=config["seq_size"],
        test_normal_window_end_begin=test_normal_window_end_begin,
        accident_timesteps=accident_timesteps,
        accident_window_labels=accident_window_labels,
        val_scores=val_scores,
        test_normal_scores=test_normal_scores,
        accident_scores=accident_scores,
        top_node_ids=top_node_ids,
        step_seconds=step_seconds,
        event_gap_seconds=event_gap_seconds,
    )

    save_model_artifact(
        output_dir=output_dir,
        model_output_name="isolation_forest.pt",
        payload={
            "model_name": "isolation_forest",
            "model": model,
            "n_estimators": args.iforest_n_estimators,
            "max_samples": args.iforest_max_samples,
            "n_jobs": args.iforest_n_jobs,
            "summary": summary,
        },
    )
    return summary


def _run_ocsvm(
    *,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    test_normal_windows: np.ndarray,
    accident_windows: np.ndarray,
    top_node_ids: list[str],
    config: dict[str, Any],
    args: argparse.Namespace,
    step_seconds: float,
    event_gap_seconds: float,
    test_normal_window_end_begin: np.ndarray,
    accident_timesteps: list[float],
    accident_window_labels: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(config["seed"])

    train_2d = train_windows.reshape(train_windows.shape[0], -1)
    if train_2d.shape[0] > args.ocsvm_max_train:
        selected = rng.choice(train_2d.shape[0], size=args.ocsvm_max_train, replace=False)
        train_2d = train_2d[selected]
    val_2d = val_windows.reshape(val_windows.shape[0], -1)
    test_normal_2d = test_normal_windows.reshape(test_normal_windows.shape[0], -1)
    accident_2d = accident_windows.reshape(accident_windows.shape[0], -1)

    model = OneClassSVM(
        kernel=args.ocsvm_kernel,
        gamma=args.ocsvm_gamma,
        nu=args.ocsvm_nu,
    )
    model.fit(train_2d)

    val_scores = -model.decision_function(val_2d)
    test_normal_scores = -model.decision_function(test_normal_2d)
    accident_scores = -model.decision_function(accident_2d)

    summary = _evaluate_model_scores(
        model_name="one_class_svm",
        output_dir=output_dir,
        threshold_percentile=config["threshold_percentile"],
        min_consecutive=config["min_consecutive"],
        cooldown_steps=config["cooldown_steps"],
        seq_size=config["seq_size"],
        test_normal_window_end_begin=test_normal_window_end_begin,
        accident_timesteps=accident_timesteps,
        accident_window_labels=accident_window_labels,
        val_scores=val_scores,
        test_normal_scores=test_normal_scores,
        accident_scores=accident_scores,
        top_node_ids=top_node_ids,
        step_seconds=step_seconds,
        event_gap_seconds=event_gap_seconds,
    )

    save_model_artifact(
        output_dir=output_dir,
        model_output_name="one_class_svm.pt",
        payload={
            "model_name": "one_class_svm",
            "model": model,
            "kernel": args.ocsvm_kernel,
            "gamma": args.ocsvm_gamma,
            "nu": args.ocsvm_nu,
            "ocsvm_max_train": args.ocsvm_max_train,
            "summary": summary,
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ensure_output_dir(Path(args.output_root) / f"accident_benchmark_{run_id}")

    print("Device:", device)
    print("Benchmark output root:", output_root)

    time_col = config["time_col"]
    node_col = config["node_col"]
    label_col = config["label_col"]
    feature_cols = config["feature_cols"]
    feature_weights = to_feature_weights_array(feature_cols, config["feature_weights"])

    df_normal_clean, df_accident_clean = load_dataframes(
        config["normal_dataset"],
        config["accident_dataset"],
        label_col=label_col,
    )
    nodes = ensure_same_nodes(df_normal_clean, df_accident_clean, node_col)
    idx_to_node = {i: node for i, node in enumerate(nodes)}

    x_normal, normal_timesteps = df_to_tensor(
        df_normal_clean,
        feature_cols,
        nodes,
        time_col=time_col,
        node_col=node_col,
    )
    x_accident, accident_timesteps = df_to_tensor(
        df_accident_clean,
        feature_cols,
        nodes,
        time_col=time_col,
        node_col=node_col,
    )
    accident_time_labels = make_time_labels(
        df_accident_clean,
        accident_timesteps,
        time_col=time_col,
        label_col=label_col,
    )

    split = split_normal_data(
        x_normal,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
    )
    scaler = fit_scaler(split.x_train)
    x_train_scaled = scale_tensor(split.x_train, scaler)
    x_val_scaled = scale_tensor(split.x_val, scaler)
    x_test_normal_scaled = scale_tensor(split.x_test_normal, scaler)
    x_accident_scaled = scale_tensor(x_accident, scaler)

    _, n_nodes, n_features = split.x_train.shape
    seq_size = config["seq_size"]
    x_train_flat = flatten_node_feature(x_train_scaled)
    x_val_flat = flatten_node_feature(x_val_scaled)
    x_test_normal_flat = flatten_node_feature(x_test_normal_scaled)
    x_accident_flat = flatten_node_feature(x_accident_scaled)

    train_x = make_windows(x_train_flat, seq_size)
    val_x = make_windows(x_val_flat, seq_size)
    test_normal_x = make_windows(x_test_normal_flat, seq_size)
    accident_x, accident_window_labels = make_windows_with_labels(
        x_accident_flat,
        accident_time_labels,
        seq_size,
    )

    input_dim = train_x.shape[2]
    train_loader = make_loader(train_x, config["batch_size"], shuffle=True)
    val_loader = make_loader(val_x, config["batch_size"], shuffle=False)
    test_normal_loader = make_loader(test_normal_x, config["batch_size"], shuffle=False)
    accident_loader = make_loader(accident_x, config["batch_size"], shuffle=False)

    step_seconds = infer_step_seconds(normal_timesteps)
    raw_event_gap_seconds = config.get("event_gap_seconds")
    if raw_event_gap_seconds is not None:
        event_gap_seconds = float(raw_event_gap_seconds)
    else:
        event_gap_multiplier = float(config.get("event_gap_multiplier", 1.0))
        event_gap_seconds = float(step_seconds * event_gap_multiplier) if not np.isnan(step_seconds) else np.nan

    test_normal_timesteps = normal_timesteps[split.val_end :]
    test_normal_window_end_begin = np.array(test_normal_timesteps[seq_size - 1 :])

    train_reference_mean = x_train_flat.reshape(x_train_flat.shape[0], n_nodes, n_features).mean(axis=0)
    baseline_top_idx = _compute_top_node_ids(accident_x, train_reference_mean, n_nodes, n_features)
    baseline_top_nodes = [idx_to_node[i] for i in baseline_top_idx]

    summaries = []

    print("Running LSTM autoencoder...")
    lstm_dir = ensure_output_dir(output_root / "lstm_autoencoder")
    lstm_model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    summaries.append(
        _run_autoencoder_model(
            model_name="lstm_autoencoder",
            model=lstm_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_normal_loader=test_normal_loader,
            accident_loader=accident_loader,
            config=config,
            device=device,
            feature_weights=feature_weights,
            n_nodes=n_nodes,
            n_features=n_features,
            seq_size=seq_size,
            test_normal_window_end_begin=test_normal_window_end_begin,
            accident_timesteps=accident_timesteps,
            accident_window_labels=accident_window_labels,
            step_seconds=step_seconds,
            event_gap_seconds=event_gap_seconds,
            idx_to_node=idx_to_node,
            output_dir=lstm_dir,
        )
    )

    print("Running GRU autoencoder...")
    gru_dir = ensure_output_dir(output_root / "gru_autoencoder")
    gru_model = GRUAutoencoder(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    summaries.append(
        _run_autoencoder_model(
            model_name="gru_autoencoder",
            model=gru_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_normal_loader=test_normal_loader,
            accident_loader=accident_loader,
            config=config,
            device=device,
            feature_weights=feature_weights,
            n_nodes=n_nodes,
            n_features=n_features,
            seq_size=seq_size,
            test_normal_window_end_begin=test_normal_window_end_begin,
            accident_timesteps=accident_timesteps,
            accident_window_labels=accident_window_labels,
            step_seconds=step_seconds,
            event_gap_seconds=event_gap_seconds,
            idx_to_node=idx_to_node,
            output_dir=gru_dir,
        )
    )

    print("Running Isolation Forest...")
    iforest_dir = ensure_output_dir(output_root / "isolation_forest")
    summaries.append(
        _run_iforest(
            train_windows=train_x,
            val_windows=val_x,
            test_normal_windows=test_normal_x,
            accident_windows=accident_x,
            top_node_ids=baseline_top_nodes,
            config=config,
            args=args,
            step_seconds=step_seconds,
            event_gap_seconds=event_gap_seconds,
            test_normal_window_end_begin=test_normal_window_end_begin,
            accident_timesteps=accident_timesteps,
            accident_window_labels=accident_window_labels,
            output_dir=iforest_dir,
        )
    )

    print("Running One-Class SVM...")
    ocsvm_dir = ensure_output_dir(output_root / "one_class_svm")
    summaries.append(
        _run_ocsvm(
            train_windows=train_x,
            val_windows=val_x,
            test_normal_windows=test_normal_x,
            accident_windows=accident_x,
            top_node_ids=baseline_top_nodes,
            config=config,
            args=args,
            step_seconds=step_seconds,
            event_gap_seconds=event_gap_seconds,
            test_normal_window_end_begin=test_normal_window_end_begin,
            accident_timesteps=accident_timesteps,
            accident_window_labels=accident_window_labels,
            output_dir=ocsvm_dir,
        )
    )

    benchmark_df = pd.DataFrame(summaries)
    benchmark_df = benchmark_df[
        [
            "model_name",
            "threshold_percentile",
            "selected_threshold",
            "normal_fp_rate",
            "event_recall",
            "window_detect_ratio",
            "mean_delay_seconds",
            "median_delay_seconds",
            "normal_alert_count",
            "false_alerts_per_hour",
            "output_dir",
        ]
    ]
    benchmark_csv = output_root / "benchmark_summary.csv"
    benchmark_json = output_root / "benchmark_summary.json"
    benchmark_df.to_csv(benchmark_csv, index=False)
    with benchmark_json.open("w", encoding="utf-8") as f:
        json.dump(benchmark_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print("Saved benchmark summary CSV:", benchmark_csv)
    print("Saved benchmark summary JSON:", benchmark_json)


if __name__ == "__main__":
    main()
