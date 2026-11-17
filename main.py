from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.configs.config import load_config
from src.preprocessing.data import (
    df_to_tensor,
    ensure_same_nodes,
    load_dataframes,
    make_time_labels,
    to_feature_weights_array,
)
from src.evaluation.event_eval import (
    build_accident_result_df,
    evaluate_event_level,
    infer_step_seconds,
    summarize_metrics,
)
from src.utils.io_utils import ensure_output_dir, save_model_artifact, save_runtime_config, save_summary_files
from src.models.model import LSTMAutoencoder
from src.models.dmfgcrn_adapter import DMFGCRNForecaster, build_forecast_windows
from src.utils.plotting import (
    plot_delay_charts,
    plot_score_distribution,
    plot_score_timeline,
    plot_summary_rates,
    plot_top_alert_nodes,
    plot_training_history,
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
from src.evaluation.scoring import (
    apply_alert_logic,
    compute_weighted_node_max_scores,
    compute_weighted_node_max_scores_from_error,
    predict_reconstruction,
    select_threshold,
)
from src.train.trainer import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM autoencoder anomaly detector pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lstm_autoencoder_accident.json",
        help="Path to JSON config file.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_output_dir(config["output_dir"])

    print("Device:", device)
    print("Output dir:", output_dir)
    print("Backend:", config.get("model_backend", "lstm_autoencoder"))

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

    print("df_normal:", df_normal_clean.shape)
    print("df_accident:", df_accident_clean.shape)
    print("\nNormal label counts:")
    print(df_normal_clean[label_col].value_counts(dropna=False))
    print("\nAccident label counts:")
    print(df_accident_clean[label_col].value_counts(dropna=False))

    nodes = ensure_same_nodes(df_normal_clean, df_accident_clean, node_col)
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    print("nodes:", len(nodes))

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
    print("X_train:", split.x_train.shape)
    print("X_val:", split.x_val.shape)
    print("X_test_normal:", split.x_test_normal.shape)

    scaler = fit_scaler(split.x_train)
    x_train_scaled = scale_tensor(split.x_train, scaler)
    x_val_scaled = scale_tensor(split.x_val, scaler)
    x_test_normal_scaled = scale_tensor(split.x_test_normal, scaler)
    x_accident_scaled = scale_tensor(x_accident, scaler)

    _, n_nodes, n_features = split.x_train.shape
    print("\nScaled shapes:")
    print("X_train_scaled:", x_train_scaled.shape)
    print("X_accident_scaled:", x_accident_scaled.shape)
    print("\nTrain scaled mean:", x_train_scaled.reshape(-1, n_features).mean(axis=0))
    print("Train scaled std:", x_train_scaled.reshape(-1, n_features).std(axis=0))

    seq_size = config["seq_size"]
    backend = str(config.get("model_backend", "lstm_autoencoder")).lower()
    model = None
    input_dim = 0
    artifact_payload: dict[str, object]

    if backend == "lstm_autoencoder":
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

        train_loader = make_loader(train_x, config["batch_size"], shuffle=True)
        val_loader = make_loader(val_x, config["batch_size"], shuffle=False)
        test_normal_loader = make_loader(test_normal_x, config["batch_size"], shuffle=False)
        accident_loader = make_loader(accident_x, config["batch_size"], shuffle=False)

        input_dim = train_x.shape[2]
        model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)
        print(model)

        model, history, _ = train_model(
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

        val_weighted, _ = compute_weighted_node_max_scores(
            val_batch_x,
            val_recon,
            n_nodes,
            n_features,
            seq_size,
            feature_weights,
        )
        test_normal_weighted, _ = compute_weighted_node_max_scores(
            test_normal_batch_x,
            test_normal_recon,
            n_nodes,
            n_features,
            seq_size,
            feature_weights,
        )
        accident_weighted, accident_weighted_node_scores = compute_weighted_node_max_scores(
            accident_batch_x,
            accident_recon,
            n_nodes,
            n_features,
            seq_size,
            feature_weights,
        )

        accident_window_end_begin = np.array(accident_timesteps[seq_size - 1 :], dtype=float)
        test_normal_timesteps = normal_timesteps[split.val_end :]
        test_normal_window_end_begin = np.array(test_normal_timesteps[seq_size - 1 :], dtype=float)

    elif backend == "dmfgcrn":
        lag = int(seq_size)
        horizon_index = int(config["dmfgcrn_horizon_index"])
        dmfgcrn_input_features = list(config["dmfgcrn_input_features"])
        dmfgcrn_output_features = list(config["dmfgcrn_output_features"])

        feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
        missing_input = [name for name in dmfgcrn_input_features if name not in feature_to_idx]
        missing_output = [name for name in dmfgcrn_output_features if name not in feature_to_idx]
        if missing_input:
            raise ValueError(f"DMFGCRN input features missing from feature_cols: {missing_input}")
        if missing_output:
            raise ValueError(f"DMFGCRN output features missing from feature_cols: {missing_output}")

        input_indices = [feature_to_idx[name] for name in dmfgcrn_input_features]
        output_indices = [feature_to_idx[name] for name in dmfgcrn_output_features]

        dmfgcrn_device_raw = str(config.get("dmfgcrn_device", str(device))).lower()
        if dmfgcrn_device_raw.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested for DMFGCRN but not available. Falling back to CPU.")
            dmfgcrn_device = torch.device("cpu")
        else:
            dmfgcrn_device = torch.device(dmfgcrn_device_raw)

        forecaster = DMFGCRNForecaster(
            project_root=Path(__file__).resolve().parent,
            checkpoint_path=Path(config["dmfgcrn_checkpoint_path"]),
            conf_path=Path(config["dmfgcrn_conf_path"]),
            num_nodes=n_nodes,
            lag=lag,
            device=dmfgcrn_device,
        )

        output_feature_weights = to_feature_weights_array(dmfgcrn_output_features, config["feature_weights"])
        steps_per_day = int(config.get("dmfgcrn_steps_per_day", 288))

        val_timesteps = normal_timesteps[split.train_end : split.val_end]
        normal_timesteps_test = normal_timesteps[split.val_end :]

        val_x, val_y, _, _ = build_forecast_windows(
            x_scaled=x_val_scaled,
            timesteps=val_timesteps,
            step_labels=np.zeros(len(val_timesteps), dtype=np.int32),
            lag=lag,
            horizon_index=horizon_index,
            input_indices=input_indices,
            output_indices=output_indices,
            use_day=bool(forecaster.args.use_day),
            use_week=bool(forecaster.args.use_week),
            steps_per_day=steps_per_day,
        )
        normal_x, normal_y, _, normal_window_end_begin = build_forecast_windows(
            x_scaled=x_test_normal_scaled,
            timesteps=normal_timesteps_test,
            step_labels=np.zeros(len(normal_timesteps_test), dtype=np.int32),
            lag=lag,
            horizon_index=horizon_index,
            input_indices=input_indices,
            output_indices=output_indices,
            use_day=bool(forecaster.args.use_day),
            use_week=bool(forecaster.args.use_week),
            steps_per_day=steps_per_day,
        )
        accident_x, accident_y, accident_window_labels, accident_window_end_begin = build_forecast_windows(
            x_scaled=x_accident_scaled,
            timesteps=accident_timesteps,
            step_labels=accident_time_labels,
            lag=lag,
            horizon_index=horizon_index,
            input_indices=input_indices,
            output_indices=output_indices,
            use_day=bool(forecaster.args.use_day),
            use_week=bool(forecaster.args.use_week),
            steps_per_day=steps_per_day,
        )

        pred_val = forecaster.predict(val_x, int(config["batch_size"]), horizon_index)
        pred_normal = forecaster.predict(normal_x, int(config["batch_size"]), horizon_index)
        pred_accident = forecaster.predict(accident_x, int(config["batch_size"]), horizon_index)

        err_val = np.abs(pred_val - val_y)
        err_normal = np.abs(pred_normal - normal_y)
        err_accident = np.abs(pred_accident - accident_y)

        val_weighted, _ = compute_weighted_node_max_scores_from_error(err_val, output_feature_weights)
        test_normal_weighted, _ = compute_weighted_node_max_scores_from_error(
            err_normal,
            output_feature_weights,
        )
        accident_weighted, accident_weighted_node_scores = compute_weighted_node_max_scores_from_error(
            err_accident,
            output_feature_weights,
        )

        test_normal_window_end_begin = normal_window_end_begin.astype(float)
        accident_window_end_begin = accident_window_end_begin.astype(float)
    else:
        raise ValueError(f"Unsupported model_backend: {backend}")

    selected_threshold = select_threshold(val_weighted, config["threshold_percentile"])
    normal_pred_window = test_normal_weighted > selected_threshold
    accident_pred_window = accident_weighted > selected_threshold

    min_consecutive = config["min_consecutive"]
    cooldown_steps = config["cooldown_steps"]
    normal_alerts = apply_alert_logic(
        test_normal_weighted,
        selected_threshold,
        min_consecutive=min_consecutive,
        cooldown_steps=cooldown_steps,
    )
    accident_alerts = apply_alert_logic(
        accident_weighted,
        selected_threshold,
        min_consecutive=min_consecutive,
        cooldown_steps=cooldown_steps,
    )

    print("Score:", config["score_name"])
    print("Threshold percentile:", config["threshold_percentile"])
    print("Threshold:", selected_threshold)
    print("\nWindow-level")
    print("Normal FP:", int(normal_pred_window.sum()), "/", len(normal_pred_window))
    print("Normal FP rate:", float(normal_pred_window.mean()))
    print("Accident detected:", int(accident_pred_window.sum()), "/", len(accident_pred_window))
    print("Accident detect ratio:", float(accident_pred_window.mean()))

    step_seconds = infer_step_seconds(normal_timesteps)
    event_alert_col = str(config.get("event_alert_col", "alert"))
    raw_event_gap_seconds = config.get("event_gap_seconds")
    if raw_event_gap_seconds is not None:
        event_gap_seconds = float(raw_event_gap_seconds)
    else:
        event_gap_multiplier = float(config.get("event_gap_multiplier", 1.0))
        event_gap_seconds = (
            float(step_seconds * event_gap_multiplier) if not np.isnan(step_seconds) else np.nan
        )
    result_df = build_accident_result_df(
        accident_weighted=accident_weighted,
        selected_threshold=selected_threshold,
        accident_pred_window=accident_pred_window,
        accident_alerts=accident_alerts,
        accident_window_labels=accident_window_labels,
        accident_weighted_node_scores=accident_weighted_node_scores,
        accident_timesteps=accident_timesteps,
        seq_size=seq_size,
        idx_to_node=idx_to_node,
        window_end_begin=accident_window_end_begin,
    )
    if event_alert_col not in result_df.columns:
        raise ValueError(
            f"event_alert_col='{event_alert_col}' not found in result dataframe columns: "
            f"{list(result_df.columns)}"
        )
    event_eval_alert = evaluate_event_level(
        result_df,
        alert_col=event_alert_col,
        gap_threshold=event_gap_seconds,
    )

    summary, summary_df = summarize_metrics(
        score_name=config["score_name"],
        threshold_percentile=config["threshold_percentile"],
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

    print(summary_df)
    print(event_eval_alert.head(20))

    save_summary_files(
        output_dir=output_dir,
        summary=summary,
        summary_df=summary_df,
        result_df=result_df,
        event_eval_alert=event_eval_alert,
    )

    plot_score_distribution(
        test_normal_weighted=test_normal_weighted,
        accident_weighted=accident_weighted,
        threshold=selected_threshold,
        threshold_percentile=config["threshold_percentile"],
        output_dir=output_dir,
    )
    plot_score_timeline(
        scores=test_normal_weighted,
        threshold=selected_threshold,
        alerts=normal_alerts,
        title="Normal test score timeline",
        output_name="normal_score_timeline.png",
        output_dir=output_dir,
    )
    plot_score_timeline(
        scores=accident_weighted,
        threshold=selected_threshold,
        alerts=accident_alerts,
        title="Accident score timeline",
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

    model_output_name = str(config.get("model_output_name", "lstm_autoencoder_weighted_node_max_p85.pt"))
    if backend == "dmfgcrn" and model_output_name.endswith(".pt"):
        model_output_name = model_output_name.replace(".pt", "_dmfgcrn.pt")

    if backend == "lstm_autoencoder":
        artifact_payload = {
            "backend": backend,
            "model_state_dict": model.state_dict() if model is not None else None,
            "input_dim": input_dim,
            "seq_size": seq_size,
            "N": n_nodes,
            "F": n_features,
            "feature_cols": feature_cols,
            "nodes": nodes,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "score_name": config["score_name"],
            "threshold_percentile": config["threshold_percentile"],
            "threshold": selected_threshold,
            "feature_weights": feature_weights,
            "min_consecutive": min_consecutive,
            "cooldown_steps": cooldown_steps,
            "step_seconds": step_seconds,
            "event_alert_col": event_alert_col,
            "event_gap_seconds": event_gap_seconds,
            "summary": summary,
        }
    else:
        artifact_payload = {
            "backend": backend,
            "seq_size": seq_size,
            "N": n_nodes,
            "F": n_features,
            "feature_cols": feature_cols,
            "nodes": nodes,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "score_name": config["score_name"],
            "threshold_percentile": config["threshold_percentile"],
            "threshold": selected_threshold,
            "feature_weights": feature_weights,
            "min_consecutive": min_consecutive,
            "cooldown_steps": cooldown_steps,
            "step_seconds": step_seconds,
            "event_alert_col": event_alert_col,
            "event_gap_seconds": event_gap_seconds,
            "summary": summary,
            "dmfgcrn_checkpoint_path": config["dmfgcrn_checkpoint_path"],
            "dmfgcrn_conf_path": config["dmfgcrn_conf_path"],
            "dmfgcrn_horizon_index": config["dmfgcrn_horizon_index"],
            "dmfgcrn_input_features": config["dmfgcrn_input_features"],
            "dmfgcrn_output_features": config["dmfgcrn_output_features"],
        }

    model_path = save_model_artifact(
        output_dir=output_dir,
        model_output_name=model_output_name,
        payload=artifact_payload,
    )

    config_to_save = dict(config)
    config_to_save["feature_cols"] = feature_cols
    config_to_save["nodes"] = nodes
    config_to_save["selected_threshold"] = selected_threshold
    saved_config_path = save_runtime_config(output_dir, config_to_save)

    print("Saved summary and result files to:", output_dir)
    print("Saved model:", model_path)
    print("Saved config:", saved_config_path)


if __name__ == "__main__":
    main()
