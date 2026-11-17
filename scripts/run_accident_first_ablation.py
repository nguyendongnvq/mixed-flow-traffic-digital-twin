from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


VARIANTS = {
    "best_checkpoint": {
        "class_loss_weights": {
            "accident": 1.0,
            "congestion_spike": 1.0,
            "zigzag": 1.0,
            "sudden_stop": 1.0,
        },
        "sampling_weights": {},
        "accident_pooling": {"enabled": False},
    },
    "weighted_loss": {
        "sampling_weights": {},
        "accident_pooling": {"enabled": False},
    },
    "weighted_loss_sampling": {
        "accident_pooling": {"enabled": False},
    },
    "weighted_loss_sampling_pooling": {},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run accident-first DMFGCRN ablations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dmfgcrn_multilabel.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/processed/multilabel/"
            "dmfgcrn_multilabel_windows_with_mixed_14d.npz"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/outputs_dmfgcrn_accident_first_ablation"),
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def merged_config(
    base_config: dict[str, Any],
    variant_overrides: dict[str, Any],
    seed: int,
    *,
    smoke: bool,
) -> dict[str, Any]:
    config = dict(base_config)
    for key, value in variant_overrides.items():
        config[key] = value
    config["seed"] = seed
    if smoke:
        config["head_epochs"] = 1
        config["fine_tune_epochs"] = 0
        config["early_stopping"] = {"patience": 1, "min_delta": 0.002}
    return config


def run_command(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def run_variant(
    *,
    variant_name: str,
    config: dict[str, Any],
    dataset: Path,
    output_dir: Path,
    smoke: bool,
) -> None:
    metrics_path = output_dir / "test_metrics_by_class.csv"
    if metrics_path.exists():
        print(f"[skip] {variant_name} seed={config['seed']}")
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    config_path = output_dir.parent / f"{output_dir.name}.config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = output_dir.parent / f"{output_dir.name}.log"
    command = [
        sys.executable,
        "-m",
        "src.train.train_dmfgcrn_multilabel",
        "--config",
        str(config_path),
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
    ]
    if smoke:
        command.extend(["--max-train-samples", "256", "--max-eval-samples", "256"])
    print(f"[run] {variant_name} seed={config['seed']}")
    run_command(command, log_path)

    run_command(
        [
            sys.executable,
            "-m",
            "src.evaluation.summarize_dmfgcrn_multilabel",
            "--metrics-csv",
            str(metrics_path),
            "--test-summary",
            str(output_dir / "test_summary.json"),
            "--output",
            str(output_dir / "go_live_decision.json"),
        ],
        log_path,
    )


def summarize(output_root: Path, seeds: list[int]) -> None:
    rows = []
    for variant_name in VARIANTS:
        for seed in seeds:
            output_dir = output_root / f"{variant_name}_seed_{seed}"
            metrics_path = output_dir / "test_metrics_by_class.csv"
            if not metrics_path.exists():
                continue
            metrics = pd.read_csv(metrics_path).set_index("class_name")
            rows.append(
                {
                    "variant": variant_name,
                    "seed": seed,
                    "accident_f1": metrics.loc["accident", "f1"],
                    "accident_recall": metrics.loc["accident", "recall"],
                    "accident_fp_rate": metrics.loc["accident", "fp_rate"],
                    "congestion_spike_f1": metrics.loc["congestion_spike", "f1"],
                    "zigzag_f1": metrics.loc["zigzag", "f1"],
                    "sudden_stop_f1": metrics.loc["sudden_stop", "f1"],
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(output_root / "ablation_runs.csv", index=False)
    if results.empty:
        return

    summary = (
        results.groupby("variant", as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["seed"])
    )
    summary["eligible"] = (
        (summary["accident_fp_rate"] <= 0.15)
        & (summary["congestion_spike_f1"] >= 0.20)
        & (summary["zigzag_f1"] >= 0.20)
        & (summary["sudden_stop_f1"] >= 0.65)
    )
    summary = summary.sort_values(
        ["eligible", "accident_f1", "accident_recall", "accident_fp_rate"],
        ascending=[False, False, False, True],
    )
    summary.to_csv(output_root / "ablation_summary.csv", index=False)
    eligible = summary.loc[summary["eligible"]]
    winner = eligible.iloc[0].to_dict() if not eligible.empty else None
    best_candidate = summary.iloc[0].to_dict()
    selection = {
        "promote": winner is not None,
        "selected": winner,
        "best_candidate": best_candidate,
        "fallback": None if winner is not None else "keep_current_production_checkpoint",
    }
    (output_root / "selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] promote={selection['promote']} "
        f"candidate={best_candidate['variant']}"
    )


def main() -> None:
    args = parse_args()
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    for variant_name, overrides in VARIANTS.items():
        for seed in seeds:
            config = merged_config(
                base_config,
                overrides,
                seed,
                smoke=args.smoke,
            )
            run_variant(
                variant_name=variant_name,
                config=config,
                dataset=args.dataset,
                output_dir=args.output_root / f"{variant_name}_seed_{seed}",
                smoke=args.smoke,
            )
    summarize(args.output_root, seeds)


if __name__ == "__main__":
    main()
