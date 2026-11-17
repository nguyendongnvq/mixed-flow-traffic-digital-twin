from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (("mixed_s06", 0.6), ("mixed_s10", 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed multi-scale SUMO dataset.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--force-config", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--max-windows-per-source", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/processed/multilabel/"
            "dmfgcrn_multilabel_windows_multiscale_v2.npz"
        ),
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_variant(variant: str, scale: float, days: int, force_config: bool) -> None:
    config_command = [
        sys.executable,
        "-m",
        "src.simulation.build_mixed_simulation_config",
        "--variant",
        variant,
        "--days",
        str(days),
        "--traffic-scale",
        str(scale),
    ]
    if force_config:
        config_command.append("--force")
    run(config_command)
    run(
        [
            sys.executable,
            "-m",
            "src.simulation.xlsx_traci_flow_runner",
            "--config",
            f"src/simulation/configs/xlsx_traci_flow_config.{variant}.json",
            "--no-gui",
        ]
    )


def process_variant(variant: str) -> None:
    processed_dir = Path("datasets/processed") / variant
    run(
        [
            sys.executable,
            "-m",
            "src.process_rawdata.export_timeseries",
            "--modes",
            variant,
            "--detector-dir",
            "datasets/raw/sumo_data/detectors",
            "--scenarios-dir",
            "datasets/raw/sumo_data",
            "--processed-root",
            "datasets/processed",
            "--summary-path",
            str(processed_dir / "timeseries_export_summary.json"),
            "--strict",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.process_rawdata.build_forecast_node_level_from_timeseries",
            "--modes",
            variant,
            "--suffix",
            "",
            "--summary-path",
            str(processed_dir / "forecast_summary.json"),
            "--strict",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.process_rawdata.build_mixed_node_labels",
            "--forecast",
            str(processed_dir / f"forecast_{variant}_node_level.csv"),
            "--window-labels",
            f"datasets/raw/sumo_data/anomaly_window_labels_{variant}.csv",
            "--output",
            str(processed_dir / f"forecast_{variant}_node_level_with_labels.csv"),
        ]
    )


def main() -> None:
    args = parse_args()
    if args.days <= 0:
        raise ValueError("--days must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    for variant, scale in VARIANTS:
        labeled_path = (
            PROJECT_ROOT
            / "datasets"
            / "processed"
            / variant
            / f"forecast_{variant}_node_level_with_labels.csv"
        )
        if labeled_path.exists():
            print(f"[skip] processed variant exists: {variant}")
            continue
        if not args.skip_simulation:
            build_variant(variant, scale, args.days, args.force_config)
        process_variant(variant)
    command = [
        sys.executable,
        "-m",
        "src.preprocessing.build_multilabel_dataset",
        "--config",
        "configs/dmfgcrn_multilabel_multiscale.json",
        "--output",
        str(args.output),
    ]
    if args.max_windows_per_source is not None:
        command.extend(
            ["--max-windows-per-source", str(args.max_windows_per_source)]
        )
    run(command)
    summary = {
        "days": args.days,
        "variants": [
            {"variant": variant, "traffic_scale": scale}
            for variant, scale in VARIANTS
        ],
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
