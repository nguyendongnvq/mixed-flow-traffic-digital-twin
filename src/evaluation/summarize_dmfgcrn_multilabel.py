from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DMFGCRN multi-label test metrics against go-live targets."
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fp-max", type=float, default=0.15)
    parser.add_argument("--target-recall-min", type=float, default=0.75)
    parser.add_argument("--accident-f1-min", type=float, default=0.60)
    parser.add_argument("--sudden-stop-f1-min", type=float, default=0.65)
    parser.add_argument("--other-f1-min", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")

    metrics = pd.read_csv(args.metrics_csv)
    summary = json.loads(args.test_summary.read_text(encoding="utf-8"))
    required = {"class_name", "f1", "recall", "fp_rate"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics CSV missing columns: {sorted(missing)}")

    class_results = []
    for row in metrics.to_dict(orient="records"):
        class_name = str(row["class_name"])
        f1_floor = (
            args.accident_f1_min
            if class_name == "accident"
            else args.sudden_stop_f1_min
            if class_name == "sudden_stop"
            else args.other_f1_min
        )
        passed = float(row["f1"]) >= f1_floor
        if class_name == "accident":
            passed = passed and float(row["fp_rate"]) <= args.target_fp_max
        class_results.append(
            {
                **row,
                "f1_floor": f1_floor,
                "passes_fp_target": float(row["fp_rate"]) <= args.target_fp_max,
                "passes_recall_target": float(row["recall"])
                >= args.target_recall_min,
                "passes_f1_floor": float(row["f1"]) >= f1_floor,
                "go_live": passed,
            }
        )

    accident_result = next(
        row for row in class_results if row["class_name"] == "accident"
    )
    secondary_floors_pass = all(
        row["passes_f1_floor"]
        for row in class_results
        if row["class_name"] != "accident"
    )
    payload = {
        "target_fp_max": args.target_fp_max,
        "target_recall_min": args.target_recall_min,
        "accident_f1_min": args.accident_f1_min,
        "sudden_stop_f1_min": args.sudden_stop_f1_min,
        "other_f1_min": args.other_f1_min,
        "accident_go_live": accident_result["go_live"],
        "secondary_floors_pass": secondary_floors_pass,
        "all_classes_go_live": accident_result["go_live"] and secondary_floors_pass,
        "class_results": class_results,
        "test_summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output={args.output}")
    print(f"accident_go_live={payload['accident_go_live']}")
    print(f"secondary_floors_pass={payload['secondary_floors_pass']}")
    print(f"all_classes_go_live={payload['all_classes_go_live']}")
    for row in class_results:
        print(
            f"{row['class_name']}: go_live={row['go_live']} "
            f"recall={float(row['recall']):.4f} fp_rate={float(row['fp_rate']):.4f}"
        )


if __name__ == "__main__":
    main()
