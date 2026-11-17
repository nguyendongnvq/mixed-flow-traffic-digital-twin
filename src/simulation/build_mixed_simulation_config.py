from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("accident", "congestion_spike", "zigzag", "sudden_stop")
DURATIONS = {
    "accident": 1800,
    "congestion_spike": 1200,
    "zigzag": 300,
    "sudden_stop": 900,
}
INTERSECTION_EDGES = {
    "pdl_ltn": ("693841066#3", "295408469#11", "-693841066#4", "-695509454#0"),
    "pdl_htp": ("295853897#4", "295408469#6", "-295853897#5", "-295408469#7"),
    "pdl_lnh": ("295853900#8", "295408469#2", "-295853900#9", "-295408469#4"),
    "xoviet_ltn": ("693841066#9", "302756236#2", "-693841066#11", "695838784#1"),
    "xoviet_htp": ("295853897#9", "694320789#1", "-295853897#11", "695838779#1"),
    "xoviet_lnh": ("295853900#15", "694314742#1", "-302756231#1", "1281556686#2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic mixed SUMO artifacts.")
    parser.add_argument("--variant", default="mixed")
    parser.add_argument("--days", type=int)
    parser.add_argument("--traffic-scale", type=float, default=0.2)
    parser.add_argument(
        "--flow-mode",
        choices=["time_varying", "constant_scaled"],
        default="time_varying",
    )
    parser.add_argument("--schedule-day-offset", type=int, default=0)
    parser.add_argument("--edge-rotation-offset", type=int, default=0)
    parser.add_argument("--sumo-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def episode_combinations() -> list[tuple[str, ...]]:
    return [
        combo
        for size in range(1, len(CLASS_NAMES) + 1)
        for combo in itertools.combinations(CLASS_NAMES, size)
    ]


def build_schedule(
    days: int,
    *,
    day_offset: int = 0,
    edge_rotation_offset: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    intersections = list(INTERSECTION_EDGES)
    schedule: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    combinations = episode_combinations()
    for day_index in range(days):
        global_day_index = day_offset + day_index
        for slot_index, combo in enumerate(combinations):
            step = day_index * 86400 + 3600 + slot_index * 5400
            chosen = [
                intersections[
                    (global_day_index + slot_index + offset) % len(intersections)
                ]
                for offset in range(len(combo))
            ]
            episode_id = f"d{global_day_index + 1:02d}_s{slot_index + 1:02d}"
            events: list[dict[str, object]] = []
            for class_offset, (class_name, intersection_id) in enumerate(zip(combo, chosen)):
                edge_choices = INTERSECTION_EDGES[intersection_id]
                edge_id = edge_choices[
                    (
                        global_day_index
                        + slot_index
                        + class_offset
                        + edge_rotation_offset
                    )
                    % len(edge_choices)
                ]
                event = {
                    "id": f"{episode_id}_{class_name}",
                    "step": step,
                    "kind": class_name,
                    "edge_id": edge_id,
                    "params": {"duration_steps": DURATIONS[class_name]},
                }
                schedule.append(event)
                events.append(
                    {
                        "kind": class_name,
                        "intersection_id": intersection_id,
                        "edge_id": edge_id,
                        "duration_steps": DURATIONS[class_name],
                    }
                )
            episodes.append(
                {
                    "episode_id": episode_id,
                    "day": global_day_index + 1,
                    "local_day": day_index + 1,
                    "slot": slot_index + 1,
                    "step": step,
                    "classes": list(combo),
                    "events": events,
                }
            )
    return schedule, episodes


def write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    days = args.days or (1 if args.variant == "mixed_smoke" else 14)
    if days <= 0:
        raise ValueError("--days must be positive")
    if args.traffic_scale <= 0:
        raise ValueError("--traffic-scale must be positive")
    if args.schedule_day_offset < 0:
        raise ValueError("--schedule-day-offset must be non-negative")
    if args.edge_rotation_offset < 0:
        raise ValueError("--edge-rotation-offset must be non-negative")

    config_dir = PROJECT_ROOT / "src" / "simulation" / "configs"
    scenario_dir = PROJECT_ROOT / "sumo" / "scenarios"
    raw_dir = PROJECT_ROOT / "datasets" / "raw" / "sumo_data"
    baseline_config = config_dir / "xlsx_traci_flow_config.baseline.json"
    template_additional = (
        PROJECT_ROOT
        / "sumo"
        / "scenarios"
        / "priority_tls.baseline.add.xml"
    )
    if not template_additional.exists():
        raise FileNotFoundError(f"Missing detector template: {template_additional}")

    variant = args.variant
    schedule, episodes = build_schedule(
        days,
        day_offset=args.schedule_day_offset,
        edge_rotation_offset=args.edge_rotation_offset,
    )
    config = deepcopy(json.loads(baseline_config.read_text(encoding="utf-8")))
    start_dt = datetime(2025, 10, 1) + timedelta(days=args.schedule_day_offset)
    end_dt = start_dt + timedelta(days=days) - timedelta(seconds=1)
    config["sumo"].update(
        {
            "sumocfg_path": f"../../../sumo/scenarios/named_roads_updated.{variant}.sumocfg",
            "runtime_route_path": f"../../../sumo/routes/named_roads_updated.traci_routes_only.{variant}.rou.xml",
            "use_gui": False,
            "start_datetime_override": start_dt.isoformat(sep=" "),
            "end_datetime_override": end_dt.isoformat(sep=" "),
            "seed": args.sumo_seed,
        }
    )
    config["traffic"]["traffic_scale"] = args.traffic_scale
    config["traffic"]["flow_mode"] = args.flow_mode
    config["anomaly"] = {
        "mode": "mixed",
        "schedule": schedule,
        "label": {
            "step_output_path": f"../../../datasets/raw/sumo_data/anomaly_step_labels_{variant}.csv",
            "window_output_path": f"../../../datasets/raw/sumo_data/anomaly_window_labels_{variant}.csv",
            "spawn_output_path": f"../../../datasets/raw/sumo_data/anomaly_spawn_records_{variant}.json",
            "window_sec": 300,
        },
    }

    additional_text = template_additional.read_text(encoding="utf-8")
    additional_text = additional_text.replace(
        "../../datasets/raw/sumo_data/detectors/lanearea_output_baseline.xml",
        f"../../datasets/raw/sumo_data/detectors/lanearea_output_{variant}.xml",
    ).replace(
        "../../datasets/raw/sumo_data/detectors/inductionloop_output_baseline.xml",
        f"../../datasets/raw/sumo_data/detectors/inductionloop_output_{variant}.xml",
    )
    sumocfg_text = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<configuration xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
        "xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/sumoConfiguration.xsd\">\n"
        "  <input>\n"
        "    <net-file value=\"../network/9nut.net.xml\" />\n"
        "    <route-files value=\"../routes/named_roads_updated.rou.xml\" />\n"
        f"    <additional-files value=\"priority_tls.{variant}.add.xml\" />\n"
        "  </input>\n"
        "  <time>\n"
        "    <begin value=\"0\" />\n"
        f"    <end value=\"{days * 86400}\" />\n"
        "    <step-length value=\"1\" />\n"
        "  </time>\n"
        "  <processing>\n"
        "    <lateral-resolution value=\"0.7\" />\n"
        "    <collision.action value=\"warn\" />\n"
        "    <collision.check-junctions value=\"true\" />\n"
        "  </processing>\n"
        "  <report>\n"
        "    <duration-log.statistics value=\"true\" />\n"
        "    <no-step-log value=\"true\" />\n"
        "    <verbose value=\"false\" />\n"
        "  </report>\n"
        "</configuration>\n"
    )
    class_counts = {
        name: sum(1 for event in schedule if event["kind"] == name)
        for name in CLASS_NAMES
    }
    metadata = {
        "variant": variant,
        "traffic_scale": args.traffic_scale,
        "flow_mode": args.flow_mode,
        "schedule_day_offset": args.schedule_day_offset,
        "edge_rotation_offset": args.edge_rotation_offset,
        "sumo_seed": args.sumo_seed,
        "days": days,
        "start_datetime": start_dt.isoformat(sep=" "),
        "end_datetime": end_dt.isoformat(sep=" "),
        "episode_count": len(episodes),
        "event_count": len(schedule),
        "class_event_counts": class_counts,
        "episodes": episodes,
    }

    write_text(
        config_dir / f"xlsx_traci_flow_config.{variant}.json",
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )
    write_text(
        scenario_dir / f"priority_tls.{variant}.add.xml",
        additional_text,
        args.force,
    )
    write_text(
        scenario_dir / f"named_roads_updated.{variant}.sumocfg",
        sumocfg_text,
        args.force,
    )
    write_text(
        raw_dir / f"mixed_schedule_{variant}.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )
    print(
        f"variant={variant} days={days} episodes={len(episodes)} "
        f"traffic_scale={args.traffic_scale} flow_mode={args.flow_mode} "
        f"events={len(schedule)} "
        f"class_counts={class_counts}"
    )


if __name__ == "__main__":
    main()
