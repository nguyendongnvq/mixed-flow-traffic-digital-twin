"""Export per-mode timeseries from SUMO detector XML and rebuild dataset artifacts.

This script replaces notebook-only steps for:
- XML detector parsing (E1 induction loop + E2 lanearea),
- long timeseries CSV export per mode,
- baseline dataset artifacts used by training scripts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODES = ["baseline", "accident", "congestion_spike", "zigzag", "sudden_stop"]
ANOMALY_CLASS_NAMES = ("accident", "congestion_spike", "zigzag", "sudden_stop")
DEFAULT_FEATURE_COLS = [
    "flow_count_5m",
    "speed",
    "e1_occupancy",
    "e2_meanOccupancy",
    "time_loss",
    "started_halts",
    "jam_veh",
]


@dataclass(frozen=True)
class ModePaths:
    mode: str
    e1_xml: Path
    e2_xml: Path
    step_labels: Path
    window_labels: Path
    spawn_records: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export traffic timeseries from detector XML for one or more anomaly modes "
            "and rebuild baseline dataset artifacts for LSTM training."
        )
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes (default: baseline,accident,congestion_spike,zigzag).",
    )
    parser.add_argument(
        "--detector-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw" / "sumo_data" / "detectors",
        help="Directory containing detector XML files.",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw" / "sumo_data",
        help="Directory containing anomaly label/record files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dataset directory. Default: datasets/processed/traffic_dataset_<today>_run01.",
    )
    parser.add_argument(
        "--global-mode",
        type=str,
        default="baseline",
        help="Mode used to build traffic_dataset.npz and training artifacts.",
    )
    parser.add_argument(
        "--window-sec",
        type=int,
        default=300,
        help="Expected detector interval seconds (default: 300).",
    )
    parser.add_argument(
        "--node-mode",
        choices=["lane", "road_dir", "road"],
        default="lane",
        help="Node aggregation mode (default: lane).",
    )
    parser.add_argument(
        "--edges-template",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw" / "sumo_data" / "priority_tls_sensors_for_edges.csv",
        help="Edge CSV template used to export graph artifacts.",
    )
    parser.add_argument("--input-steps", type=int, default=12, help="Input steps for traffic_sequence_dataset.npz.")
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=12,
        help="Horizon steps for traffic_sequence_dataset.npz.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="flow_count_5m",
        help="Target feature in traffic_sequence_dataset.npz.",
    )
    parser.add_argument(
        "--skip-global-dataset",
        action="store_true",
        help="Only export per-mode timeseries CSV, skip traffic_dataset*.npz artifacts.",
    )
    parser.add_argument(
        "--skip-graph-export",
        action="store_true",
        help="Skip graph artifact export even when edges template is available.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any expected input artifact is missing or inconsistent.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return PROJECT_ROOT / "datasets" / "processed" / f"traffic_dataset_{stamp}_run01"


def parse_modes(value: str) -> list[str]:
    modes = [x.strip() for x in value.split(",") if x.strip()]
    if not modes:
        raise ValueError("No mode specified in --modes.")
    return modes


def parse_detector_id(detector_id: str) -> dict[str, Any]:
    base = detector_id[:-5] if detector_id.endswith("_loop") else detector_id
    m = re.match(r"(?P<road>.+)_lane_(?P<lane>\d+)_(?P<dir>[NSEW])$", base)
    if not m:
        return {
            "base_id": base,
            "road_key": base,
            "lane_idx": np.nan,
            "direction": None,
        }
    return {
        "base_id": base,
        "road_key": m.group("road"),
        "lane_idx": int(m.group("lane")),
        "direction": m.group("dir"),
    }


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return float(np.average(values_arr[mask], weights=weights_arr[mask]))


def get_node_id(df: pd.DataFrame, node_mode: str) -> pd.Series:
    if node_mode == "lane":
        return df["base_id"].astype(str)
    if node_mode == "road_dir":
        return df["road_key"].astype(str) + "_" + df["direction"].fillna("X").astype(str)
    if node_mode == "road":
        return df["road_key"].astype(str)
    raise ValueError(f"Unsupported node mode: {node_mode}")


def parse_e1_xml(xml_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "interval":
            continue
        rid = str(elem.attrib.get("id", ""))
        meta = parse_detector_id(rid)
        rows.append(
            {
                "begin": safe_float(elem.attrib.get("begin"), 0.0),
                "end": safe_float(elem.attrib.get("end"), 0.0),
                "detector_id_e1": rid,
                "base_id": meta["base_id"],
                "road_key": meta["road_key"],
                "lane_idx": meta["lane_idx"],
                "direction": meta["direction"],
                "nVehContrib_e1": safe_float(elem.attrib.get("nVehContrib"), 0.0),
                "flow_veh_h_e1": safe_float(elem.attrib.get("flow"), 0.0),
                "e1_occupancy": safe_float(elem.attrib.get("occupancy"), np.nan),
                "speed_e1": safe_float(elem.attrib.get("speed"), np.nan),
                "harmonic_speed_e1": safe_float(elem.attrib.get("harmonicMeanSpeed"), np.nan),
                "length_e1": safe_float(elem.attrib.get("length"), np.nan),
                "nVehEntered_e1": safe_float(elem.attrib.get("nVehEntered"), 0.0),
            }
        )
        elem.clear()

    df = pd.DataFrame(rows)
    for col in ["e1_occupancy", "speed_e1", "harmonic_speed_e1", "length_e1"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    return df


def parse_e2_xml(xml_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "interval":
            continue
        rid = str(elem.attrib.get("id", ""))
        meta = parse_detector_id(rid)
        rows.append(
            {
                "begin": safe_float(elem.attrib.get("begin"), 0.0),
                "end": safe_float(elem.attrib.get("end"), 0.0),
                "detector_id_e2": rid,
                "base_id": meta["base_id"],
                "road_key": meta["road_key"],
                "lane_idx": meta["lane_idx"],
                "direction": meta["direction"],
                "sampledSeconds": safe_float(elem.attrib.get("sampledSeconds"), 0.0),
                "nVehEntered_e2": safe_float(elem.attrib.get("nVehEntered"), 0.0),
                "nVehLeft_e2": safe_float(elem.attrib.get("nVehLeft"), 0.0),
                "nVehSeen_e2": safe_float(elem.attrib.get("nVehSeen"), 0.0),
                "meanSpeed_e2": safe_float(elem.attrib.get("meanSpeed"), np.nan),
                "meanTimeLoss": safe_float(elem.attrib.get("meanTimeLoss"), np.nan),
                "e2_meanOccupancy": safe_float(elem.attrib.get("meanOccupancy"), np.nan),
                "maxOccupancy": safe_float(elem.attrib.get("maxOccupancy"), np.nan),
                "meanMaxJamLengthInVehicles": safe_float(elem.attrib.get("meanMaxJamLengthInVehicles"), 0.0),
                "maxJamLengthInVehicles": safe_float(elem.attrib.get("maxJamLengthInVehicles"), 0.0),
                "meanHaltingDuration": safe_float(elem.attrib.get("meanHaltingDuration"), 0.0),
                "startedHalts": safe_float(elem.attrib.get("startedHalts"), 0.0),
                "meanVehicleNumber": safe_float(elem.attrib.get("meanVehicleNumber"), 0.0),
                "maxVehicleNumber": safe_float(elem.attrib.get("maxVehicleNumber"), 0.0),
            }
        )
        elem.clear()

    df = pd.DataFrame(rows)
    for col in ["meanSpeed_e2", "meanTimeLoss", "e2_meanOccupancy", "maxOccupancy"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    return df


def aggregate_to_node(df: pd.DataFrame, node_mode: str) -> pd.DataFrame:
    df = df.copy()
    df["node_id"] = get_node_id(df, node_mode=node_mode)

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["begin", "end", "node_id"], dropna=False, sort=True)
    for (begin, end, node_id), g in grouped:
        rows.append(
            {
                "begin": begin,
                "end": end,
                "node_id": node_id,
                "flow_count_5m": g["nVehEntered_e1"].sum(min_count=1),
                "flow_veh_h": g["flow_veh_h_e1"].sum(min_count=1),
                "speed": weighted_mean(g["meanSpeed_e2"], g["nVehSeen_e2"].replace(0, np.nan)),
                "harmonic_speed": weighted_mean(g["harmonic_speed_e1"], g["nVehContrib_e1"].replace(0, np.nan)),
                "e1_occupancy": weighted_mean(g["e1_occupancy"], g["nVehContrib_e1"].replace(0, np.nan)),
                "e2_meanOccupancy": weighted_mean(g["e2_meanOccupancy"], g["sampledSeconds"].replace(0, np.nan)),
                "time_loss": weighted_mean(g["meanTimeLoss"], g["nVehSeen_e2"].replace(0, np.nan)),
                "started_halts": g["startedHalts"].sum(min_count=1),
                "jam_veh": g["maxJamLengthInVehicles"].max(),
                "vehicle_number_mean": g["meanVehicleNumber"].sum(min_count=1),
            }
        )

    out = pd.DataFrame(rows).sort_values(["begin", "node_id"]).reset_index(drop=True)
    fill_zero_cols = [
        "flow_count_5m",
        "flow_veh_h",
        "started_halts",
        "jam_veh",
        "vehicle_number_mean",
        "e1_occupancy",
        "e2_meanOccupancy",
        "time_loss",
    ]
    for col in fill_zero_cols:
        out[col] = out[col].fillna(0.0)

    out["speed_is_missing"] = out["speed"].isna().astype(int)
    out["harmonic_speed_is_missing"] = out["harmonic_speed"].isna().astype(int)
    out["speed"] = out["speed"].fillna(0.0)
    out["harmonic_speed"] = out["harmonic_speed"].fillna(0.0)
    return out


def densify_timeseries(node_ts: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    all_times = np.sort(node_ts["begin"].unique())
    all_nodes = np.sort(node_ts["node_id"].unique())
    full_index = pd.MultiIndex.from_product([all_times, all_nodes], names=["begin", "node_id"])

    out = node_ts.set_index(["begin", "node_id"]).reindex(full_index).reset_index()
    step_seconds = pd.Series(all_times).diff().dropna()
    median_step = int(step_seconds.median()) if len(step_seconds) else 300

    out["end"] = out["end"].fillna(out["begin"] + median_step)
    fill_zero_cols = [
        "flow_count_5m",
        "flow_veh_h",
        "speed",
        "harmonic_speed",
        "e1_occupancy",
        "e2_meanOccupancy",
        "time_loss",
        "started_halts",
        "jam_veh",
        "vehicle_number_mean",
        "speed_is_missing",
        "harmonic_speed_is_missing",
    ]
    for col in fill_zero_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    return out, median_step


def split_joined_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(";") if item.strip())


def rebuild_window_labels(step_csv: Path, window_csv: Path, window_sec: int, mode: str) -> None:
    buckets: dict[int, dict[str, Any]] = {}
    with step_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = pd.read_csv(fh)
    required = {"step", "anomaly_mode", "is_anomaly", "active_event_ids", "active_edge_ids"}
    missing = required - set(reader.columns)
    if missing:
        raise ValueError(f"Step CSV missing columns: {sorted(missing)}")

    for _, row in reader.iterrows():
        step = int(float(row["step"]))
        window_start = (step // window_sec) * window_sec
        bucket = buckets.setdefault(
            window_start,
            {
                "step_count": 0,
                "anomaly_step_count": 0,
                "event_ids": set(),
                "anomaly_types": set(),
                "edge_ids": set(),
                "event_ids_by_class": {name: set() for name in ANOMALY_CLASS_NAMES},
                "edge_ids_by_class": {name: set() for name in ANOMALY_CLASS_NAMES},
                "class_step_counts": {name: 0 for name in ANOMALY_CLASS_NAMES},
            },
        )
        bucket["step_count"] += 1
        is_anomaly = int(float(row.get("is_anomaly", 0)))
        if is_anomaly:
            bucket["anomaly_step_count"] += 1
        for event_id in split_joined_ids(str(row.get("active_event_ids", ""))):
            bucket["event_ids"].add(event_id)
        for anomaly_type in split_joined_ids(str(row.get("active_anomaly_types", ""))):
            bucket["anomaly_types"].add(anomaly_type)
        for edge_id in split_joined_ids(str(row.get("active_edge_ids", ""))):
            bucket["edge_ids"].add(edge_id)
        for class_name in ANOMALY_CLASS_NAMES:
            for event_id in split_joined_ids(
                str(row.get(f"active_event_ids_{class_name}", ""))
            ):
                bucket["event_ids_by_class"][class_name].add(event_id)
            for edge_id in split_joined_ids(
                str(row.get(f"active_edge_ids_{class_name}", ""))
            ):
                bucket["edge_ids_by_class"][class_name].add(edge_id)
            bucket["class_step_counts"][class_name] += int(
                float(row.get(class_name, 0) or 0)
            )

    if not buckets:
        raise ValueError(f"Step CSV has no rows: {step_csv}")

    rows: list[dict[str, Any]] = []
    for window_start in sorted(buckets):
        bucket = buckets[window_start]
        step_count = int(bucket["step_count"])
        anomaly_step_count = int(bucket["anomaly_step_count"])
        coverage = (anomaly_step_count / step_count) if step_count > 0 else 0.0
        output_row = {
                "window_start": window_start,
                "window_end": window_start + window_sec,
                "step_count": step_count,
                "anomaly_step_count": anomaly_step_count,
                "anomaly_coverage_ratio": f"{coverage:.6f}",
                "anomaly_mode": mode,
                "is_anomaly_window": 1 if anomaly_step_count > 0 else 0,
                "active_event_ids": ";".join(sorted(bucket["event_ids"])),
                "active_anomaly_types": ";".join(sorted(bucket["anomaly_types"])),
                "active_edge_ids": ";".join(sorted(bucket["edge_ids"])),
        }
        for class_name in ANOMALY_CLASS_NAMES:
            output_row[f"active_event_ids_{class_name}"] = ";".join(
                sorted(bucket["event_ids_by_class"][class_name])
            )
            output_row[f"active_edge_ids_{class_name}"] = ";".join(
                sorted(bucket["edge_ids_by_class"][class_name])
            )
            output_row[class_name] = int(
                bucket["class_step_counts"][class_name] > 0
            )
        rows.append(output_row)

    window_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(window_csv, index=False)


def candidate_spawn_paths(scenarios_dir: Path, mode: str) -> list[Path]:
    out = [scenarios_dir / f"anomaly_spawn_records_{mode}.json"]
    out.append(scenarios_dir / f"anomaly_{mode}_spawn_records_{mode}.json")
    if mode == "accident":
        out.append(scenarios_dir / "anomaly_accident_spawn_records_accident.json")
    return out


def resolve_mode_paths(
    mode: str,
    detector_dir: Path,
    scenarios_dir: Path,
    strict: bool,
    window_sec: int,
) -> ModePaths:
    e1_xml = detector_dir / f"inductionloop_output_{mode}.xml"
    e2_xml = detector_dir / f"lanearea_output_{mode}.xml"
    step_labels = scenarios_dir / f"anomaly_step_labels_{mode}.csv"
    window_labels = scenarios_dir / f"anomaly_window_labels_{mode}.csv"

    required_missing = [path for path in [e1_xml, e2_xml, step_labels] if not path.exists()]
    if required_missing:
        raise FileNotFoundError(f"Missing required files for mode '{mode}': {[str(p) for p in required_missing]}")

    spawn_records = None
    for candidate in candidate_spawn_paths(scenarios_dir, mode):
        if candidate.exists():
            spawn_records = candidate
            break

    if mode != "baseline" and spawn_records is None:
        message = f"Spawn records file not found for mode '{mode}'."
        if strict:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}")

    if not window_labels.exists():
        print(f"[WARN] Missing {window_labels}. Rebuilding from step labels...")
        rebuild_window_labels(step_labels, window_labels, window_sec=window_sec, mode=mode)

    return ModePaths(
        mode=mode,
        e1_xml=e1_xml,
        e2_xml=e2_xml,
        step_labels=step_labels,
        window_labels=window_labels,
        spawn_records=spawn_records,
    )


def build_tensor(node_df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.sort(node_df["begin"].unique())
    nodes = np.sort(node_df["node_id"].unique())
    time_to_idx = {t: i for i, t in enumerate(times)}
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    data = np.zeros((len(times), len(nodes), len(feature_cols)), dtype=np.float32)
    temp = node_df[["begin", "node_id", *feature_cols]].copy()
    for _, row in temp.iterrows():
        t = time_to_idx[row["begin"]]
        n = node_to_idx[row["node_id"]]
        data[t, n, :] = row[feature_cols].to_numpy(dtype=np.float32)
    return data, times, nodes


def build_node_metadata(merged: pd.DataFrame, node_mode: str) -> pd.DataFrame:
    node_df = merged.assign(node_id=get_node_id(merged, node_mode))
    out = (
        node_df.groupby("node_id", as_index=False)
        .agg(
            road_key=("road_key", "first"),
            direction=("direction", lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else None),
            lane_idx=("lane_idx", lambda s: int(s.dropna().iloc[0]) if s.dropna().shape[0] else None),
            lane_count=("lane_idx", lambda s: int(s.dropna().nunique()) if s.dropna().shape[0] else 0),
        )
        .sort_values("node_id")
        .reset_index(drop=True)
    )
    return out


def make_sliding_windows(
    data: np.ndarray,
    input_steps: int,
    horizon_steps: int,
    target_feature_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    total_steps = data.shape[0]
    for start in range(total_steps - input_steps - horizon_steps + 1):
        x = data[start : start + input_steps]
        y = data[start + input_steps : start + input_steps + horizon_steps, :, target_feature_idx]
        x_list.append(x)
        y_list.append(y)
    if not x_list:
        raise ValueError(
            f"Not enough timesteps ({total_steps}) for input_steps={input_steps}, horizon_steps={horizon_steps}."
        )
    return np.stack(x_list).astype(np.float32), np.stack(y_list).astype(np.float32)


def load_edges_csv(edges_path: Path, nodes: np.ndarray) -> tuple[pd.DataFrame, list[str], list[str]]:
    node_list = [str(n) for n in nodes]
    node_set = set(node_list)

    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")

    edges_df = pd.read_csv(edges_path)
    required_cols = {"from", "to", "cost"}
    if not required_cols.issubset(edges_df.columns):
        raise ValueError(f"Edges file must include columns: {sorted(required_cols)}")

    for col in ["from", "to"]:
        edges_df[col] = edges_df[col].astype(str).str.strip()
        edges_df.loc[edges_df[col].isin(["", "nan", "None"]), col] = np.nan

    edges_df["cost"] = pd.to_numeric(edges_df["cost"], errors="coerce")
    edges_df = edges_df.dropna(subset=["from", "to"]).copy()
    if edges_df.shape[0] == 0:
        return edges_df, [], []

    edges_df["cost"] = edges_df["cost"].fillna(1.0).astype(float)
    edges_df = (
        edges_df.groupby(["from", "to"], as_index=False)
        .agg(cost=("cost", "min"))
        .sort_values(["from", "to"])
        .reset_index(drop=True)
    )
    unknown_from = sorted(set(edges_df["from"]) - node_set)
    unknown_to = sorted(set(edges_df["to"]) - node_set)
    return edges_df, unknown_from, unknown_to


def build_graph_arrays(nodes: np.ndarray, edges_df: pd.DataFrame) -> dict[str, Any]:
    node_list = [str(n) for n in nodes]
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    n_nodes = len(node_list)

    adj_binary_directed = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adj_cost_directed = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    edge_pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []

    for _, row in edges_df.iterrows():
        i = node_to_idx[str(row["from"])]
        j = node_to_idx[str(row["to"])]
        c = float(row["cost"])
        adj_binary_directed[i, j] = 1.0
        adj_cost_directed[i, j] = c
        edge_pairs.append((i, j))
        edge_weights.append(c)

    adj_binary_symmetric = np.maximum(adj_binary_directed, adj_binary_directed.T).astype(np.float32)
    adj_cost_symmetric = adj_cost_directed.copy()
    for i in range(n_nodes):
        for j in range(n_nodes):
            a = float(adj_cost_directed[i, j])
            b = float(adj_cost_directed[j, i])
            if a > 0 and b > 0:
                v = min(a, b)
            elif a > 0:
                v = a
            elif b > 0:
                v = b
            else:
                v = 0.0
            adj_cost_symmetric[i, j] = v

    if edge_pairs:
        edge_index = np.array(edge_pairs, dtype=np.int64).T
        edge_weight = np.array(edge_weights, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)

    node_index_df = pd.DataFrame({"node_id": node_list, "node_index": np.arange(n_nodes, dtype=int)})
    return {
        "node_index_df": node_index_df,
        "adj_binary_directed": adj_binary_directed,
        "adj_binary_symmetric": adj_binary_symmetric,
        "adj_cost_directed": adj_cost_directed,
        "adj_cost_symmetric": adj_cost_symmetric,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
    }


def export_graph_artifacts(output_dir: Path, edges_template: Path, nodes: np.ndarray) -> dict[str, Any]:
    edges_out = output_dir / "edges_template.csv"
    if edges_template.resolve() != edges_out.resolve():
        shutil.copy2(edges_template, edges_out)

    edges_df, unknown_from, unknown_to = load_edges_csv(edges_template, nodes)
    if unknown_from or unknown_to:
        raise ValueError(
            "Edges file contains node IDs not present in nodes.csv. "
            f"unknown_from={unknown_from}, unknown_to={unknown_to}"
        )

    if len(edges_df) == 0:
        print("[WARN] Edges CSV is empty. Skipping graph arrays export.")
        return {"edges_rows": 0, "num_edges_directed": 0}

    arrays = build_graph_arrays(nodes, edges_df)
    arrays["node_index_df"].to_csv(output_dir / "node_index.csv", index=False)
    np.save(output_dir / "adjacency_binary_directed.npy", arrays["adj_binary_directed"])
    np.save(output_dir / "adjacency_binary_symmetric.npy", arrays["adj_binary_symmetric"])
    np.save(output_dir / "adjacency_cost_directed.npy", arrays["adj_cost_directed"])
    np.save(output_dir / "adjacency_cost_symmetric.npy", arrays["adj_cost_symmetric"])
    np.save(output_dir / "edge_index.npy", arrays["edge_index"])
    np.save(output_dir / "edge_weight.npy", arrays["edge_weight"])
    np.savez_compressed(
        output_dir / "graph_dataset.npz",
        nodes=np.array(nodes, dtype=object),
        edge_index=arrays["edge_index"],
        edge_weight=arrays["edge_weight"],
        adjacency_binary_directed=arrays["adj_binary_directed"],
        adjacency_binary_symmetric=arrays["adj_binary_symmetric"],
        adjacency_cost_directed=arrays["adj_cost_directed"],
        adjacency_cost_symmetric=arrays["adj_cost_symmetric"],
    )
    return {
        "edges_rows": int(len(edges_df)),
        "num_edges_directed": int(arrays["edge_index"].shape[1]),
        "binary_directed_nonzero": int((arrays["adj_binary_directed"] > 0).sum()),
        "binary_symmetric_nonzero": int((arrays["adj_binary_symmetric"] > 0).sum()),
    }


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.global_mode not in modes and not args.skip_global_dataset:
        raise ValueError(f"--global-mode '{args.global_mode}' must be included in --modes.")
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be > 0.")
    if args.target_col not in DEFAULT_FEATURE_COLS and not args.skip_global_dataset:
        raise ValueError(f"--target-col '{args.target_col}' is not in supported features: {DEFAULT_FEATURE_COLS}")

    mode_summaries: dict[str, Any] = {}
    mode_timeseries: dict[str, pd.DataFrame] = {}
    mode_merged: dict[str, pd.DataFrame] = {}

    for mode in modes:
        paths = resolve_mode_paths(
            mode=mode,
            detector_dir=args.detector_dir,
            scenarios_dir=args.scenarios_dir,
            strict=args.strict,
            window_sec=args.window_sec,
        )
        print(f"[INFO] Processing mode '{mode}'...")
        e1_df = parse_e1_xml(paths.e1_xml)
        e2_df = parse_e2_xml(paths.e2_xml)
        merged = pd.merge(
            e1_df,
            e2_df,
            on=["begin", "end", "base_id", "road_key", "lane_idx", "direction"],
            how="outer",
            validate="one_to_one",
        )
        zero_fill_cols = [
            "nVehContrib_e1",
            "flow_veh_h_e1",
            "nVehEntered_e1",
            "sampledSeconds",
            "nVehEntered_e2",
            "nVehLeft_e2",
            "nVehSeen_e2",
            "meanMaxJamLengthInVehicles",
            "maxJamLengthInVehicles",
            "meanHaltingDuration",
            "startedHalts",
            "meanVehicleNumber",
            "maxVehicleNumber",
        ]
        for col in zero_fill_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        node_ts = aggregate_to_node(merged, node_mode=args.node_mode)
        node_ts_full, median_step = densify_timeseries(node_ts)
        if median_step != args.window_sec:
            message = (
                f"Mode '{mode}' has median step {median_step}s, expected {args.window_sec}s. "
                "Continuing with detected median step."
            )
            if args.strict:
                raise ValueError(message)
            print(f"[WARN] {message}")

        per_mode_csv = output_dir / f"{mode}_traffic_timeseries_long.csv"
        node_ts_full.to_csv(per_mode_csv, index=False)
        if mode == args.global_mode:
            node_ts_full.to_csv(output_dir / "traffic_timeseries_long.csv", index=False)

        mode_timeseries[mode] = node_ts_full
        mode_merged[mode] = merged

        label_copy_targets = [paths.step_labels, paths.window_labels]
        if paths.spawn_records is not None:
            label_copy_targets.append(paths.spawn_records)
        for source_path in label_copy_targets:
            shutil.copy2(source_path, output_dir / source_path.name)

        spawn_count = None
        if paths.spawn_records is not None:
            payload = json.loads(paths.spawn_records.read_text(encoding="utf-8"))
            records = payload.get("records", []) if isinstance(payload, dict) else []
            spawn_count = int(len(records)) if isinstance(records, list) else None

        mode_summaries[mode] = {
            "e1_xml": str(paths.e1_xml),
            "e2_xml": str(paths.e2_xml),
            "step_labels": str(paths.step_labels),
            "window_labels": str(paths.window_labels),
            "spawn_records": str(paths.spawn_records) if paths.spawn_records is not None else None,
            "spawn_record_count": spawn_count,
            "timeseries_csv": str(per_mode_csv),
            "time_steps": int(node_ts_full["begin"].nunique()),
            "nodes": int(node_ts_full["node_id"].nunique()),
            "rows": int(len(node_ts_full)),
            "median_step_sec": int(median_step),
        }
        print(
            f"[INFO] Mode '{mode}' exported: "
            f"timesteps={mode_summaries[mode]['time_steps']}, "
            f"nodes={mode_summaries[mode]['nodes']}, rows={mode_summaries[mode]['rows']}"
        )

    dataset_summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "modes": mode_summaries,
        "global_mode": args.global_mode,
        "node_mode": args.node_mode,
        "window_sec": int(args.window_sec),
        "features": list(DEFAULT_FEATURE_COLS),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if not args.skip_global_dataset:
        base_df = mode_timeseries[args.global_mode]
        base_merged = mode_merged[args.global_mode]

        data, times, nodes = build_tensor(base_df, DEFAULT_FEATURE_COLS)
        np.savez_compressed(
            output_dir / "traffic_dataset.npz",
            data=data.astype(np.float32),
            times=np.array(times, dtype=np.float64),
            nodes=np.array(nodes, dtype=object),
            features=np.array(DEFAULT_FEATURE_COLS, dtype=object),
        )

        node_meta = build_node_metadata(base_merged, node_mode=args.node_mode)
        node_meta.to_csv(output_dir / "node_metadata.csv", index=False)
        pd.DataFrame({"node_id": nodes}).to_csv(output_dir / "nodes.csv", index=False)

        t_len = int(len(times))
        train_end = int(t_len * 5 / 7)
        val_end = int(t_len * 6 / 7)
        splits = {"train": [0, train_end], "val": [train_end, val_end], "test": [val_end, t_len]}
        (output_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

        target_idx = DEFAULT_FEATURE_COLS.index(args.target_col)
        x_seq, y_seq = make_sliding_windows(data, args.input_steps, args.horizon_steps, target_feature_idx=target_idx)
        np.savez_compressed(
            output_dir / "traffic_sequence_dataset.npz",
            X=x_seq,
            Y=y_seq,
            feature_names=np.array(DEFAULT_FEATURE_COLS, dtype=object),
            target_col=np.array([args.target_col], dtype=object),
            input_steps=np.array([args.input_steps]),
            horizon_steps=np.array([args.horizon_steps]),
        )

        train_start, train_end = splits["train"]
        train_data = data[train_start:train_end]
        mean = train_data.mean(axis=(0, 1), keepdims=True)
        std = train_data.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        data_norm = (data - mean) / std
        np.savez_compressed(
            output_dir / "traffic_dataset_normalized.npz",
            data=data_norm.astype(np.float32),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            times=np.array(times, dtype=np.float64),
            nodes=np.array(nodes, dtype=object),
            features=np.array(DEFAULT_FEATURE_COLS, dtype=object),
        )

        graph_summary = None
        if not args.skip_graph_export:
            graph_summary = export_graph_artifacts(output_dir, args.edges_template, nodes=np.array(nodes, dtype=object))

        dataset_summary["global_dataset"] = {
            "dataset_npz": str(output_dir / "traffic_dataset.npz"),
            "dataset_norm_npz": str(output_dir / "traffic_dataset_normalized.npz"),
            "sequence_npz": str(output_dir / "traffic_sequence_dataset.npz"),
            "splits_json": str(output_dir / "splits.json"),
            "timeseries_long": str(output_dir / "traffic_timeseries_long.csv"),
            "shape": [int(v) for v in data.shape],
            "time_steps": int(data.shape[0]),
            "nodes": int(data.shape[1]),
            "features": int(data.shape[2]),
            "input_steps": int(args.input_steps),
            "horizon_steps": int(args.horizon_steps),
            "target_col": args.target_col,
            "splits": splits,
            "graph_summary": graph_summary,
        }

    summary_path = output_dir / "dataset_export_summary.json"
    summary_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    print(f"[DONE] Export completed. Summary: {summary_path}")


if __name__ == "__main__":
    main()
