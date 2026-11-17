"""Config-driven anomaly scheduler and label export utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .anomaly_injector import AnomalyInjector
from .anomaly_types import (
    ANOMALY_BASELINE_MODE,
    ANOMALY_CLASS_NAMES,
    ANOMALY_MIXED_MODE,
    ANOMALY_SUPPORTED_MODES,
    ScheduledAnomalyEvent,
    StepLabelRow,
    WindowLabelRow,
    normalize_mode,
)


@dataclass(frozen=True)
class AnomalyRuntimeConfig:
    """Resolved anomaly runtime config for one simulation run."""

    mode: str
    schedule: tuple[ScheduledAnomalyEvent, ...]
    duration: dict[str, int | str] | None
    step_output_path: Path | None
    window_output_path: Path | None
    spawn_output_path: Path | None
    window_sec: int


class AnomalyScheduler:
    """Inject anomalies by schedule and emit labels for each simulation step."""

    def __init__(
        self,
        runtime: AnomalyRuntimeConfig,
        start_dt: datetime,
        injector: AnomalyInjector | None,
        valid_edge_ids: set[str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.start_dt = start_dt
        self.injector = injector
        self.valid_edge_ids = set(valid_edge_ids or [])
        self.step_rows: list[StepLabelRow] = []
        self._schedule_by_step: dict[int, list[ScheduledAnomalyEvent]] = {}
        for item in runtime.schedule:
            self._schedule_by_step.setdefault(item.step, []).append(item)

        known_edges = set(self.valid_edge_ids)
        if self.injector is not None:
            known_edges.update(self.injector.valid_edge_ids)
        if known_edges:
            missing = {
                event.edge_id
                for event in runtime.schedule
                if event.edge_id not in known_edges
            }
            if missing:
                raise ValueError(
                    "anomaly.schedule contains unknown edge_id: "
                    + ", ".join(sorted(missing))
                )

    @property
    def mode(self) -> str:
        return self.runtime.mode

    def on_step(self, sim_step: int, current_dt: datetime) -> list[str]:
        """Inject due events and append one step-level label row."""

        logs: list[str] = []
        if self.injector is not None:
            self.injector.step(sim_step)

        triggered_ids: list[str] = []
        succeeded_ids: list[str] = []
        failed_ids: list[str] = []
        failed_messages: list[str] = []

        for event in self._schedule_by_step.get(sim_step, []):
            triggered_ids.append(event.schedule_id)
            if self.runtime.mode == ANOMALY_BASELINE_MODE:
                message = (
                    f"[anomaly][skip] baseline mode ignores scheduled event "
                    f"{event.schedule_id} at step={sim_step}."
                )
                failed_ids.append(event.schedule_id)
                failed_messages.append(message)
                logs.append(message)
                continue
            if self.injector is None:
                message = (
                    f"[anomaly][fail] no injector available for event {event.schedule_id} "
                    f"at step={sim_step}."
                )
                failed_ids.append(event.schedule_id)
                failed_messages.append(message)
                logs.append(message)
                continue
            ok, message = self.injector.inject(
                kind=event.kind,
                edge_id=event.edge_id,
                params=event.params,
            )
            if ok:
                succeeded_ids.append(event.schedule_id)
                logs.append(
                    f"[anomaly][ok] id={event.schedule_id} step={sim_step} "
                    f"edge={event.edge_id} {message}"
                )
            else:
                failed_ids.append(event.schedule_id)
                failed_messages.append(message)
                logs.append(
                    f"[anomaly][fail] id={event.schedule_id} step={sim_step} "
                    f"edge={event.edge_id} {message}"
                )

        active_events = self.injector.get_active_events() if self.injector is not None else ()
        active_event_ids = ";".join(sorted(event.event_id for event in active_events))
        active_anomaly_types = ";".join(sorted({event.kind for event in active_events}))
        active_edge_ids = ";".join(sorted({event.edge_id for event in active_events}))
        active_type_set = {event.kind for event in active_events}
        event_ids_by_class = {
            class_name: ";".join(
                sorted(event.event_id for event in active_events if event.kind == class_name)
            )
            for class_name in ANOMALY_CLASS_NAMES
        }
        edge_ids_by_class = {
            class_name: ";".join(
                sorted({event.edge_id for event in active_events if event.kind == class_name})
            )
            for class_name in ANOMALY_CLASS_NAMES
        }
        row = StepLabelRow(
            step=sim_step,
            sim_datetime=current_dt,
            anomaly_mode=self.runtime.mode,
            is_anomaly=1 if active_events else 0,
            active_event_count=len(active_events),
            active_event_ids=active_event_ids,
            active_anomaly_types=active_anomaly_types,
            active_edge_ids=active_edge_ids,
            active_event_ids_accident=event_ids_by_class["accident"],
            active_event_ids_congestion_spike=event_ids_by_class["congestion_spike"],
            active_event_ids_zigzag=event_ids_by_class["zigzag"],
            active_event_ids_sudden_stop=event_ids_by_class["sudden_stop"],
            active_edge_ids_accident=edge_ids_by_class["accident"],
            active_edge_ids_congestion_spike=edge_ids_by_class["congestion_spike"],
            active_edge_ids_zigzag=edge_ids_by_class["zigzag"],
            active_edge_ids_sudden_stop=edge_ids_by_class["sudden_stop"],
            accident=int("accident" in active_type_set),
            congestion_spike=int("congestion_spike" in active_type_set),
            zigzag=int("zigzag" in active_type_set),
            sudden_stop=int("sudden_stop" in active_type_set),
            triggered_schedule_ids=";".join(triggered_ids),
            succeeded_schedule_ids=";".join(succeeded_ids),
            failed_schedule_ids=";".join(failed_ids),
            failed_messages=" | ".join(failed_messages),
        )
        self.step_rows.append(row)
        return logs

    def write_outputs(self) -> None:
        """Write step-level and window-level label files if configured."""

        if self.runtime.step_output_path is not None:
            self.runtime.step_output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.runtime.step_output_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "step",
                        "sim_datetime",
                        "anomaly_mode",
                        "is_anomaly",
                        "active_event_count",
                        "active_event_ids",
                        "active_anomaly_types",
                        "active_edge_ids",
                        *[f"active_event_ids_{name}" for name in ANOMALY_CLASS_NAMES],
                        *[f"active_edge_ids_{name}" for name in ANOMALY_CLASS_NAMES],
                        *ANOMALY_CLASS_NAMES,
                        "triggered_schedule_ids",
                        "succeeded_schedule_ids",
                        "failed_schedule_ids",
                        "failed_messages",
                    ],
                )
                writer.writeheader()
                for row in self.step_rows:
                    writer.writerow(
                        {
                            "step": row.step,
                            "sim_datetime": row.sim_datetime.isoformat(sep=" "),
                            "anomaly_mode": row.anomaly_mode,
                            "is_anomaly": row.is_anomaly,
                            "active_event_count": row.active_event_count,
                            "active_event_ids": row.active_event_ids,
                            "active_anomaly_types": row.active_anomaly_types,
                            "active_edge_ids": row.active_edge_ids,
                            "active_event_ids_accident": row.active_event_ids_accident,
                            "active_event_ids_congestion_spike": row.active_event_ids_congestion_spike,
                            "active_event_ids_zigzag": row.active_event_ids_zigzag,
                            "active_event_ids_sudden_stop": row.active_event_ids_sudden_stop,
                            "active_edge_ids_accident": row.active_edge_ids_accident,
                            "active_edge_ids_congestion_spike": row.active_edge_ids_congestion_spike,
                            "active_edge_ids_zigzag": row.active_edge_ids_zigzag,
                            "active_edge_ids_sudden_stop": row.active_edge_ids_sudden_stop,
                            "accident": row.accident,
                            "congestion_spike": row.congestion_spike,
                            "zigzag": row.zigzag,
                            "sudden_stop": row.sudden_stop,
                            "triggered_schedule_ids": row.triggered_schedule_ids,
                            "succeeded_schedule_ids": row.succeeded_schedule_ids,
                            "failed_schedule_ids": row.failed_schedule_ids,
                            "failed_messages": row.failed_messages,
                        }
                    )

        if self.runtime.window_output_path is not None:
            window_rows = build_window_labels(
                step_rows=self.step_rows,
                window_sec=self.runtime.window_sec,
                mode=self.runtime.mode,
            )
            self.runtime.window_output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.runtime.window_output_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "window_start",
                        "window_end",
                        "step_count",
                        "anomaly_step_count",
                        "anomaly_coverage_ratio",
                        "anomaly_mode",
                        "is_anomaly_window",
                        "active_event_ids",
                        "active_anomaly_types",
                        "active_edge_ids",
                        *[f"active_event_ids_{name}" for name in ANOMALY_CLASS_NAMES],
                        *[f"active_edge_ids_{name}" for name in ANOMALY_CLASS_NAMES],
                        *ANOMALY_CLASS_NAMES,
                    ],
                )
                writer.writeheader()
                for row in window_rows:
                    writer.writerow(
                        {
                            "window_start": row.window_start,
                            "window_end": row.window_end,
                            "step_count": row.step_count,
                            "anomaly_step_count": row.anomaly_step_count,
                            "anomaly_coverage_ratio": f"{row.anomaly_coverage_ratio:.6f}",
                            "anomaly_mode": row.anomaly_mode,
                            "is_anomaly_window": row.is_anomaly_window,
                            "active_event_ids": row.active_event_ids,
                            "active_anomaly_types": row.active_anomaly_types,
                            "active_edge_ids": row.active_edge_ids,
                            "active_event_ids_accident": row.active_event_ids_accident,
                            "active_event_ids_congestion_spike": row.active_event_ids_congestion_spike,
                            "active_event_ids_zigzag": row.active_event_ids_zigzag,
                            "active_event_ids_sudden_stop": row.active_event_ids_sudden_stop,
                            "active_edge_ids_accident": row.active_edge_ids_accident,
                            "active_edge_ids_congestion_spike": row.active_edge_ids_congestion_spike,
                            "active_edge_ids_zigzag": row.active_edge_ids_zigzag,
                            "active_edge_ids_sudden_stop": row.active_edge_ids_sudden_stop,
                            "accident": row.accident,
                            "congestion_spike": row.congestion_spike,
                            "zigzag": row.zigzag,
                            "sudden_stop": row.sudden_stop,
                        }
                    )

        if self.runtime.spawn_output_path is not None:
            records = ()
            if self.injector is not None:
                records = self.injector.get_spawn_records()
            self.runtime.spawn_output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "anomaly_mode": self.runtime.mode,
                "record_count": len(records),
                "records": [
                    {
                        "step": record.step,
                        "mode": record.anomaly_mode,
                        "schedule_id": None,
                        "activation_mode": record.activation_mode,
                        "lane": record.lane_index,
                        "lane_index": record.lane_index,
                        "edge_id": record.edge_id,
                        "vehicle_id": record.vehicle_id,
                        "event_id": record.event_id,
                    }
                    for record in records
                ],
            }
            with self.runtime.spawn_output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)


def parse_anomaly_runtime_config(
    raw_cfg: Any,
    *,
    config_dir: Path,
) -> AnomalyRuntimeConfig:
    """Parse anomaly runtime section from runner JSON config."""

    if raw_cfg is None:
        return AnomalyRuntimeConfig(
            mode=ANOMALY_BASELINE_MODE,
            schedule=(),
            duration=None,
            step_output_path=None,
            window_output_path=None,
            spawn_output_path=None,
            window_sec=300,
        )

    cfg = dict(raw_cfg)
    mode = normalize_mode(str(cfg.get("mode", ANOMALY_BASELINE_MODE)))
    if mode not in ANOMALY_SUPPORTED_MODES:
        raise ValueError(
            "anomaly.mode must be one of: " + ", ".join(sorted(ANOMALY_SUPPORTED_MODES))
        )

    raw_schedule = cfg.get("schedule", [])
    if not isinstance(raw_schedule, list):
        raise ValueError("anomaly.schedule must be a list")

    schedule: list[ScheduledAnomalyEvent] = []
    for idx, item_any in enumerate(raw_schedule):
        item = dict(item_any)
        step = int(item["step"])
        if step < 0:
            raise ValueError(f"anomaly.schedule[{idx}].step must be >= 0")
        edge_id = str(item.get("edge_id", "")).strip()
        if edge_id == "":
            legacy_junction = str(item.get("junction_id", "")).strip()
            if legacy_junction:
                raise ValueError(
                    f"anomaly.schedule[{idx}] now requires 'edge_id' (junction_id is no longer supported)."
                )
            raise ValueError(f"anomaly.schedule[{idx}].edge_id cannot be empty")
        params_any = item.get("params", {})
        if params_any is None:
            params_any = {}
        if not isinstance(params_any, dict):
            raise ValueError(f"anomaly.schedule[{idx}].params must be an object")
        event_kind = normalize_mode(str(item.get("kind", mode)))
        if event_kind not in ANOMALY_CLASS_NAMES:
            if mode == ANOMALY_MIXED_MODE:
                raise ValueError(
                    f"anomaly.schedule[{idx}].kind must be one of: "
                    + ", ".join(ANOMALY_CLASS_NAMES)
                )
            if mode != ANOMALY_BASELINE_MODE:
                raise ValueError(
                    f"anomaly.schedule[{idx}].kind '{event_kind}' is not supported."
                )
        schedule.append(
            ScheduledAnomalyEvent(
                schedule_id=str(item.get("id") or f"evt_{idx:04d}"),
                step=step,
                kind=event_kind,
                edge_id=edge_id,
                params=dict(params_any),
            )
        )
    schedule.sort(key=lambda item: (item.step, item.schedule_id))

    duration_cfg = _parse_duration_config(cfg.get("duration"))

    label_cfg = cfg.get("label", {})
    if label_cfg is None:
        label_cfg = {}
    if not isinstance(label_cfg, dict):
        raise ValueError("anomaly.label must be an object")

    default_base = (config_dir / "../../sumo/scenarios").resolve()
    default_step_name = f"anomaly_step_labels_{mode}.csv"
    default_window_name = f"anomaly_window_labels_{mode}.csv"
    step_output_path = _resolve_optional_path(
        config_dir,
        label_cfg.get("step_output_path", str(default_base / default_step_name)),
    )
    window_output_path = _resolve_optional_path(
        config_dir,
        label_cfg.get("window_output_path", str(default_base / default_window_name)),
    )
    default_spawn_name = f"anomaly_spawn_records_{mode}.json"
    spawn_output_raw = label_cfg.get("spawn_output_path")
    if spawn_output_raw is None:
        # Backward-compatible key.
        spawn_output_raw = label_cfg.get(
            "accident_spawn_output_path",
            str(default_base / default_spawn_name),
        )
    spawn_output_path = _resolve_optional_path(
        config_dir,
        spawn_output_raw,
    )

    window_sec = int(label_cfg.get("window_sec", 300))
    if window_sec <= 0:
        raise ValueError("anomaly.label.window_sec must be > 0")

    return AnomalyRuntimeConfig(
        mode=mode,
        schedule=tuple(schedule),
        duration=duration_cfg,
        step_output_path=step_output_path,
        window_output_path=window_output_path,
        spawn_output_path=spawn_output_path,
        window_sec=window_sec,
    )


def build_window_labels(
    step_rows: list[StepLabelRow],
    window_sec: int,
    mode: str,
) -> list[WindowLabelRow]:
    """Aggregate step labels into fixed windows for downstream datasets."""

    if window_sec <= 0:
        raise ValueError("window_sec must be > 0")

    if not step_rows:
        return []

    @dataclass
    class _Bucket:
        step_count: int = 0
        anomaly_step_count: int = 0
        event_ids: set[str] = None  # type: ignore[assignment]
        anomaly_types: set[str] = None  # type: ignore[assignment]
        edge_ids: set[str] = None  # type: ignore[assignment]
        event_ids_by_class: dict[str, set[str]] = None  # type: ignore[assignment]
        edge_ids_by_class: dict[str, set[str]] = None  # type: ignore[assignment]
        class_step_counts: dict[str, int] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            self.event_ids = set()
            self.anomaly_types = set()
            self.edge_ids = set()
            self.event_ids_by_class = {name: set() for name in ANOMALY_CLASS_NAMES}
            self.edge_ids_by_class = {name: set() for name in ANOMALY_CLASS_NAMES}
            self.class_step_counts = {name: 0 for name in ANOMALY_CLASS_NAMES}

    buckets: dict[int, _Bucket] = {}
    for row in step_rows:
        window_start = (row.step // window_sec) * window_sec
        bucket = buckets.setdefault(window_start, _Bucket())
        bucket.step_count += 1
        if row.is_anomaly:
            bucket.anomaly_step_count += 1
        for event_id in _split_joined_ids(row.active_event_ids):
            bucket.event_ids.add(event_id)
        for anomaly_type in _split_joined_ids(row.active_anomaly_types):
            bucket.anomaly_types.add(anomaly_type)
        for edge_id in _split_joined_ids(row.active_edge_ids):
            bucket.edge_ids.add(edge_id)
        for class_name in ANOMALY_CLASS_NAMES:
            for event_id in _split_joined_ids(
                getattr(row, f"active_event_ids_{class_name}")
            ):
                bucket.event_ids_by_class[class_name].add(event_id)
            for edge_id in _split_joined_ids(
                getattr(row, f"active_edge_ids_{class_name}")
            ):
                bucket.edge_ids_by_class[class_name].add(edge_id)
            bucket.class_step_counts[class_name] += int(getattr(row, class_name))

    out: list[WindowLabelRow] = []
    for window_start in sorted(buckets):
        bucket = buckets[window_start]
        anomaly_ratio = (
            bucket.anomaly_step_count / bucket.step_count if bucket.step_count > 0 else 0.0
        )
        out.append(
            WindowLabelRow(
                window_start=window_start,
                window_end=window_start + window_sec,
                step_count=bucket.step_count,
                anomaly_step_count=bucket.anomaly_step_count,
                anomaly_coverage_ratio=anomaly_ratio,
                anomaly_mode=mode,
                is_anomaly_window=1 if bucket.anomaly_step_count > 0 else 0,
                active_event_ids=";".join(sorted(bucket.event_ids)),
                active_anomaly_types=";".join(sorted(bucket.anomaly_types)),
                active_edge_ids=";".join(sorted(bucket.edge_ids)),
                active_event_ids_accident=";".join(
                    sorted(bucket.event_ids_by_class["accident"])
                ),
                active_event_ids_congestion_spike=";".join(
                    sorted(bucket.event_ids_by_class["congestion_spike"])
                ),
                active_event_ids_zigzag=";".join(
                    sorted(bucket.event_ids_by_class["zigzag"])
                ),
                active_event_ids_sudden_stop=";".join(
                    sorted(bucket.event_ids_by_class["sudden_stop"])
                ),
                active_edge_ids_accident=";".join(
                    sorted(bucket.edge_ids_by_class["accident"])
                ),
                active_edge_ids_congestion_spike=";".join(
                    sorted(bucket.edge_ids_by_class["congestion_spike"])
                ),
                active_edge_ids_zigzag=";".join(
                    sorted(bucket.edge_ids_by_class["zigzag"])
                ),
                active_edge_ids_sudden_stop=";".join(
                    sorted(bucket.edge_ids_by_class["sudden_stop"])
                ),
                accident=int(bucket.class_step_counts["accident"] > 0),
                congestion_spike=int(bucket.class_step_counts["congestion_spike"] > 0),
                zigzag=int(bucket.class_step_counts["zigzag"] > 0),
                sudden_stop=int(bucket.class_step_counts["sudden_stop"] > 0),
            )
        )
    return out


def _split_joined_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(";") if item.strip())


def _resolve_optional_path(config_dir: Path, raw: Any) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def _parse_duration_config(raw: Any) -> dict[str, int | str] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("anomaly.duration must be an object")

    policy = str(raw.get("policy", "")).strip().lower()
    if policy not in {"fixed", "random_range"}:
        raise ValueError("anomaly.duration.policy must be 'fixed' or 'random_range'")

    if policy == "fixed":
        steps = int(raw.get("steps", 0))
        if steps <= 0:
            raise ValueError("anomaly.duration.steps must be > 0 for fixed policy")
        return {"policy": "fixed", "steps": steps}

    min_steps = int(raw.get("min_steps", 0))
    max_steps = int(raw.get("max_steps", 0))
    if min_steps <= 0 or max_steps <= 0:
        raise ValueError("anomaly.duration.min_steps/max_steps must be > 0")
    if max_steps < min_steps:
        raise ValueError("anomaly.duration.max_steps must be >= min_steps")
    return {
        "policy": "random_range",
        "min_steps": min_steps,
        "max_steps": max_steps,
    }
