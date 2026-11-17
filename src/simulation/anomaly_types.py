"""Shared anomaly data types for SUMO TraCI simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


ANOMALY_BASELINE_MODE = "baseline"
ANOMALY_MIXED_MODE = "mixed"
ANOMALY_CLASS_NAMES: tuple[str, ...] = (
    "accident",
    "congestion_spike",
    "zigzag",
    "sudden_stop",
)
ANOMALY_SUPPORTED_MODES: tuple[str, ...] = (
    ANOMALY_BASELINE_MODE,
    ANOMALY_MIXED_MODE,
    *ANOMALY_CLASS_NAMES,
)


@dataclass
class ActiveEvent:
    """One active anomaly instance in simulation."""

    event_id: str
    kind: str
    edge_id: str
    start_step: int
    end_step: int
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduledAnomalyEvent:
    """One scheduled anomaly trigger configured by user."""

    schedule_id: str
    step: int
    kind: str
    edge_id: str
    params: dict[str, Any]


@dataclass(frozen=True)
class AccidentSpawnRecord:
    """One successful accident spawn on a specific lane at a specific step."""

    step: int
    edge_id: str
    lane_index: int
    vehicle_id: str
    event_id: str


@dataclass(frozen=True)
class SpawnRecord:
    """One successful anomaly activation/spawn record."""

    step: int
    anomaly_mode: str
    edge_id: str
    event_id: str
    vehicle_id: str | None
    lane_index: int | None
    activation_mode: str | None


@dataclass(frozen=True)
class StepLabelRow:
    """Step-level label record exported after simulation."""

    step: int
    sim_datetime: datetime
    anomaly_mode: str
    is_anomaly: int
    active_event_count: int
    active_event_ids: str
    active_anomaly_types: str
    active_edge_ids: str
    active_event_ids_accident: str
    active_event_ids_congestion_spike: str
    active_event_ids_zigzag: str
    active_event_ids_sudden_stop: str
    active_edge_ids_accident: str
    active_edge_ids_congestion_spike: str
    active_edge_ids_zigzag: str
    active_edge_ids_sudden_stop: str
    accident: int
    congestion_spike: int
    zigzag: int
    sudden_stop: int
    triggered_schedule_ids: str
    succeeded_schedule_ids: str
    failed_schedule_ids: str
    failed_messages: str

@dataclass(frozen=True)
class WindowLabelRow:
    """Window-level anomaly label aggregate."""

    window_start: int
    window_end: int
    step_count: int
    anomaly_step_count: int
    anomaly_coverage_ratio: float
    anomaly_mode: str
    is_anomaly_window: int
    active_event_ids: str
    active_anomaly_types: str
    active_edge_ids: str
    active_event_ids_accident: str
    active_event_ids_congestion_spike: str
    active_event_ids_zigzag: str
    active_event_ids_sudden_stop: str
    active_edge_ids_accident: str
    active_edge_ids_congestion_spike: str
    active_edge_ids_zigzag: str
    active_edge_ids_sudden_stop: str
    accident: int
    congestion_spike: int
    zigzag: int
    sudden_stop: int


def normalize_mode(raw: str) -> str:
    """Normalize mode key for anomaly mode validation."""

    return str(raw).strip().lower()
