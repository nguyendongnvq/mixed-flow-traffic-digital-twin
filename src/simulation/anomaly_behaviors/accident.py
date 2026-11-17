"""Accident anomaly behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import traci
from traci import exceptions as traci_exceptions

from ..anomaly_types import ActiveEvent

if TYPE_CHECKING:
    from ..anomaly_injector import AnomalyInjector


ACCIDENT_DURATION = 1800
ACCIDENT_SPAWN_RETRIES = 6


def inject(
    injector: "AnomalyInjector", edge_id: str, params: dict[str, Any] | None = None
) -> tuple[bool, str]:
    """Inject stopped accident blocker(s) on target edge."""

    cfg = params or {}
    duration = max(1, int(cfg.get("duration_steps", ACCIDENT_DURATION)))
    lane_count = injector._edge_lane_count(edge_id)

    spawn_result = injector._spawn_accident_block(edge_id=edge_id)
    if spawn_result is not None:
        vehicle_id, lane_index = spawn_result
        event = ActiveEvent(
            event_id=injector._next_event_id(kind="accident", edge_id=edge_id),
            kind="accident",
            edge_id=edge_id,
            start_step=injector.current_step,
            end_step=injector.current_step + duration,
            payload={
                "vehicle_ids": [vehicle_id],
                "lane_indices": [lane_index],
                "duration_steps": duration,
                "activation_mode": "spawned",
                "pending_setstop": False,
            },
        )
        injector.active_events.append(event)
        injector._record_accident_spawn(
            step=injector.current_step,
            edge_id=edge_id,
            lane_index=lane_index,
            vehicle_id=vehicle_id,
            event_id=event.event_id,
        )
        return (
            True,
            f"[accident] Spawned 1 stopped vehicle on edge {edge_id} "
            f"(lane_count={lane_count}, lane={lane_index}) for {duration} steps.",
        )

    # Fallback requested: no edge speed manipulation. Try setStop on existing vehicle.
    ok, fallback_msg, vehicle_id, lane_idx = _apply_setstop_on_existing_vehicle(
        injector=injector,
        edge_id=edge_id,
        duration_steps=duration,
    )
    if ok:
        event = ActiveEvent(
            event_id=injector._next_event_id(kind="accident", edge_id=edge_id),
            kind="accident",
            edge_id=edge_id,
            start_step=injector.current_step,
            end_step=injector.current_step + duration,
            payload={
                "vehicle_ids": [vehicle_id],
                "lane_indices": [lane_idx],
                "duration_steps": duration,
                "activation_mode": "setstop_existing",
                "pending_setstop": False,
            },
        )
        injector.active_events.append(event)
        injector._record_spawn(
            step=injector.current_step,
            anomaly_mode="accident",
            edge_id=edge_id,
            event_id=event.event_id,
            vehicle_id=vehicle_id,
            lane_index=lane_idx,
            activation_mode="setstop_existing",
        )
        return (
            True,
            f"[accident] Spawn failed; fallback setStop on existing vehicle {vehicle_id} "
            f"(edge={edge_id}, lane={lane_idx}) for {duration} steps.",
        )

    # Keep event active and retry setStop until success. This avoids schedule failure.
    event = ActiveEvent(
        event_id=injector._next_event_id(kind="accident", edge_id=edge_id),
        kind="accident",
        edge_id=edge_id,
        start_step=injector.current_step,
        end_step=injector.current_step + duration,
        payload={
            "vehicle_ids": [],
            "lane_indices": [],
            "duration_steps": duration,
            "activation_mode": "pending_setstop",
            "pending_setstop": True,
            "pending_reason": fallback_msg,
        },
    )
    injector.active_events.append(event)
    return (
        True,
        f"[accident] Spawn failed and no vehicle to setStop now on edge {edge_id}. "
        "Event scheduled in pending mode and will retry each step.",
    )


def update(injector: "AnomalyInjector", event: ActiveEvent, current_step: int) -> None:
    """Retry pending setStop activation for accident events."""

    if not bool(event.payload.get("pending_setstop", False)):
        return

    remaining = max(1, event.end_step - current_step)
    ok, message, vehicle_id, lane_idx = _apply_setstop_on_existing_vehicle(
        injector=injector,
        edge_id=event.edge_id,
        duration_steps=remaining,
    )
    if not ok:
        event.payload["pending_reason"] = message
        return

    event.payload["pending_setstop"] = False
    event.payload["activation_mode"] = "setstop_existing"
    event.payload["vehicle_ids"] = [vehicle_id]
    event.payload["lane_indices"] = [lane_idx]
    event.payload["activated_step"] = current_step
    injector._record_spawn(
        step=current_step,
        anomaly_mode="accident",
        edge_id=event.edge_id,
        event_id=event.event_id,
        vehicle_id=vehicle_id,
        lane_index=lane_idx,
        activation_mode="setstop_existing",
    )


def finish(injector: "AnomalyInjector", event: ActiveEvent) -> None:
    """Cleanup accident effect when event expires."""

    mode = str(event.payload.get("activation_mode", ""))
    vehicle_ids_any = event.payload.get("vehicle_ids", [])
    vehicle_ids = [str(item) for item in vehicle_ids_any] if isinstance(vehicle_ids_any, list) else []

    if mode == "spawned":
        for vehicle_id in vehicle_ids:
            injector._release_vehicle(vehicle_id)
        return

    for vehicle_id in vehicle_ids:
        injector._release_vehicle(vehicle_id)


def _apply_setstop_on_existing_vehicle(
    injector: "AnomalyInjector",
    edge_id: str,
    duration_steps: int,
) -> tuple[bool, str, str, int]:
    """Apply setStop to one existing vehicle on the target edge."""

    vehicle_id = injector._pick_vehicle_on_edge(edge_id, require_multilane=False)
    if vehicle_id is None:
        return False, f"No vehicle available on edge {edge_id}.", "", -1

    try:
        lane_idx = int(traci.vehicle.getLaneIndex(vehicle_id))
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_pos = float(traci.vehicle.getLanePosition(vehicle_id))
        lane_len = float(traci.lane.getLength(lane_id))

        target_pos = lane_pos + 0.5
        target_pos = max(0.5, target_pos)
        target_pos = min(target_pos, max(1.0, lane_len - 0.5))

        traci.vehicle.setStop(
            vehicle_id,
            edge_id,
            pos=target_pos,
            laneIndex=lane_idx,
            duration=max(1, int(duration_steps)),
        )
        traci.vehicle.setColor(vehicle_id, (170, 0, 0, 255))
        return (
            True,
            f"setStop applied to {vehicle_id} on edge {edge_id} lane {lane_idx}.",
            vehicle_id,
            lane_idx,
        )
    except traci_exceptions.TraCIException as exc:
        return False, f"setStop failed on existing vehicle {vehicle_id}: {exc}", "", -1


def spawn_vehicle_on_lane(
    injector: "AnomalyInjector",
    vehicle_id: str,
    route_id: str,
    edge_id: str,
    lane_index: int,
    depart_pos: float,
    stop_pos: float,
) -> bool:
    """Spawn one stopped accident vehicle at target lane."""

    try:
        traci_type = injector._pick_random_vtype_for_lane(edge_id=edge_id, lane_index=lane_index)
        if traci_type is None:
            return False
        traci.vehicle.add(
            vehicle_id,
            route_id,
            typeID=traci_type,
            depart="now",
            departLane=str(lane_index),
            departPos=f"{depart_pos:.2f}",
            departSpeed="0",
        )
        traci.vehicle.setStop(
            vehicle_id,
            edge_id,
            pos=stop_pos,
            laneIndex=lane_index,
            duration=max(1, int(injector.current_event_duration_hint or ACCIDENT_DURATION)),
        )
        traci.vehicle.setColor(vehicle_id, (170, 0, 0, 255))
        return True
    except traci_exceptions.TraCIException:
        injector._remove_vehicle(vehicle_id)
        return False


def candidate_positions(lane_length: float, near_junction_at_end: bool) -> tuple[float, ...]:
    """Candidate stop positions for accident vehicle spawning."""

    if lane_length <= 3.0:
        return ()
    offsets = (8.0, 14.0, 22.0, 30.0)
    max_pos = max(2.0, lane_length - 1.0)
    positions: list[float] = []
    for offset in offsets:
        raw_pos = lane_length - offset if near_junction_at_end else offset
        pos = min(max(2.0, raw_pos), max_pos)
        rounded = round(pos, 2)
        if rounded not in positions:
            positions.append(rounded)
    return tuple(positions)
