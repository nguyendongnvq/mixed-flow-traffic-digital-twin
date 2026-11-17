"""Sudden-stop anomaly behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import traci
from traci import exceptions as traci_exceptions

from ..anomaly_types import ActiveEvent

if TYPE_CHECKING:
    from ..anomaly_injector import AnomalyInjector


SUDDEN_STOP_DURATION = 1800
SUDDEN_STOP_COMPACT_VTYPE_PRIORITY = ("motorbike", "bicycle", "carFast")


def inject(
    injector: "AnomalyInjector", edge_id: str, params: dict[str, Any] | None = None
) -> tuple[bool, str]:
    """Force one nearby vehicle to stop; fallback to spawned stop-point."""

    cfg = params or {}
    duration = max(1, int(cfg.get("duration_steps", SUDDEN_STOP_DURATION)))
    vehicle_id = injector._pick_vehicle_on_edge(edge_id, require_multilane=False)
    if vehicle_id is not None:
        ok, message = _apply_stop_on_vehicle(vehicle_id)
        if ok:
            event = ActiveEvent(
                event_id=injector._next_event_id(kind="sudden_stop", edge_id=edge_id),
                kind="sudden_stop",
                edge_id=edge_id,
                start_step=injector.current_step,
                end_step=injector.current_step + duration,
                payload={
                    "vehicle_id": vehicle_id,
                    "duration_steps": duration,
                    "activation_mode": "set_speed_zero",
                    "pending_spawn": False,
                },
            )
            injector.active_events.append(event)
            injector._record_spawn(
                step=injector.current_step,
                anomaly_mode="sudden_stop",
                edge_id=edge_id,
                event_id=event.event_id,
                vehicle_id=vehicle_id,
                lane_index=None,
                activation_mode="set_speed_zero",
            )
            return (
                True,
                f"[sudden_stop] Vehicle {vehicle_id} forced to stop on edge {edge_id} "
                f"for {duration} steps.",
            )

        # Existing vehicle is present but could not be forced to stop;
        # fallback to spawned stop-point path below.
        spawn_ok, spawn_msg, spawned_vehicle_id, lane_idx = _spawn_stop_point(
            injector=injector,
            edge_id=edge_id,
            duration_steps=duration,
        )
        if spawn_ok:
            event = ActiveEvent(
                event_id=injector._next_event_id(kind="sudden_stop", edge_id=edge_id),
                kind="sudden_stop",
                edge_id=edge_id,
                start_step=injector.current_step,
                end_step=injector.current_step + duration,
                payload={
                    "vehicle_id": spawned_vehicle_id,
                    "duration_steps": duration,
                    "activation_mode": "spawn_stop_point",
                    "pending_spawn": False,
                    "lane_index": lane_idx,
                },
            )
            injector.active_events.append(event)
            injector._record_spawn(
                step=injector.current_step,
                anomaly_mode="sudden_stop",
                edge_id=edge_id,
                event_id=event.event_id,
                vehicle_id=spawned_vehicle_id,
                lane_index=lane_idx,
                activation_mode="spawn_stop_point",
            )
            return True, spawn_msg

        event = ActiveEvent(
            event_id=injector._next_event_id(kind="sudden_stop", edge_id=edge_id),
            kind="sudden_stop",
            edge_id=edge_id,
            start_step=injector.current_step,
            end_step=injector.current_step + duration,
            payload={
                "vehicle_id": "",
                "duration_steps": duration,
                "activation_mode": "pending_spawn",
                "pending_spawn": True,
                "pending_reason": f"{message} | {spawn_msg}",
            },
        )
        injector.active_events.append(event)
        return (
            True,
            f"[sudden_stop] Existing vehicle stop failed; fallback spawn pending on {edge_id}. "
            "Will retry each step until event end.",
        )

    spawn_ok, spawn_msg, spawned_vehicle_id, lane_idx = _spawn_stop_point(
        injector=injector,
        edge_id=edge_id,
        duration_steps=duration,
    )
    if spawn_ok:
        event = ActiveEvent(
            event_id=injector._next_event_id(kind="sudden_stop", edge_id=edge_id),
            kind="sudden_stop",
            edge_id=edge_id,
            start_step=injector.current_step,
            end_step=injector.current_step + duration,
            payload={
                "vehicle_id": spawned_vehicle_id,
                "duration_steps": duration,
                "activation_mode": "spawn_stop_point",
                "pending_spawn": False,
                "lane_index": lane_idx,
            },
        )
        injector.active_events.append(event)
        injector._record_spawn(
            step=injector.current_step,
            anomaly_mode="sudden_stop",
            edge_id=edge_id,
            event_id=event.event_id,
            vehicle_id=spawned_vehicle_id,
            lane_index=lane_idx,
            activation_mode="spawn_stop_point",
        )
        return True, spawn_msg

    event = ActiveEvent(
        event_id=injector._next_event_id(kind="sudden_stop", edge_id=edge_id),
        kind="sudden_stop",
        edge_id=edge_id,
        start_step=injector.current_step,
        end_step=injector.current_step + duration,
        payload={
            "vehicle_id": "",
            "duration_steps": duration,
            "activation_mode": "pending_spawn",
            "pending_spawn": True,
            "pending_reason": spawn_msg,
        },
    )
    injector.active_events.append(event)
    return (
        True,
        f"[sudden_stop] No vehicle now on edge {edge_id}; fallback spawn pending. "
        "Will retry each step until event end.",
    )


def update(injector: "AnomalyInjector", event: ActiveEvent, current_step: int) -> None:
    """Retry pending stop-point spawn if first attempt could not activate."""

    if not bool(event.payload.get("pending_spawn", False)):
        return

    remaining = max(1, event.end_step - current_step)
    spawn_ok, spawn_msg, vehicle_id, lane_idx = _spawn_stop_point(
        injector=injector,
        edge_id=event.edge_id,
        duration_steps=remaining,
    )
    if not spawn_ok:
        event.payload["pending_reason"] = spawn_msg
        return

    event.payload["pending_spawn"] = False
    event.payload["activation_mode"] = "spawn_stop_point"
    event.payload["vehicle_id"] = vehicle_id
    event.payload["lane_index"] = lane_idx
    event.payload["activated_step"] = current_step
    injector._record_spawn(
        step=current_step,
        anomaly_mode="sudden_stop",
        edge_id=event.edge_id,
        event_id=event.event_id,
        vehicle_id=vehicle_id,
        lane_index=lane_idx,
        activation_mode="spawn_stop_point",
    )


def finish(injector: "AnomalyInjector", event: ActiveEvent) -> None:
    """Reset the impacted vehicle speed/color when event expires."""

    activation_mode = str(event.payload.get("activation_mode", ""))
    vehicle_id = str(event.payload.get("vehicle_id", ""))
    if activation_mode == "spawn_stop_point":
        injector._release_vehicle(vehicle_id)
        return

    injector._release_vehicle(vehicle_id)


def _apply_stop_on_vehicle(vehicle_id: str) -> tuple[bool, str]:
    try:
        traci.vehicle.setSpeed(vehicle_id, 0.0)
        traci.vehicle.setColor(vehicle_id, (255, 128, 0, 255))
    except traci_exceptions.TraCIException as exc:
        return False, f"[sudden_stop] Failed to force stop vehicle {vehicle_id}: {exc}"
    return True, f"[sudden_stop] Vehicle {vehicle_id} forced to stop."


def _spawn_stop_point(
    injector: "AnomalyInjector",
    *,
    edge_id: str,
    duration_steps: int,
) -> tuple[bool, str, str, int]:
    lane_count = injector._edge_lane_count(edge_id)
    if lane_count < 1:
        return False, f"[sudden_stop] Edge {edge_id} has no lane for stop-point.", "", -1

    route_id = injector._ensure_single_edge_route(edge_id)
    lane_indices = list(range(lane_count))
    injector.random.shuffle(lane_indices)

    for lane_idx in lane_indices:
        lane_id = f"{edge_id}_{lane_idx}"
        try:
            lane_length = float(traci.lane.getLength(lane_id))
        except traci_exceptions.TraCIException:
            continue

        for stop_pos in _candidate_stop_positions(lane_length):
            depart_pos = max(0.5, stop_pos - 1.5)
            clean = edge_id.replace(":", "_").replace("#", "_").replace("-", "m")
            vehicle_id = (
                f"sstop_{clean}_{injector.current_step}_{injector.spawn_counter}_l{lane_idx}"
            )
            injector.spawn_counter += 1
            type_id = _pick_compact_vtype_for_lane(
                injector=injector,
                edge_id=edge_id,
                lane_index=lane_idx,
            )

            try:
                traci.vehicle.add(
                    vehicle_id,
                    route_id,
                    typeID=type_id,
                    depart="now",
                    departLane=str(lane_idx),
                    departPos=f"{depart_pos:.2f}",
                    departSpeed="max",
                )
                traci.vehicle.setStop(
                    vehicle_id,
                    edge_id,
                    pos=max(0.5, float(stop_pos)),
                    laneIndex=int(lane_idx),
                    duration=max(1, int(duration_steps)),
                )
                traci.vehicle.setSpeed(vehicle_id, 0.0)
                traci.vehicle.setColor(vehicle_id, (255, 128, 0, 255))
                return (
                    True,
                    f"[sudden_stop] Spawned stop-point vehicle {vehicle_id} on edge {edge_id} "
                    f"lane={lane_idx} for {duration_steps} steps.",
                    vehicle_id,
                    lane_idx,
                )
            except traci_exceptions.TraCIException:
                if injector._vehicle_exists(vehicle_id):
                    injector._remove_vehicle(vehicle_id)
                continue

    return (
        False,
        f"[sudden_stop] Could not spawn stop-point on edge {edge_id} now.",
        "",
        -1,
    )


def _candidate_stop_positions(lane_length: float) -> tuple[float, ...]:
    usable = max(2.0, float(lane_length))
    candidates = (
        max(1.0, usable * 0.8),
        max(1.0, usable * 0.6),
        max(1.0, usable * 0.4),
    )
    cap = max(1.0, usable - 2.0)
    return tuple(min(pos, cap) for pos in candidates)


def _pick_compact_vtype_for_lane(
    injector: "AnomalyInjector",
    *,
    edge_id: str,
    lane_index: int,
) -> str:
    lane_id = f"{edge_id}_{lane_index}"
    try:
        allowed = set(traci.lane.getAllowed(lane_id))
        disallowed = set(traci.lane.getDisallowed(lane_id))
    except traci_exceptions.TraCIException:
        allowed = set()
        disallowed = set()

    try:
        type_ids = list(traci.vehicletype.getIDList())
    except traci_exceptions.TraCIException:
        type_ids = []

    for type_id in SUDDEN_STOP_COMPACT_VTYPE_PRIORITY:
        if type_id not in type_ids:
            continue
        try:
            vclass = traci.vehicletype.getVehicleClass(type_id)
        except traci_exceptions.TraCIException:
            continue
        lane_ok = True
        if allowed:
            lane_ok = vclass in allowed
        if disallowed and vclass in disallowed:
            lane_ok = False
        if lane_ok:
            return type_id

    return injector._pick_random_vtype_for_lane(edge_id, lane_index) or injector._pick_random_vtype()
