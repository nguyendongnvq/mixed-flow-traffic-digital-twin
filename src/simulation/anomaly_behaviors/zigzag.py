"""Zigzag anomaly behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import traci
from traci import exceptions as traci_exceptions

from ..anomaly_types import ActiveEvent

if TYPE_CHECKING:
    from ..anomaly_injector import AnomalyInjector


ZIGZAG_DURATION = 90
ZIGZAG_INTERVAL = 4
ZIGZAG_TARGET_SPEED_MPS = 5.0
ZIGZAG_SPAWN_MAX_ATTEMPTS_PER_STEP = 32


def inject(
    injector: "AnomalyInjector", edge_id: str, params: dict[str, Any] | None = None
) -> tuple[bool, str]:
    """Start zigzag lane switching for one spawned vehicle on target edge."""

    cfg = params or {}
    duration = max(1, int(cfg.get("duration_steps", ZIGZAG_DURATION)))
    interval = max(1, int(cfg.get("interval_steps", ZIGZAG_INTERVAL)))
    target_speed = max(0.1, float(cfg.get("target_speed_mps", ZIGZAG_TARGET_SPEED_MPS)))

    event = ActiveEvent(
        event_id=injector._next_event_id(kind="zigzag", edge_id=edge_id),
        kind="zigzag",
        edge_id=edge_id,
        start_step=injector.current_step,
        end_step=injector.current_step + 1,
        payload={
            "vehicle_id": "",
            "next_spawn_step": injector.current_step,
            "next_switch_step": injector.current_step + 1,
            "interval_steps": interval,
            "duration_steps": duration,
            "target_speed_mps": target_speed,
            "spawned_by_event": True,
            "spawned_once": False,
        },
    )
    injector.active_events.append(event)

    vehicle_id = _spawn_zigzag_vehicle(injector, event, injector.current_step)
    if vehicle_id is None:
        return (
            True,
            f"[zigzag] Event started on edge {edge_id}; pending vehicle spawn "
            f"(target_speed={target_speed:.1f} m/s). Will retry each step.",
        )

    event.payload["spawned_once"] = True
    event.end_step = injector.current_step + duration
    return (
        True,
        f"[zigzag] Spawned vehicle {vehicle_id} on edge {edge_id} at "
        f"{target_speed:.1f} m/s.",
    )


def update(injector: "AnomalyInjector", event: ActiveEvent, current_step: int) -> None:
    """Perform lane switching periodically while zigzag event is active."""

    duration = max(1, int(event.payload.get("duration_steps", ZIGZAG_DURATION)))
    next_spawn = int(event.payload.get("next_spawn_step", current_step))
    spawned_once = bool(event.payload.get("spawned_once", False))
    vehicle_id = str(event.payload.get("vehicle_id", ""))

    if not injector._vehicle_exists(vehicle_id):
        event.payload["vehicle_id"] = ""
        if current_step >= next_spawn:
            spawned_vehicle = _spawn_zigzag_vehicle(injector, event, current_step)
            if spawned_vehicle is not None:
                if not spawned_once:
                    event.payload["spawned_once"] = True
                    event.end_step = current_step + duration
                event.payload["next_switch_step"] = current_step + 1
            else:
                event.payload["next_spawn_step"] = current_step + 1
                if current_step >= event.end_step:
                    event.end_step = current_step + 1
        elif current_step >= event.end_step:
            event.end_step = current_step + 1
        return

    next_switch = int(event.payload.get("next_switch_step", current_step + 1))
    interval = max(1, int(event.payload.get("interval_steps", ZIGZAG_INTERVAL)))
    if current_step < next_switch or current_step >= event.end_step:
        return

    try:
        road_id = traci.vehicle.getRoadID(vehicle_id)
        if not road_id or road_id.startswith(":"):
            return
        lane_count = traci.edge.getLaneNumber(road_id)
        if lane_count < 2:
            return
        lane_index = traci.vehicle.getLaneIndex(vehicle_id)
        target_lane = lane_count - 1 if lane_index == 0 else 0
        traci.vehicle.changeLane(vehicle_id, target_lane, 2)
        event.payload["next_switch_step"] = current_step + interval
    except traci_exceptions.TraCIException:
        return


def finish(injector: "AnomalyInjector", event: ActiveEvent) -> None:
    """Release the zigzag vehicle when the event expires."""

    vehicle_id = str(event.payload.get("vehicle_id", ""))
    injector._release_vehicle(vehicle_id, restore_max_speed=True)


def _spawn_zigzag_vehicle(
    injector: "AnomalyInjector", event: ActiveEvent, current_step: int
) -> str | None:
    """Spawn one zigzag vehicle on target edge with low initial speed."""

    edge_id = event.edge_id
    target_speed = max(
        0.1,
        float(event.payload.get("target_speed_mps", ZIGZAG_TARGET_SPEED_MPS)),
    )
    route_id = injector._ensure_single_edge_route(edge_id)
    clean = edge_id.replace(":", "_").replace("#", "_").replace("-", "m")

    for _ in range(ZIGZAG_SPAWN_MAX_ATTEMPTS_PER_STEP):
        vehicle_id = f"zig_{clean}_{current_step}_{injector.spawn_counter}"
        injector.spawn_counter += 1
        try:
            traci.vehicle.add(
                vehicle_id,
                route_id,
                typeID="carFast",
                depart="now",
                departLane="best",
                departPos="random_free",
                departSpeed=f"{target_speed:.2f}",
            )
            traci.vehicle.setMaxSpeed(vehicle_id, target_speed)
            traci.vehicle.setSpeed(vehicle_id, target_speed)
            traci.vehicle.setColor(vehicle_id, (180, 0, 255, 255))
            event.payload["vehicle_id"] = vehicle_id
            event.payload["next_spawn_step"] = current_step + 1
            injector._record_spawn(
                step=current_step,
                anomaly_mode="zigzag",
                edge_id=edge_id,
                event_id=event.event_id,
                vehicle_id=vehicle_id,
                lane_index=None,
                activation_mode="spawned_vehicle",
            )
            return vehicle_id
        except traci_exceptions.TraCIException:
            continue
    return None
