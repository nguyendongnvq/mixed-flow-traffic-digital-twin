"""Congestion-spike anomaly behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import traci
from traci import exceptions as traci_exceptions

from ..anomaly_types import ActiveEvent

if TYPE_CHECKING:
    from ..anomaly_injector import AnomalyInjector


CONGESTION_DURATION = 900
CONGESTION_INTERVAL = 100
CONGESTION_WAVE_SIZE = 4
CONGESTION_TARGET_SPEED_MPS = 5.0
CONGESTION_MAX_ATTEMPTS_PER_STEP = 32


def inject(
    injector: "AnomalyInjector", edge_id: str, params: dict[str, Any] | None = None
) -> tuple[bool, str]:
    """Start a congestion wave generator on one edge."""

    cfg = params or {}
    duration = max(1, int(cfg.get("duration_steps", CONGESTION_DURATION)))
    interval = max(1, int(cfg.get("interval_steps", CONGESTION_INTERVAL)))
    wave_size = max(1, int(cfg.get("wave_size", CONGESTION_WAVE_SIZE)))
    target_speed = max(0.1, float(cfg.get("target_speed_mps", CONGESTION_TARGET_SPEED_MPS)))

    event = ActiveEvent(
        event_id=injector._next_event_id(kind="congestion_spike", edge_id=edge_id),
        kind="congestion_spike",
        edge_id=edge_id,
        start_step=injector.current_step,
        end_step=injector.current_step + duration,
        payload={
            "next_spawn_step": injector.current_step,
            "interval_steps": interval,
            "wave_size": wave_size,
            "duration_steps": duration,
            "target_speed_mps": target_speed,
            "pending_wave_remaining": wave_size,
        },
    )
    injector.active_events.append(event)
    spawned = spawn_wave(injector, event, injector.current_step)
    remaining = max(0, int(event.payload.get("pending_wave_remaining", 0)))
    event.payload["next_spawn_step"] = (
        injector.current_step + 1 if remaining > 0 else injector.current_step + interval
    )

    return (
        True,
        f"[congestion_spike] Started on edge {edge_id} for {duration} steps. "
        f"Spawned {spawned}/{wave_size} vehicles in first wave, "
        f"target_speed={target_speed:.1f} m/s.",
    )


def update(injector: "AnomalyInjector", event: ActiveEvent, current_step: int) -> None:
    """Emit periodic extra wave vehicles while congestion event is active."""

    next_spawn = int(event.payload.get("next_spawn_step", current_step + 1))
    interval = max(1, int(event.payload.get("interval_steps", CONGESTION_INTERVAL)))
    if current_step >= next_spawn and current_step < event.end_step:
        spawn_wave(injector, event, current_step)
        remaining = max(0, int(event.payload.get("pending_wave_remaining", 0)))
        event.payload["next_spawn_step"] = current_step + (1 if remaining > 0 else interval)


def spawn_wave(injector: "AnomalyInjector", event: ActiveEvent, step: int) -> int:
    """Spawn one congestion wave on the configured target edge."""

    wave_size = max(1, int(event.payload.get("wave_size", CONGESTION_WAVE_SIZE)))
    remaining = max(0, int(event.payload.get("pending_wave_remaining", wave_size)))
    if remaining == 0:
        remaining = wave_size
    target_speed = max(
        0.1,
        float(event.payload.get("target_speed_mps", CONGESTION_TARGET_SPEED_MPS)),
    )

    edge_id = event.edge_id
    spawned = 0
    attempts = 0
    while remaining > 0 and attempts < CONGESTION_MAX_ATTEMPTS_PER_STEP:
        attempts += 1
        route_id = injector._ensure_single_edge_route(edge_id)
        clean = edge_id.replace(":", "_").replace("#", "_").replace("-", "m")
        vehicle_id = f"cong_{clean}_{step}_{injector.spawn_counter}"
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
            traci.vehicle.setColor(vehicle_id, (255, 255, 0, 255))
            injector._record_spawn(
                step=step,
                anomaly_mode="congestion_spike",
                edge_id=edge_id,
                event_id=event.event_id,
                vehicle_id=vehicle_id,
                lane_index=None,
                activation_mode="wave_spawn",
            )
            spawned += 1
            remaining -= 1
        except traci_exceptions.TraCIException:
            continue
    event.payload["pending_wave_remaining"] = remaining
    return spawned
