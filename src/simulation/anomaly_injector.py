"""TraCI anomaly injector with edge-target behaviors."""

from __future__ import annotations

import random
from typing import Any

import traci
from traci import exceptions as traci_exceptions

from .anomaly_behaviors import accident, congestion_spike, sudden_stop, zigzag
from .anomaly_types import AccidentSpawnRecord, ActiveEvent, SpawnRecord


class AnomalyInjector:
    """Injects configured anomalies on target edges."""

    def __init__(self, valid_edge_ids: set[str], seed: int = 42) -> None:
        self.valid_edge_ids = set(valid_edge_ids)
        self.random = random.Random(seed)
        self.active_events: list[ActiveEvent] = []
        self.current_step = 0
        self.spawn_counter = 0
        self.event_counter = 0
        self._single_edge_route_cache: dict[str, str] = {}
        self.current_event_duration_hint: int | None = None
        self._accident_spawn_records: list[AccidentSpawnRecord] = []
        self._spawn_records: list[SpawnRecord] = []

    def step(self, current_step: int) -> None:
        """Update active events and clean finished ones."""

        self.current_step = current_step
        keep_events: list[ActiveEvent] = []
        for event in self.active_events:
            if event.kind == "accident":
                accident.update(self, event, current_step)
            elif event.kind == "sudden_stop":
                sudden_stop.update(self, event, current_step)
            elif event.kind == "congestion_spike":
                congestion_spike.update(self, event, current_step)
            elif event.kind == "zigzag":
                zigzag.update(self, event, current_step)

            if current_step >= event.end_step:
                self._finish_event(event)
            else:
                keep_events.append(event)
        self.active_events = keep_events

    def inject(
        self, kind: str, edge_id: str, params: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        """Inject one anomaly event by type at given edge."""

        if edge_id not in self.valid_edge_ids:
            return False, f"Unknown edge_id '{edge_id}' for anomaly injection."

        if kind == "accident":
            duration_hint = int((params or {}).get("duration_steps", accident.ACCIDENT_DURATION))
            self.current_event_duration_hint = max(1, duration_hint)
            try:
                return accident.inject(self, edge_id, params)
            finally:
                self.current_event_duration_hint = None
        if kind == "sudden_stop":
            return sudden_stop.inject(self, edge_id, params)
        if kind == "congestion_spike":
            return congestion_spike.inject(self, edge_id, params)
        if kind == "zigzag":
            return zigzag.inject(self, edge_id, params)
        return False, f"Unsupported anomaly kind '{kind}'."

    def event_count(self) -> int:
        return len(self.active_events)

    def get_active_events(self) -> tuple[ActiveEvent, ...]:
        return tuple(self.active_events)

    def get_accident_spawn_records(self) -> tuple[AccidentSpawnRecord, ...]:
        """Return successful accident spawn records for this run."""

        return tuple(self._accident_spawn_records)

    def get_spawn_records(self) -> tuple[SpawnRecord, ...]:
        """Return successful spawn/activation records for this run."""

        return tuple(self._spawn_records)

    def _finish_event(self, event: ActiveEvent) -> None:
        if event.kind == "accident":
            accident.finish(self, event)
            return
        if event.kind == "sudden_stop":
            sudden_stop.finish(self, event)
            return
        if event.kind == "zigzag":
            zigzag.finish(self, event)

    def _next_event_id(self, kind: str, edge_id: str) -> str:
        clean = edge_id.replace(":", "_").replace("#", "_").replace("-", "m")
        event_id = f"{kind}_{clean}_{self.current_step}_{self.event_counter}"
        self.event_counter += 1
        return event_id

    def _edge_lane_count(self, edge_id: str) -> int:
        """Return lane count for a non-internal edge."""

        try:
            return int(traci.edge.getLaneNumber(edge_id))
        except traci_exceptions.TraCIException:
            return 0

    def _spawn_accident_block(self, edge_id: str) -> tuple[str, int] | None:
        """
        Spawn one accident-blocking vehicle on one lane of the target edge.
        """

        try:
            lane_count = traci.edge.getLaneNumber(edge_id)
        except traci_exceptions.TraCIException:
            return None
        if lane_count < 1:
            return None

        route_id = self._ensure_single_edge_route(edge_id)
        lane_indices = list(range(int(lane_count)))
        self.random.shuffle(lane_indices)

        for lane_idx in lane_indices:

            try:
                lane_length = float(traci.lane.getLength(f"{edge_id}_{lane_idx}"))
            except traci_exceptions.TraCIException:
                continue

            for near_junction_at_end in (True, False):
                for stop_pos in accident.candidate_positions(
                    lane_length=lane_length,
                    near_junction_at_end=near_junction_at_end,
                ):
                    depart_pos = max(0.5, stop_pos - 1.5)
                    for _ in range(accident.ACCIDENT_SPAWN_RETRIES):
                        clean = edge_id.replace(":", "_").replace("#", "_").replace("-", "m")
                        vehicle_id = (
                            f"accident_{clean}_{self.current_step}_{self.spawn_counter}_l{lane_idx}"
                        )
                        self.spawn_counter += 1
                        if accident.spawn_vehicle_on_lane(
                            self,
                            vehicle_id=vehicle_id,
                            route_id=route_id,
                            edge_id=edge_id,
                            lane_index=lane_idx,
                            depart_pos=depart_pos,
                            stop_pos=stop_pos,
                        ):
                            return vehicle_id, lane_idx
                        if self._vehicle_exists(vehicle_id):
                            self._remove_vehicle(vehicle_id)

        return None

    def _record_accident_spawn(
        self,
        *,
        step: int,
        edge_id: str,
        lane_index: int,
        vehicle_id: str,
        event_id: str,
    ) -> None:
        """Record one successful accident spawn for downstream JSON labels."""

        self._accident_spawn_records.append(
            AccidentSpawnRecord(
                step=int(step),
                edge_id=str(edge_id),
                lane_index=int(lane_index),
                vehicle_id=str(vehicle_id),
                event_id=str(event_id),
            )
        )
        self._record_spawn(
            step=step,
            anomaly_mode="accident",
            edge_id=edge_id,
            event_id=event_id,
            vehicle_id=vehicle_id,
            lane_index=lane_index,
            activation_mode="spawned",
        )

    def _record_spawn(
        self,
        *,
        step: int,
        anomaly_mode: str,
        edge_id: str,
        event_id: str,
        vehicle_id: str | None = None,
        lane_index: int | None = None,
        activation_mode: str | None = None,
    ) -> None:
        """Record one successful anomaly activation/spawn."""

        lane: int | None
        if lane_index is None:
            lane = None
        else:
            lane = int(lane_index)
        self._spawn_records.append(
            SpawnRecord(
                step=int(step),
                anomaly_mode=str(anomaly_mode),
                edge_id=str(edge_id),
                event_id=str(event_id),
                vehicle_id=None if vehicle_id is None else str(vehicle_id),
                lane_index=lane,
                activation_mode=None if activation_mode is None else str(activation_mode),
            )
        )

    def _pick_random_vtype(self) -> str:
        try:
            type_ids = list(traci.vehicletype.getIDList())
        except traci_exceptions.TraCIException:
            type_ids = []
        preferred = [vt for vt in ("carFast", "motorbike", "bicycle", "bus") if vt in type_ids]
        if preferred:
            return self.random.choice(preferred)
        if type_ids:
            return self.random.choice(type_ids)
        return "DEFAULT_VEHTYPE"

    def _pick_random_vtype_for_lane(self, edge_id: str, lane_index: int) -> str | None:
        """Pick a vType whose vClass is allowed on the target lane."""

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
        if not type_ids:
            return None

        preferred = [vt for vt in ("carFast", "motorbike", "bicycle", "bus", "minibus") if vt in type_ids]
        ordered = preferred + [vt for vt in type_ids if vt not in preferred]
        candidates: list[str] = []

        for type_id in ordered:
            try:
                vclass = traci.vehicletype.getVehicleClass(type_id)
            except traci_exceptions.TraCIException:
                continue

            lane_ok = True
            # Empty allowed list means unrestricted by allow-list.
            if allowed:
                lane_ok = vclass in allowed
            if disallowed and vclass in disallowed:
                lane_ok = False
            if lane_ok:
                candidates.append(type_id)

        if candidates:
            return self.random.choice(candidates)
        return None

    def _remove_vehicle(self, vehicle_id: str) -> bool:
        if not vehicle_id or not self._vehicle_exists(vehicle_id):
            return False
        try:
            traci.vehicle.remove(vehicle_id)
            return True
        except traci_exceptions.TraCIException:
            return False

    def _release_vehicle(self, vehicle_id: str, *, restore_max_speed: bool = False) -> bool:
        """Release an anomaly-controlled vehicle without removing it at event boundary."""

        if not vehicle_id or not self._vehicle_exists(vehicle_id):
            return False

        if restore_max_speed:
            try:
                type_id = traci.vehicle.getTypeID(vehicle_id)
                max_speed = traci.vehicletype.getMaxSpeed(type_id)
                traci.vehicle.setMaxSpeed(vehicle_id, max_speed)
            except traci_exceptions.TraCIException:
                pass

        try:
            traci.vehicle.setSpeed(vehicle_id, -1)
            traci.vehicle.setColor(vehicle_id, (255, 255, 255, 255))
        except traci_exceptions.TraCIException:
            return False
        return True

    def _pick_vehicle_on_edge(self, edge_id: str, require_multilane: bool) -> str | None:
        candidates: list[tuple[float, str]] = []
        for vehicle_id in traci.vehicle.getIDList():
            try:
                road_id = traci.vehicle.getRoadID(vehicle_id)
                if road_id != edge_id:
                    continue
                if require_multilane and traci.edge.getLaneNumber(road_id) < 2:
                    continue
                lane_pos = float(traci.vehicle.getLanePosition(vehicle_id))
                candidates.append((-lane_pos, vehicle_id))
            except traci_exceptions.TraCIException:
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _ensure_single_edge_route(self, edge_id: str) -> str:
        existing = self._single_edge_route_cache.get(edge_id)
        if existing is not None:
            return existing

        clean_edge = edge_id.replace(":", "_").replace("#", "_").replace("-", "m")
        route_id = f"tmp_cong_{clean_edge}"
        try:
            traci.route.add(route_id, [edge_id])
        except traci_exceptions.TraCIException:
            pass
        self._single_edge_route_cache[edge_id] = route_id
        return route_id

    @staticmethod
    def _vehicle_exists(vehicle_id: str) -> bool:
        if not vehicle_id:
            return False
        return vehicle_id in traci.vehicle.getIDList()
