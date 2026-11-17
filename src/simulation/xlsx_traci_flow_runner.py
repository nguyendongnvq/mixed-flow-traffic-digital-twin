"""Run SUMO flows from Excel definitions via TraCI.

This module implements Phase 1/3 simulation setup and TraCI stepping from
pipeline.md. Traffic rates are read from SUMO-Simulate.xlsx and converted to
runtime vehicle insertion without hardcoded flow constants in source code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import traci
from traci import exceptions as traci_exceptions

from .anomaly_injector import AnomalyInjector
from .anomaly_scheduler import AnomalyScheduler, parse_anomaly_runtime_config
from .anomaly_types import ANOMALY_BASELINE_MODE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECONDS_PER_DAY = 24 * 60 * 60

NS_MAIN = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")
DATE_TIME_TEXT_RE = re.compile(
    r"(?P<h>\d{1,2}:\d{2}:\d{2}).*?(?P<d>\d{1,2}/\d{1,2}/\d{4})"
)
PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
FLOAT_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
HOLIDAY_DATE_RE = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})")


@dataclass(frozen=True)
class DaySlot:
    """One time interval in a day with per-profile/vehicle hourly rates."""

    start_second: int
    end_second: int
    rates: dict[str, dict[str, tuple[float, float]]]


@dataclass(frozen=True)
class RouteWeight:
    """Configured route target and allocation weight."""

    route_id: str
    weight: float
    depart_lane: str


@dataclass(frozen=True)
class RouteProfile:
    """Route mapping for one lane profile from Excel."""

    profile_id: str
    depart_lane: str
    routes: tuple[RouteWeight, ...]


@dataclass(frozen=True)
class StreamDef:
    """A concrete stream: profile + vehicle + route."""

    profile_id: str
    vehicle_key: str
    sumo_vtype: str
    route_id: str
    depart_lane: str
    weight_share: float


@dataclass(frozen=True)
class NetRoutingModel:
    """Lane permissions and edge-to-edge lane connections for route validation."""

    lane_permissions: dict[str, dict[int, tuple[set[str] | None, set[str]]]]
    connection_lane_pairs: dict[tuple[str, str], tuple[tuple[int, int], ...]]


class XlsxSheet:
    """Minimal XLSX sheet reader without external dependencies."""

    def __init__(self, xlsx_path: Path, sheet_name: str) -> None:
        self.xlsx_path = xlsx_path
        self.sheet_name = sheet_name
        self._cells: dict[tuple[int, str], str] = {}
        self._load()

    def get(self, cell_ref: str) -> str | None:
        col, row = split_cell_ref(cell_ref)
        return self._cells.get((row, col))

    def get_row_col(self, row: int, col: str) -> str | None:
        return self._cells.get((row, col.upper()))

    def _load(self) -> None:
        with zipfile.ZipFile(self.xlsx_path) as archive:
            shared = self._read_shared_strings(archive)
            sheet_target = self._resolve_sheet_target(archive, self.sheet_name)
            sheet_root = ET.fromstring(archive.read(sheet_target))
            sheet_data = sheet_root.find("main:sheetData", NS_MAIN)
            if sheet_data is None:
                return
            for row_node in sheet_data.findall("main:row", NS_MAIN):
                for cell_node in row_node.findall("main:c", NS_MAIN):
                    ref = cell_node.get("r")
                    if not ref:
                        continue
                    match = CELL_REF_RE.match(ref)
                    if not match:
                        continue
                    col = match.group(1)
                    row = int(match.group(2))
                    value = self._decode_cell(cell_node, shared)
                    if value is not None and value.strip() != "":
                        self._cells[(row, col)] = value

    @staticmethod
    def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
        path = "xl/sharedStrings.xml"
        if path not in archive.namelist():
            return []
        root = ET.fromstring(archive.read(path))
        out: list[str] = []
        for si in root.findall("main:si", NS_MAIN):
            text_node = si.find("main:t", NS_MAIN)
            if text_node is not None:
                out.append(text_node.text or "")
                continue
            parts: list[str] = []
            for run_node in si.findall("main:r", NS_MAIN):
                run_text = run_node.find("main:t", NS_MAIN)
                if run_text is not None:
                    parts.append(run_text.text or "")
            out.append("".join(parts))
        return out

    @staticmethod
    def _resolve_sheet_target(archive: zipfile.ZipFile, sheet_name: str) -> str:
        wb_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.get("Id"): rel.get("Target", "") for rel in rel_root}
        sheets = wb_root.find("main:sheets", NS_MAIN)
        if sheets is None:
            raise ValueError("workbook.xml does not contain sheets")
        for node in sheets.findall("main:sheet", NS_MAIN):
            if node.get("name") != sheet_name:
                continue
            rel_id = node.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            if not rel_id:
                break
            target = rel_map.get(rel_id)
            if not target:
                break
            if target.startswith("xl/"):
                return target
            return f"xl/{target}"
        raise ValueError(f"Sheet '{sheet_name}' not found in {archive.filename}")

    @staticmethod
    def _decode_cell(cell_node: ET.Element, shared: list[str]) -> str | None:
        cell_type = cell_node.get("t")
        if cell_type == "inlineStr":
            text_node = cell_node.find("main:is/main:t", NS_MAIN)
            return (text_node.text or "") if text_node is not None else None
        value_node = cell_node.find("main:v", NS_MAIN)
        if value_node is None:
            return None
        raw = value_node.text or ""
        if cell_type == "s":
            try:
                idx = int(raw)
            except ValueError:
                return raw
            return shared[idx] if 0 <= idx < len(shared) else raw
        return raw


class ExcelTrafficProfile:
    """Extracts simulation window, slots, and calendar modifiers from Excel."""

    def __init__(self, sheet: XlsxSheet, xlsx_cfg: dict[str, Any], rate_bound: str) -> None:
        self.sheet = sheet
        self.xlsx_cfg = xlsx_cfg
        self.rate_bound = rate_bound
        self.start_dt: datetime
        self.end_dt: datetime
        self.template_by_serial: dict[int, list[DaySlot]]
        self.weekend_factor: float
        self.holiday_factors: dict[date, float]
        self._parse()

    def pick_slots_for_date(self, sim_date: date) -> list[DaySlot]:
        serial = date_to_excel_serial(sim_date)
        if serial in self.template_by_serial:
            return self.template_by_serial[serial]
        if len(self.template_by_serial) == 1:
            return next(iter(self.template_by_serial.values()))
        raise KeyError(f"No Excel template rows for date serial {serial} ({sim_date.isoformat()})")

    def factor_for_date(self, sim_date: date) -> float:
        factor = 1.0
        if sim_date.weekday() >= 5:
            factor *= self.weekend_factor
        factor *= self.holiday_factors.get(sim_date, 1.0)
        return factor

    def _parse(self) -> None:
        self.start_dt, self.end_dt = self._parse_simulation_window()
        self.template_by_serial = self._parse_day_slots()
        self.weekend_factor = self._parse_weekend_factor()
        self.holiday_factors = self._parse_holiday_factors()

    def _parse_simulation_window(self) -> tuple[datetime, datetime]:
        start_ref = str(self.xlsx_cfg["start_datetime_cell"])
        end_ref = str(self.xlsx_cfg["end_datetime_cell"])
        start_raw = self.sheet.get(start_ref)
        end_raw = self.sheet.get(end_ref)
        if not start_raw or not end_raw:
            raise ValueError(f"Missing simulation window cells: {start_ref}, {end_ref}")
        start_dt = parse_datetime_from_text(start_raw)
        end_dt = parse_datetime_from_text(end_raw)
        if end_dt <= start_dt:
            raise ValueError(f"Excel simulation end <= start ({start_dt} -> {end_dt})")
        return start_dt, end_dt

    def _parse_day_slots(self) -> dict[int, list[DaySlot]]:
        table_cfg = dict(self.xlsx_cfg["table"])
        start_row = int(table_cfg["start_row"])
        end_row = int(table_cfg["end_row"])
        date_col = str(table_cfg["date_col"]).upper()
        start_col = str(table_cfg["start_time_col"]).upper()
        end_col = str(table_cfg["end_time_col"]).upper()
        column_map = dict(table_cfg["column_map"])

        grouped: dict[int, list[DaySlot]] = {}
        for row in range(start_row, end_row + 1):
            date_raw = self.sheet.get_row_col(row, date_col)
            start_raw = self.sheet.get_row_col(row, start_col)
            end_raw = self.sheet.get_row_col(row, end_col)
            if not date_raw and not start_raw and not end_raw:
                continue

            if date_raw is None or start_raw is None or end_raw is None:
                continue
            try:
                date_serial = int(round(parse_float(date_raw)))
                start_second = int(round(parse_float(start_raw) * SECONDS_PER_DAY))
                end_second = int(round(parse_float(end_raw) * SECONDS_PER_DAY))
            except ValueError:
                continue
            start_second = max(0, min(start_second, SECONDS_PER_DAY))
            end_second = max(0, min(end_second, SECONDS_PER_DAY))
            if end_second <= start_second:
                continue

            rates: dict[str, dict[str, tuple[float, float]]] = {}
            for profile_id, vehicle_map_any in column_map.items():
                vehicle_map = dict(vehicle_map_any)
                rates[profile_id] = {}
                for vehicle_key, bound_cols_any in vehicle_map.items():
                    bound_cols = dict(bound_cols_any)
                    min_col = str(bound_cols["min"]).upper()
                    max_col = str(bound_cols["max"]).upper()
                    min_raw = self.sheet.get_row_col(row, min_col)
                    max_raw = self.sheet.get_row_col(row, max_col)
                    if min_raw is None or max_raw is None:
                        raise ValueError(
                            f"Missing Excel value row={row} profile={profile_id} "
                            f"vehicle={vehicle_key} min={min_col} max={max_col}"
                        )
                    min_rate = parse_float(min_raw)
                    max_rate = parse_float(max_raw)
                    rates[profile_id][normalize_key(vehicle_key)] = (min_rate, max_rate)

            grouped.setdefault(date_serial, []).append(
                DaySlot(
                    start_second=start_second,
                    end_second=end_second,
                    rates=rates,
                )
            )

        if not grouped:
            raise ValueError("No day-slot rows parsed from Excel table")
        for serial, slots in grouped.items():
            slots.sort(key=lambda s: (s.start_second, s.end_second))
            grouped[serial] = tuple(slots)  # type: ignore[assignment]
        return grouped

    def _parse_weekend_factor(self) -> float:
        cal_cfg = dict(self.xlsx_cfg["calendar"])
        weekend_cell = str(cal_cfg["weekend_factor_cell"])
        raw = self.sheet.get(weekend_cell)
        if not raw:
            raise ValueError(f"Missing weekend factor cell: {weekend_cell}")
        match = PERCENT_RE.search(raw)
        if not match:
            raise ValueError(f"Cannot parse weekend percent from cell {weekend_cell}: {raw}")
        value = parse_float(match.group(1))
        return value / 100.0

    def _parse_holiday_factors(self) -> dict[date, float]:
        cal_cfg = dict(self.xlsx_cfg["calendar"])
        holiday_rows = cal_cfg.get("holiday_rows", [])
        out: dict[date, float] = {}
        if not isinstance(holiday_rows, list):
            raise ValueError("calendar.holiday_rows must be a list")

        for row_cfg_any in holiday_rows:
            row_cfg = dict(row_cfg_any)
            date_cell = str(row_cfg["date_cell"])
            factor_cell = str(row_cfg["factor_cell"])
            date_raw = self.sheet.get(date_cell)
            factor_raw = self.sheet.get(factor_cell)
            if not date_raw or not factor_raw:
                raise ValueError(
                    f"Missing holiday cells date={date_cell} factor={factor_cell}"
                )
            date_match = HOLIDAY_DATE_RE.search(date_raw)
            if not date_match:
                raise ValueError(f"Cannot parse holiday day/month from {date_cell}: {date_raw}")
            day = int(date_match.group(1))
            month = int(date_match.group(2))
            factor_match = FLOAT_RE.search(factor_raw)
            if not factor_match:
                raise ValueError(f"Cannot parse holiday factor from {factor_cell}: {factor_raw}")
            factor = parse_float(factor_match.group(1))

            holiday_date = date(self.start_dt.year, month, day)
            out[holiday_date] = factor
        return out


class TraciFlowRunner:
    """Runs live TraCI generation based on parsed Excel slots."""

    def __init__(
        self,
        profile: ExcelTrafficProfile,
        streams: tuple[StreamDef, ...],
        rate_bound: str,
        traffic_scale: float,
        flow_mode: str,
        sumocfg_path: Path,
        runtime_route_path: Path,
        use_gui: bool,
        step_length: float,
        seed: int,
        extra_sumo_args: list[str],
        max_seconds: int | None = None,
        anomaly_scheduler: AnomalyScheduler | None = None,
    ) -> None:
        self.profile = profile
        self.streams = streams
        self.rate_bound = rate_bound
        self.traffic_scale = traffic_scale
        self.flow_mode = flow_mode
        self.sumocfg_path = sumocfg_path
        self.runtime_route_path = runtime_route_path
        self.use_gui = use_gui
        self.step_length = step_length
        self.seed = seed
        self.extra_sumo_args = extra_sumo_args
        self.max_seconds = max_seconds
        self.anomaly_scheduler = anomaly_scheduler
        self.random = random.Random(seed)
        self.stream_carry: dict[tuple[str, str, str], float] = {
            (s.profile_id, s.vehicle_key, s.route_id): 0.0 for s in streams
        }
        self.vehicle_counter = 0
        self.hourly_spawn_by_route: dict[int, dict[str, int]] = {}
        self.last_logged_hour = -1
        self.successful_spawn_total = 0
        self.constant_base_rates = self._build_constant_base_rates()

    def run(self) -> None:
        binary = resolve_sumo_binary(self.use_gui)
        # SUMO does not create parent directories for detector output files.
        scenario = ET.parse(self.sumocfg_path).getroot()
        for entry in scenario.findall("./input/additional-files"):
            for name in entry.attrib["value"].split(","):
                additional = (self.sumocfg_path.parent / name.strip()).resolve()
                for detector in ET.parse(additional).getroot():
                    output = detector.get("file")
                    if output:
                        (additional.parent / output).resolve().parent.mkdir(parents=True, exist_ok=True)
        sim_total_seconds = int((self.profile.end_dt - self.profile.start_dt).total_seconds()) + 1
        if self.max_seconds is not None:
            sim_total_seconds = min(sim_total_seconds, self.max_seconds)
        if sim_total_seconds <= 0:
            raise ValueError("Computed simulation duration must be > 0")

        cmd = [
            binary,
            "-c",
            str(self.sumocfg_path),
            "-r",
            str(self.runtime_route_path),
            "--start",
            "--end",
            str(sim_total_seconds),
        ]
        cmd.extend(self.extra_sumo_args)

        print(f"[run] SUMO binary: {binary}")
        print(f"[run] sumocfg: {self.sumocfg_path}")
        print(f"[run] route override: {self.runtime_route_path}")
        print(f"[run] start: {self.profile.start_dt.isoformat(sep=' ')}")
        print(f"[run] end: {self.profile.end_dt.isoformat(sep=' ')}")
        print(f"[run] effective duration (seconds): {sim_total_seconds}")
        print(f"[run] stream count: {len(self.streams)}")
        print(f"[run] flow mode: {self.flow_mode}")
        if self.anomaly_scheduler is not None:
            print(f"[run] anomaly mode: {self.anomaly_scheduler.mode}")
            print(
                f"[run] anomaly scheduled events: "
                f"{len(self.anomaly_scheduler.runtime.schedule)}"
            )

        traci.start(cmd)
        try:
            sim_second = 0
            while sim_second < sim_total_seconds:
                current_dt = self.profile.start_dt + timedelta(seconds=sim_second)
                if self.anomaly_scheduler is not None:
                    logs = self.anomaly_scheduler.on_step(sim_second, current_dt)
                    for message in logs:
                        print(message)
                self._spawn_for_second(sim_second)
                next_target = sim_second + self.step_length
                try:
                    traci.simulationStep(next_target)
                except traci_exceptions.FatalTraCIError as exc:
                    hint = (
                        "SUMO process exited unexpectedly. "
                        "This network is known to crash under load with internal links enabled. "
                        "Use extra_sumo_args: ['--no-internal-links','true']."
                    )
                    raise RuntimeError(
                        f"{exc} | sim_second={sim_second} | next_target={next_target}. {hint}"
                    ) from exc
                sim_second = int(traci.simulation.getTime())
                self._log_completed_hours(sim_second)
                if sim_second > 0 and sim_second % 3600 == 0:
                    clock = format_seconds(sim_second)
                    print(f"[run] progressed to {clock} ({sim_second} s)")
        finally:
            try:
                traci.close()
            except traci_exceptions.TraCIException:
                pass
            if self.anomaly_scheduler is not None:
                self.anomaly_scheduler.write_outputs()
                if self.anomaly_scheduler.runtime.step_output_path is not None:
                    print(
                        "[run] anomaly step labels: "
                        f"{self.anomaly_scheduler.runtime.step_output_path}"
                    )
                if self.anomaly_scheduler.runtime.window_output_path is not None:
                    print(
                        "[run] anomaly window labels: "
                        f"{self.anomaly_scheduler.runtime.window_output_path}"
                    )
                if self.anomaly_scheduler.runtime.spawn_output_path is not None:
                    print(
                        "[run] anomaly spawn labels: "
                        f"{self.anomaly_scheduler.runtime.spawn_output_path}"
                    )
        self._log_remaining_hours(sim_total_seconds)
        print(f"[run] successful spawned vehicles: {self.successful_spawn_total}")
        print("[run] Finished.")

    def _spawn_for_second(self, sim_second: int) -> None:
        current_dt = self.profile.start_dt + timedelta(seconds=sim_second)
        day_factor = 1.0
        slot = None
        if self.flow_mode == "time_varying":
            current_date = current_dt.date()
            second_in_day = (
                current_dt.hour * 3600 + current_dt.minute * 60 + current_dt.second
            )
            slot = pick_slot(second_in_day, self.profile.pick_slots_for_date(current_date))
            if slot is None:
                return
            day_factor = self.profile.factor_for_date(current_date)

        for stream in self.streams:
            if self.flow_mode == "constant_scaled":
                base_rate = self.constant_base_rates.get(
                    (stream.profile_id, stream.vehicle_key),
                    0.0,
                )
            else:
                if slot is None:
                    continue
                bounds = slot.rates.get(stream.profile_id, {}).get(stream.vehicle_key)
                if bounds is None:
                    continue
                min_rate, max_rate = bounds
                if max_rate < min_rate:
                    min_rate, max_rate = max_rate, min_rate
                if self.rate_bound == "min":
                    base_rate = min_rate
                elif self.rate_bound == "max":
                    base_rate = max_rate
                else:
                    base_rate = self.random.uniform(min_rate, max_rate)
            if base_rate <= 0:
                continue

            rate_per_hour = (
                base_rate
                * stream.weight_share
                * day_factor
                * self.traffic_scale
            )
            stream_key = (stream.profile_id, stream.vehicle_key, stream.route_id)
            carry = self.stream_carry[stream_key] + (rate_per_hour * self.step_length / 3600.0)
            spawn_count = int(carry)
            self.stream_carry[stream_key] = carry - spawn_count
            if spawn_count <= 0:
                continue

            for _ in range(spawn_count):
                self._spawn_vehicle(stream, sim_second)

    def _build_constant_base_rates(self) -> dict[tuple[str, str], float]:
        if self.flow_mode != "constant_scaled":
            return {}

        weighted: dict[tuple[str, str], list[float]] = {}
        durations: dict[tuple[str, str], float] = {}
        for slots in self.profile.template_by_serial.values():
            for slot in slots:
                duration = float(max(0, slot.end_second - slot.start_second))
                if duration <= 0:
                    continue
                for profile_id, vehicle_rates in slot.rates.items():
                    for vehicle_key, bounds in vehicle_rates.items():
                        min_rate, max_rate = bounds
                        if max_rate < min_rate:
                            min_rate, max_rate = max_rate, min_rate
                        key = (profile_id, vehicle_key)
                        values = weighted.setdefault(key, [0.0, 0.0])
                        values[0] += min_rate * duration
                        values[1] += max_rate * duration
                        durations[key] = durations.get(key, 0.0) + duration

        out: dict[tuple[str, str], float] = {}
        for key, values in weighted.items():
            duration = durations[key]
            min_rate = values[0] / duration
            max_rate = values[1] / duration
            if self.rate_bound == "min":
                out[key] = min_rate
            elif self.rate_bound == "max":
                out[key] = max_rate
            else:
                out[key] = self.random.uniform(min_rate, max_rate)
        return out

    def _spawn_vehicle(self, stream: StreamDef, sim_second: int) -> None:
        veh_id = (
            f"xlsx_{stream.profile_id}_{stream.vehicle_key}_{sim_second}_{self.vehicle_counter}"
        )
        self.vehicle_counter += 1
        try:
            traci.vehicle.add(
                veh_id,
                stream.route_id,
                typeID=stream.sumo_vtype,
                depart="now",
                departLane="free",
                departPos="base",
                departSpeed="desired",
            )
            hour_idx = sim_second // 3600
            route_counts = self.hourly_spawn_by_route.setdefault(hour_idx, {})
            route_counts[stream.route_id] = route_counts.get(stream.route_id, 0) + 1
            self.successful_spawn_total += 1
        except traci_exceptions.TraCIException:
            # Keep run alive if one insertion fails due temporary lane constraints.
            pass

    def _log_completed_hours(self, current_sim_second: int) -> None:
        completed_hours = current_sim_second // 3600
        last_completed_hour = completed_hours - 1
        while self.last_logged_hour < last_completed_hour:
            next_hour = self.last_logged_hour + 1
            self._log_hour_bucket(next_hour, partial_until_second=None)
            self.last_logged_hour = next_hour

    def _log_remaining_hours(self, sim_total_seconds: int) -> None:
        final_sim_second = max(0, sim_total_seconds - 1)
        last_hour_idx = final_sim_second // 3600
        while self.last_logged_hour < last_hour_idx:
            next_hour = self.last_logged_hour + 1
            partial_until_second = None
            if next_hour == last_hour_idx:
                partial_until_second = final_sim_second
            self._log_hour_bucket(next_hour, partial_until_second=partial_until_second)
            self.last_logged_hour = next_hour

    def _log_hour_bucket(self, hour_idx: int, partial_until_second: int | None) -> None:
        start_second = hour_idx * 3600
        end_second = start_second + 3599
        partial_tag = ""
        if partial_until_second is not None and partial_until_second < end_second:
            end_second = partial_until_second
            partial_tag = " partial"

        route_counts = self.hourly_spawn_by_route.get(hour_idx, {})
        total = sum(route_counts.values())
        if route_counts:
            detail = ", ".join(
                f"{route_id}:{count}" for route_id, count in sorted(route_counts.items())
            )
            print(
                f"[run][spawn]{partial_tag} {format_seconds(start_second)}-"
                f"{format_seconds(end_second)} total={total} by_route={detail}"
            )
            return
        print(
            f"[run][spawn]{partial_tag} {format_seconds(start_second)}-"
            f"{format_seconds(end_second)} total=0 by_route=none"
        )


def build_streams(
    route_profiles_cfg: dict[str, Any],
    vehicle_type_map: dict[str, str],
    valid_routes: set[str],
    route_lane_count: dict[str, int],
    route_edges_by_id: dict[str, tuple[str, ...]],
    vtype_vclass_map: dict[str, str],
    routing_model: NetRoutingModel,
    enable_30_thang_4_routes: bool,
) -> tuple[StreamDef, ...]:
    route_profiles: list[RouteProfile] = []
    for profile_id, profile_any in route_profiles_cfg.items():
        profile_cfg = dict(profile_any)
        depart_lane_raw = profile_cfg["depart_lane"]
        depart_lane = str(depart_lane_raw)
        routes_cfg = profile_cfg.get("routes", [])
        if not isinstance(routes_cfg, list) or not routes_cfg:
            raise ValueError(f"route_profiles.{profile_id}.routes must be a non-empty list")
        route_weights: list[RouteWeight] = []
        excluded_30_04_count = 0
        for route_any in routes_cfg:
            route_cfg = dict(route_any)
            route_id = str(route_cfg["id"])
            if not enable_30_thang_4_routes and route_id.startswith("r_duong_30_thang_4"):
                excluded_30_04_count += 1
                continue
            if route_id not in valid_routes:
                raise ValueError(f"Configured route '{route_id}' not found in route file")
            weight = float(route_cfg.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"Route weight must be > 0 (profile={profile_id}, route={route_id})")
            route_depart_lane = str(route_cfg.get("depart_lane", depart_lane))
            if route_depart_lane != "best":
                lane_idx = int(route_depart_lane)
                lane_count = route_lane_count.get(route_id)
                if lane_count is None:
                    raise ValueError(f"Cannot resolve lane count for route '{route_id}'")
                if lane_idx < 0 or lane_idx >= lane_count:
                    raise ValueError(
                        f"Invalid depart lane {lane_idx} for route {route_id} (lane_count={lane_count})"
                    )
            route_weights.append(
                RouteWeight(route_id=route_id, weight=weight, depart_lane=route_depart_lane)
            )
        if excluded_30_04_count > 0:
            print(
                "[summary] excluded routes due config "
                f"profile={profile_id} count={excluded_30_04_count} "
                "prefix=r_duong_30_thang_4*"
            )
        route_profiles.append(
            RouteProfile(
                profile_id=profile_id,
                depart_lane=depart_lane,
                routes=tuple(route_weights),
            )
        )

    streams: list[StreamDef] = []
    for profile in route_profiles:
        for vehicle_key, sumo_vtype in vehicle_type_map.items():
            vclass = vtype_vclass_map.get(sumo_vtype)
            if not vclass:
                raise ValueError(
                    f"Vehicle type '{sumo_vtype}' (key={vehicle_key}) has no vClass in route file"
                )

            valid_vehicle_routes: list[RouteWeight] = []
            invalid_route_ids: list[str] = []
            for route in profile.routes:
                route_edges = route_edges_by_id.get(route.route_id)
                if not route_edges or not route_is_valid_for_vclass(
                    route_edges, vclass, routing_model
                ):
                    invalid_route_ids.append(route.route_id)
                    continue
                valid_vehicle_routes.append(route)

            if invalid_route_ids:
                detail = ", ".join(sorted(set(invalid_route_ids)))
                print(
                    "[summary] filtered invalid routes "
                    f"profile={profile.profile_id} vehicle={vehicle_key} "
                    f"vClass={vclass} count={len(invalid_route_ids)} routes={detail}"
                )
            if not valid_vehicle_routes:
                print(
                    "[summary] skipped vehicle profile due no valid routes "
                    f"profile={profile.profile_id} vehicle={vehicle_key} vClass={vclass}"
                )
                continue

            total_weight = sum(route.weight for route in valid_vehicle_routes)
            for route in valid_vehicle_routes:
                streams.append(
                    StreamDef(
                        profile_id=profile.profile_id,
                        vehicle_key=vehicle_key,
                        sumo_vtype=sumo_vtype,
                        route_id=route.route_id,
                        depart_lane=route.depart_lane,
                        weight_share=route.weight / total_weight,
                    )
                )
    return tuple(streams)


def prepare_routes_only_file(source_route: Path, output_route: Path) -> None:
    tree = ET.parse(source_route)
    root = tree.getroot()
    for flow in list(root.findall("flow")):
        root.remove(flow)
    ET.indent(root)
    output_route.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_route, encoding="utf-8", xml_declaration=True)


def parse_route_ids(route_file: Path) -> set[str]:
    root = ET.parse(route_file).getroot()
    out: set[str] = set()
    for node in root.findall("route"):
        route_id = node.get("id")
        if route_id:
            out.add(route_id)
    return out


def parse_route_first_edges(route_file: Path) -> dict[str, str]:
    root = ET.parse(route_file).getroot()
    out: dict[str, str] = {}
    for node in root.findall("route"):
        route_id = node.get("id")
        edges_raw = node.get("edges", "")
        if not route_id or not edges_raw.strip():
            continue
        first_edge = edges_raw.split()[0]
        out[route_id] = first_edge
    return out


def parse_route_edges(route_file: Path) -> dict[str, tuple[str, ...]]:
    root = ET.parse(route_file).getroot()
    out: dict[str, tuple[str, ...]] = {}
    for node in root.findall("route"):
        route_id = node.get("id")
        edges_raw = node.get("edges", "")
        if not route_id:
            continue
        edges = tuple(edge_id for edge_id in edges_raw.split() if edge_id)
        if edges:
            out[route_id] = edges
    return out


def parse_route_vclass_map(route_file: Path) -> dict[str, str]:
    root = ET.parse(route_file).getroot()
    out: dict[str, str] = {}
    for node in root.findall("vType"):
        type_id = node.get("id")
        if not type_id:
            continue
        vclass = node.get("vClass") or node.get("vclass")
        if vclass:
            out[type_id] = vclass
    return out


def parse_sumocfg_net_path(sumocfg_path: Path) -> Path:
    root = ET.parse(sumocfg_path).getroot()
    input_node = root.find("input")
    if input_node is None:
        raise ValueError(f"Missing <input> in {sumocfg_path}")
    net_node = input_node.find("net-file")
    if net_node is None:
        raise ValueError(f"Missing <net-file> in {sumocfg_path}")
    value = (net_node.get("value") or "").strip()
    if not value:
        raise ValueError(f"Empty net-file value in {sumocfg_path}")
    return (sumocfg_path.parent / value).resolve()


def parse_edge_lane_counts(net_file: Path) -> dict[str, int]:
    root = ET.parse(net_file).getroot()
    out: dict[str, int] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id:
            continue
        if edge.get("function") == "internal":
            continue
        lane_count = len(edge.findall("lane"))
        out[edge_id] = lane_count
    return out


def parse_net_routing_model(net_file: Path) -> NetRoutingModel:
    root = ET.parse(net_file).getroot()
    lane_permissions: dict[str, dict[int, tuple[set[str] | None, set[str]]]] = {}
    connection_lane_pairs: dict[tuple[str, str], list[tuple[int, int]]] = {}

    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        edge_id = edge.get("id")
        if not edge_id:
            continue
        edge_lanes: dict[int, tuple[set[str] | None, set[str]]] = {}
        for lane in edge.findall("lane"):
            lane_index_raw = lane.get("index")
            if lane_index_raw is None:
                continue
            lane_index = int(lane_index_raw)
            allow_raw = lane.get("allow")
            disallow_raw = lane.get("disallow")
            allow_set = set(allow_raw.split()) if allow_raw else None
            disallow_set = set(disallow_raw.split()) if disallow_raw else set()
            edge_lanes[lane_index] = (allow_set, disallow_set)
        lane_permissions[edge_id] = edge_lanes

    for connection in root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        from_lane_raw = connection.get("fromLane")
        to_lane_raw = connection.get("toLane")
        if (
            from_edge is None
            or to_edge is None
            or from_lane_raw is None
            or to_lane_raw is None
        ):
            continue
        if from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        from_lane = int(from_lane_raw)
        to_lane = int(to_lane_raw)
        key = (from_edge, to_edge)
        connection_lane_pairs.setdefault(key, []).append((from_lane, to_lane))

    return NetRoutingModel(
        lane_permissions=lane_permissions,
        connection_lane_pairs={
            key: tuple(value) for key, value in connection_lane_pairs.items()
        },
    )


def lane_allows_vclass(
    model: NetRoutingModel, edge_id: str, lane_idx: int, vclass: str
) -> bool:
    lane_map = model.lane_permissions.get(edge_id, {})
    lane_info = lane_map.get(lane_idx)
    if lane_info is None:
        return False
    allow_set, disallow_set = lane_info
    if allow_set is not None:
        return vclass in allow_set
    return vclass not in disallow_set


def route_is_valid_for_vclass(
    route_edges: tuple[str, ...], vclass: str, model: NetRoutingModel
) -> bool:
    if not route_edges:
        return False
    first_edge = route_edges[0]
    first_lanes = model.lane_permissions.get(first_edge, {})
    if not any(
        lane_allows_vclass(model, first_edge, lane_idx, vclass)
        for lane_idx in first_lanes
    ):
        return False
    for from_edge, to_edge in zip(route_edges, route_edges[1:]):
        lane_pairs = model.connection_lane_pairs.get((from_edge, to_edge), ())
        if not lane_pairs:
            return False
        if not any(
            lane_allows_vclass(model, from_edge, from_lane, vclass)
            and lane_allows_vclass(model, to_edge, to_lane, vclass)
            for from_lane, to_lane in lane_pairs
        ):
            return False
    return True


def parse_route_lane_counts(route_file: Path, net_file: Path) -> dict[str, int]:
    route_first_edges = parse_route_first_edges(route_file)
    edge_lanes = parse_edge_lane_counts(net_file)
    out: dict[str, int] = {}
    for route_id, edge_id in route_first_edges.items():
        if edge_id in edge_lanes:
            out[route_id] = edge_lanes[edge_id]
            continue
        flipped = edge_id[1:] if edge_id.startswith("-") else edge_id
        if flipped in edge_lanes:
            out[route_id] = edge_lanes[flipped]
            continue
        out[route_id] = 0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SUMO vehicle flows via TraCI from SUMO-Simulate.xlsx."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/simulation/xlsx_traci_flow_config.json"),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config/xlsx and print summary without launching SUMO.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Optional runtime cap for debugging.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Force command-line SUMO (sumo) instead of sumo-gui.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Root JSON config must be an object")
    return data


def resolve_path(config_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def resolve_sumo_binary(use_gui: bool) -> str:
    binary_name = "sumo-gui" if use_gui else "sumo"
    in_path = shutil.which(binary_name)
    if in_path:
        return in_path

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        suffix = ".exe" if os.name == "nt" else ""
        candidate = Path(sumo_home) / "bin" / f"{binary_name}{suffix}"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Cannot find '{binary_name}'. Add SUMO to PATH or set SUMO_HOME/bin."
    )


def parse_datetime_from_text(raw: str) -> datetime:
    match = DATE_TIME_TEXT_RE.search(raw)
    if not match:
        raise ValueError(f"Cannot parse datetime from: {raw}")
    t_obj = time.fromisoformat(match.group("h"))
    d_obj = datetime.strptime(match.group("d"), "%d/%m/%Y").date()
    return datetime.combine(d_obj, t_obj)


def parse_datetime_or_time(raw: str, anchor_date: date) -> datetime:
    text = str(raw).strip()
    if text == "":
        raise ValueError("Datetime override cannot be empty")

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        return parsed
    except ValueError:
        pass

    try:
        tm = time.fromisoformat(text)
        return datetime.combine(anchor_date, tm)
    except ValueError as exc:
        raise ValueError(
            f"Invalid datetime override '{text}'. "
            "Use 'YYYY-MM-DD HH:MM[:SS]' or 'HH:MM[:SS]'."
        ) from exc


def apply_window_overrides(profile: ExcelTrafficProfile, sumo_cfg: dict[str, Any]) -> None:
    excel_start = profile.start_dt
    excel_end = profile.end_dt

    start_raw = sumo_cfg.get("start_datetime_override")
    end_raw = sumo_cfg.get("end_datetime_override")

    start_dt = excel_start
    end_dt = excel_end
    if start_raw is not None and str(start_raw).strip() != "":
        start_dt = parse_datetime_or_time(str(start_raw), anchor_date=excel_start.date())
    if end_raw is not None and str(end_raw).strip() != "":
        end_dt = parse_datetime_or_time(str(end_raw), anchor_date=excel_end.date())

    if start_dt < excel_start or start_dt > excel_end:
        raise ValueError(
            "sumo.start_datetime_override is outside Excel window: "
            f"{start_dt} not in [{excel_start}, {excel_end}]"
        )
    if end_dt < excel_start or end_dt > excel_end:
        raise ValueError(
            "sumo.end_datetime_override is outside Excel window: "
            f"{end_dt} not in [{excel_start}, {excel_end}]"
        )
    if end_dt <= start_dt:
        raise ValueError(
            f"Invalid run window after overrides: start={start_dt}, end={end_dt}"
        )

    profile.start_dt = start_dt
    profile.end_dt = end_dt


def parse_float(raw: str) -> float:
    cleaned = str(raw).strip().replace(",", ".")
    if cleaned == "":
        raise ValueError("Cannot parse float from empty string")
    return float(cleaned)


def split_cell_ref(cell_ref: str) -> tuple[str, int]:
    match = CELL_REF_RE.match(cell_ref.upper())
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    return match.group(1), int(match.group(2))


def pick_slot(second_in_day: int, slots: list[DaySlot]) -> DaySlot | None:
    for slot in slots:
        if slot.start_second <= second_in_day < slot.end_second:
            return slot
    return None


def format_seconds(total_seconds: int) -> str:
    day = total_seconds // SECONDS_PER_DAY
    rem = total_seconds % SECONDS_PER_DAY
    h = rem // 3600
    m = (rem % 3600) // 60
    s = rem % 60
    if day > 0:
        return f"D+{day} {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_key(raw: str) -> str:
    txt = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    txt = txt.lower().strip()
    txt = txt.replace(" ", "_")
    txt = re.sub(r"[^a-z0-9_]+", "", txt)
    return txt


def date_to_excel_serial(d: date) -> int:
    # Excel 1900 date system.
    base = date(1899, 12, 30)
    return (d - base).days


def print_summary(
    profile: ExcelTrafficProfile,
    streams: tuple[StreamDef, ...],
    rate_bound: str,
    traffic_scale: float,
    flow_mode: str,
    anomaly_scheduler: AnomalyScheduler | None = None,
) -> None:
    start = profile.start_dt
    end = profile.end_dt
    duration_sec = int((end - start).total_seconds()) + 1
    print(f"[summary] start={start.isoformat(sep=' ')}")
    print(f"[summary] end={end.isoformat(sep=' ')}")
    print(f"[summary] duration_seconds={duration_sec}")
    print(f"[summary] rate_bound={rate_bound}")
    print(f"[summary] traffic_scale={traffic_scale}")
    print(f"[summary] flow_mode={flow_mode}")
    print(f"[summary] template_dates={len(profile.template_by_serial)}")
    for serial, slots in profile.template_by_serial.items():
        print(f"[summary] date_serial={serial} slots={len(slots)}")
        for slot in slots[:3]:
            print(
                f"  - {format_seconds(slot.start_second)} -> {format_seconds(slot.end_second)}"
            )
    print(f"[summary] weekend_factor={profile.weekend_factor}")
    if profile.holiday_factors:
        for h_date, factor in sorted(profile.holiday_factors.items()):
            print(f"[summary] holiday={h_date.isoformat()} factor={factor}")
    print(f"[summary] stream_count={len(streams)}")
    if anomaly_scheduler is not None:
        print(f"[summary] anomaly_mode={anomaly_scheduler.mode}")
        print(f"[summary] anomaly_schedule_count={len(anomaly_scheduler.runtime.schedule)}")
        if anomaly_scheduler.runtime.duration is not None:
            print(f"[summary] anomaly_duration={anomaly_scheduler.runtime.duration}")
        if anomaly_scheduler.runtime.step_output_path is not None:
            print(
                "[summary] anomaly_step_output="
                f"{anomaly_scheduler.runtime.step_output_path}"
            )
        if anomaly_scheduler.runtime.window_output_path is not None:
            print(
                "[summary] anomaly_window_output="
                f"{anomaly_scheduler.runtime.window_output_path}"
            )
        if anomaly_scheduler.runtime.spawn_output_path is not None:
            print(
                "[summary] anomaly_spawn_output="
                f"{anomaly_scheduler.runtime.spawn_output_path}"
            )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_dir = config_path.parent

    config = load_config(config_path)

    sumo_cfg = dict(config["sumo"])
    xlsx_cfg = dict(config["xlsx"])
    traffic_cfg = dict(config["traffic"])

    use_gui = bool(sumo_cfg.get("use_gui", True))
    if args.no_gui:
        use_gui = False

    step_length = float(sumo_cfg.get("step_length_seconds", 1.0))
    if step_length <= 0:
        raise ValueError("sumo.step_length_seconds must be > 0")

    seed = int(sumo_cfg.get("seed", 42))
    extra_sumo_args = sumo_cfg.get("extra_sumo_args", [])
    if not isinstance(extra_sumo_args, list):
        raise ValueError("sumo.extra_sumo_args must be a list")
    extra_sumo_args = [str(v) for v in extra_sumo_args]

    sumocfg_path = resolve_path(config_dir, str(sumo_cfg["sumocfg_path"]))
    if not sumocfg_path.exists():
        raise FileNotFoundError(f"sumocfg not found: {sumocfg_path}")

    route_source = resolve_path(config_dir, str(sumo_cfg["route_source_path"]))
    if not route_source.exists():
        raise FileNotFoundError(f"route_source_path not found: {route_source}")

    runtime_route = resolve_path(config_dir, str(sumo_cfg["runtime_route_path"]))

    xlsx_path = resolve_path(config_dir, str(xlsx_cfg["path"]))
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    sheet_name = str(xlsx_cfg["sheet_name"])
    rate_bound = normalize_key(str(traffic_cfg.get("rate_bound", "max")))
    if rate_bound not in {"min", "max", "random"}:
        raise ValueError("traffic.rate_bound must be 'min', 'max', or 'random'")
    traffic_scale = float(traffic_cfg.get("traffic_scale", 1.0))
    if traffic_scale <= 0:
        raise ValueError("traffic.traffic_scale must be > 0")
    flow_mode = normalize_key(str(traffic_cfg.get("flow_mode", "time_varying")))
    if flow_mode not in {"time_varying", "constant_scaled"}:
        raise ValueError("traffic.flow_mode must be 'time_varying' or 'constant_scaled'")
    enable_30_thang_4_routes_raw = traffic_cfg.get("enable_30_thang_4_routes", True)
    if not isinstance(enable_30_thang_4_routes_raw, bool):
        raise ValueError("traffic.enable_30_thang_4_routes must be a boolean")
    enable_30_thang_4_routes = enable_30_thang_4_routes_raw

    vehicle_type_map_raw = dict(traffic_cfg["vehicle_type_map"])
    vehicle_type_map: dict[str, str] = {}
    for key, value in vehicle_type_map_raw.items():
        vehicle_type_map[normalize_key(str(key))] = str(value)

    route_profiles_cfg = dict(traffic_cfg["route_profiles"])

    prepare_routes_only_file(route_source, runtime_route)

    valid_routes = parse_route_ids(runtime_route)
    if not valid_routes:
        raise ValueError(f"No <route> found in {runtime_route}")

    net_file = parse_sumocfg_net_path(sumocfg_path)
    edge_lane_count = parse_edge_lane_counts(net_file)
    route_lane_count = parse_route_lane_counts(runtime_route, net_file)
    route_edges_by_id = parse_route_edges(runtime_route)
    vtype_vclass_map = parse_route_vclass_map(runtime_route)
    routing_model = parse_net_routing_model(net_file)

    streams = build_streams(
        route_profiles_cfg=route_profiles_cfg,
        vehicle_type_map=vehicle_type_map,
        valid_routes=valid_routes,
        route_lane_count=route_lane_count,
        route_edges_by_id=route_edges_by_id,
        vtype_vclass_map=vtype_vclass_map,
        routing_model=routing_model,
        enable_30_thang_4_routes=enable_30_thang_4_routes,
    )
    if not streams:
        raise ValueError("No valid streams after route compatibility filtering.")

    sheet = XlsxSheet(xlsx_path=xlsx_path, sheet_name=sheet_name)
    profile = ExcelTrafficProfile(
        sheet=sheet,
        xlsx_cfg=xlsx_cfg,
        rate_bound=rate_bound,
    )
    apply_window_overrides(profile, sumo_cfg)

    anomaly_scheduler: AnomalyScheduler | None = None
    raw_anomaly_cfg = config.get("anomaly")
    if raw_anomaly_cfg is not None:
        anomaly_runtime = parse_anomaly_runtime_config(raw_anomaly_cfg, config_dir=config_dir)
        anomaly_injector: AnomalyInjector | None = None
        if anomaly_runtime.mode != ANOMALY_BASELINE_MODE:
            valid_edge_ids = {
                edge_id
                for edge_id, lane_count in edge_lane_count.items()
                if lane_count > 0
            }
            if not valid_edge_ids:
                raise RuntimeError(
                    "No valid edges found for anomaly scheduling. "
                    "Cannot inject anomaly mode != baseline."
                )
            anomaly_injector = AnomalyInjector(valid_edge_ids=valid_edge_ids, seed=seed)
        anomaly_scheduler = AnomalyScheduler(
            runtime=anomaly_runtime,
            start_dt=profile.start_dt,
            injector=anomaly_injector,
            valid_edge_ids={edge_id for edge_id, lanes in edge_lane_count.items() if lanes > 0},
        )

    # Validate that all configured vehicle keys exist in Excel column map.
    excel_vehicle_keys: set[str] = set()
    table_cfg = dict(xlsx_cfg["table"])
    for _profile_id, vehicle_map_any in dict(table_cfg["column_map"]).items():
        vehicle_map = dict(vehicle_map_any)
        for v_key in vehicle_map:
            excel_vehicle_keys.add(normalize_key(v_key))
    configured_vehicle_keys = set(vehicle_type_map.keys())
    missing_keys = configured_vehicle_keys - excel_vehicle_keys
    if missing_keys:
        raise ValueError(
            "Configured traffic.vehicle_type_map contains keys missing in xlsx.table.column_map: "
            + ", ".join(sorted(missing_keys))
        )

    print_summary(
        profile,
        streams,
        rate_bound,
        traffic_scale,
        flow_mode,
        anomaly_scheduler=anomaly_scheduler,
    )
    if args.dry_run:
        print("[summary] dry-run complete, no TraCI execution.")
        return

    runner = TraciFlowRunner(
        profile=profile,
        streams=streams,
        rate_bound=rate_bound,
        traffic_scale=traffic_scale,
        flow_mode=flow_mode,
        sumocfg_path=sumocfg_path,
        runtime_route_path=runtime_route,
        use_gui=use_gui,
        step_length=step_length,
        seed=seed,
        extra_sumo_args=extra_sumo_args,
        max_seconds=args.max_seconds,
        anomaly_scheduler=anomaly_scheduler,
    )
    runner.run()


if __name__ == "__main__":
    main()
