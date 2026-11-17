"""Network parsing helpers for anomaly scheduling/injection."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JunctionContext:
    """Metadata for one selectable junction."""

    junction_id: str
    junction_type: str
    routes: tuple[str, ...]
    incoming_edges: tuple[str, ...]
    outgoing_edges: tuple[str, ...]
    x: float
    y: float


class ScenarioNetworkIndex:
    """Builds a route-aware junction index from SUMO net + route files."""

    def __init__(self, junctions: dict[str, JunctionContext]) -> None:
        self.junctions = junctions

    @classmethod
    def from_files(cls, net_file: Path, route_file: Path) -> "ScenarioNetworkIndex":
        edges, junction_meta, incoming, outgoing = cls._parse_net(net_file)
        routes = cls._parse_routes(route_file)
        junction_to_routes: dict[str, set[str]] = defaultdict(set)

        for route_id, edge_ids in routes.items():
            for edge_a, edge_b in zip(edge_ids, edge_ids[1:]):
                a_meta = edges.get(edge_a)
                b_meta = edges.get(edge_b)
                if a_meta is None or b_meta is None:
                    continue
                transit_junction = a_meta[1]
                if b_meta[0] != transit_junction:
                    continue
                junction_to_routes[transit_junction].add(route_id)

        contexts: dict[str, JunctionContext] = {}
        for junction_id, route_ids in junction_to_routes.items():
            if len(route_ids) < 2:
                continue
            j_info = junction_meta.get(junction_id, {})
            j_type = j_info.get("type", "unknown")
            if junction_id.startswith(":") or j_type in {"internal", "dead_end"}:
                continue
            contexts[junction_id] = JunctionContext(
                junction_id=junction_id,
                junction_type=j_type,
                routes=tuple(sorted(route_ids)),
                incoming_edges=tuple(sorted(incoming.get(junction_id, set()))),
                outgoing_edges=tuple(sorted(outgoing.get(junction_id, set()))),
                x=float(j_info.get("x", 0.0)),
                y=float(j_info.get("y", 0.0)),
            )

        return cls(contexts)

    @staticmethod
    def _parse_routes(route_file: Path) -> dict[str, list[str]]:
        route_map: dict[str, list[str]] = {}
        root = ET.parse(route_file).getroot()
        for route_node in root.findall("route"):
            route_id = route_node.get("id")
            edges_raw = route_node.get("edges", "").strip()
            if not route_id or not edges_raw:
                continue
            route_map[route_id] = edges_raw.split()
        return route_map

    @staticmethod
    def _parse_net(
        net_file: Path,
    ) -> tuple[
        dict[str, tuple[str, str]],
        dict[str, dict[str, str]],
        dict[str, set[str]],
        dict[str, set[str]],
    ]:
        edges: dict[str, tuple[str, str]] = {}
        junctions: dict[str, dict[str, str]] = {}
        incoming: dict[str, set[str]] = defaultdict(set)
        outgoing: dict[str, set[str]] = defaultdict(set)

        root = ET.parse(net_file).getroot()
        for node in root:
            if node.tag == "junction":
                junction_id = node.get("id")
                if not junction_id:
                    continue
                junctions[junction_id] = {
                    "type": node.get("type", "unknown"),
                    "x": node.get("x", "0"),
                    "y": node.get("y", "0"),
                }
                continue

            if node.tag != "edge":
                continue
            edge_id = node.get("id")
            edge_from = node.get("from")
            edge_to = node.get("to")
            function = node.get("function", "")
            if (
                not edge_id
                or not edge_from
                or not edge_to
                or edge_id.startswith(":")
                or function == "internal"
            ):
                continue

            edges[edge_id] = (edge_from, edge_to)
            incoming[edge_to].add(edge_id)
            outgoing[edge_from].add(edge_id)

        return edges, junctions, incoming, outgoing


def parse_sumocfg_paths(sumocfg: Path) -> tuple[Path, Path]:
    """Read net + primary route file from a sumocfg file."""

    root = ET.parse(sumocfg).getroot()
    input_node = root.find("input")
    if input_node is None:
        raise ValueError(f"Missing <input> section in {sumocfg}")

    net_node = input_node.find("net-file")
    route_node = input_node.find("route-files")
    if net_node is None or route_node is None:
        raise ValueError(f"Missing net-file or route-files in {sumocfg}")

    net_value = (net_node.get("value") or "").strip()
    routes_value = (route_node.get("value") or "").strip()
    if not net_value or not routes_value:
        raise ValueError(f"Empty net-file or route-files value in {sumocfg}")

    route_file_rel = routes_value.split(",")[0].strip()
    net_path = (sumocfg.parent / net_value).resolve()
    route_path = (sumocfg.parent / route_file_rel).resolve()
    return net_path, route_path
