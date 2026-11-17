from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"a": NS_MAIN, "r": NS_REL, "pr": NS_PKG_REL}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Daily Traffic Density Profile figure from SUMO-Simulate.xlsx."
    )
    parser.add_argument(
        "--xlsx-path",
        type=str,
        default="SUMO-Simulate.xlsx",
        help="Path to input Excel file.",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default="thiết lập khung thời gian",
        help="Worksheet name containing hourly density data.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="daily_traffic_density_profile.png",
        help="Output PNG figure path.",
    )
    return parser.parse_args()


def _col_from_ref(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out: list[str] = []
    for si in root.findall("a:si", NS):
        txt = "".join((node.text or "") for node in si.findall(".//a:t", NS))
        out.append(txt)
    return out


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    typ = cell.attrib.get("t")
    value_node = cell.find("a:v", NS)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if typ == "s":
        idx = int(raw)
        if 0 <= idx < len(shared_strings):
            return shared_strings[idx]
    return raw


def _resolve_sheet_xml_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
    target_sheet = None
    for sheet in workbook_root.findall("a:sheets/a:sheet", NS):
        if sheet.attrib.get("name") == sheet_name:
            target_sheet = sheet
            break
    if target_sheet is None:
        raise ValueError(f"Sheet not found: {sheet_name}")

    rid = target_sheet.attrib.get(f"{{{NS_REL}}}id")
    if rid is None:
        raise ValueError(f"Missing relationship id for sheet: {sheet_name}")

    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_target = None
    for rel in rel_root.findall("pr:Relationship", NS):
        if rel.attrib.get("Id") == rid:
            rel_target = rel.attrib.get("Target")
            break
    if rel_target is None:
        raise ValueError(f"Missing relationship target for sheet: {sheet_name}")

    rel_target = rel_target.lstrip("/")
    if not rel_target.startswith("xl/"):
        rel_target = f"xl/{rel_target}"
    return rel_target


def _load_sheet_rows(xlsx_path: Path, sheet_name: str) -> dict[int, dict[str, str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_xml_path = _resolve_sheet_xml_path(zf, sheet_name)
        sheet_root = ET.fromstring(zf.read(sheet_xml_path))

    row_map: dict[int, dict[str, str]] = {}
    for row in sheet_root.findall("a:sheetData/a:row", NS):
        row_id_raw = row.attrib.get("r")
        if row_id_raw is None:
            continue
        row_id = int(row_id_raw)
        cells: dict[str, str] = {}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r", "")
            col = _col_from_ref(ref)
            if not col:
                continue
            cells[col] = _cell_value(cell, shared_strings).strip()
        row_map[row_id] = cells
    return row_map


def _find_header_row(row_map: dict[int, dict[str, str]]) -> int:
    for row_id in sorted(row_map):
        row = row_map[row_id]
        b_val = row.get("B", "").strip().lower()
        c_val = row.get("C", "").strip().lower()
        f_val = row.get("F", "").strip().lower()
        if b_val == "ngày" and "giờ bắt đầu" in c_val and "% q_max cho mỗi lane" in f_val:
            return row_id
    raise ValueError("Cannot find header row for daily profile table.")


def _parse_density_ratio(raw: str) -> float:
    normalized = raw.replace("–", "-").replace("—", "-")
    normalized = normalized.replace(",", ".")
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%", normalized)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 200.0

    single_match = re.search(r"(\d+(?:\.\d+)?)\s*%", normalized)
    if single_match:
        return float(single_match.group(1)) / 100.0

    raise ValueError(f"Cannot parse density range: {raw}")


def _excel_time_fraction_to_hour(raw: str) -> float:
    value = float(raw)
    return value * 24.0


def _hour_to_hhmm(hour: float) -> str:
    total_minutes = int(round(hour * 60.0)) % (24 * 60)
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def _extract_daily_profile(
    row_map: dict[int, dict[str, str]],
    header_row: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id in sorted(row_map):
        if row_id <= header_row:
            continue
        row = row_map[row_id]
        start_raw = row.get("C", "").strip()
        end_raw = row.get("D", "").strip()
        density_raw = row.get("F", "").strip()

        if not start_raw and not end_raw and not density_raw:
            if records:
                break
            continue
        if not start_raw or not end_raw or not density_raw:
            continue

        try:
            start_hour = _excel_time_fraction_to_hour(start_raw)
            end_hour = _excel_time_fraction_to_hour(end_raw)
            density_ratio = _parse_density_ratio(density_raw)
        except ValueError:
            continue

        center_hour = (start_hour + end_hour) / 2.0
        records.append(
            {
                "row_id": row_id,
                "start_hour": start_hour,
                "end_hour": end_hour,
                "center_hour": center_hour,
                "density_ratio": density_ratio,
                "density_range_raw": density_raw,
            }
        )

    if not records:
        raise ValueError("No daily profile rows were extracted from the table.")
    return records


def _pick_peak(records: list[dict[str, Any]], start_hour: float, end_hour: float) -> dict[str, Any]:
    candidates = [
        r for r in records if start_hour <= float(r["center_hour"]) < end_hour
    ]
    if not candidates:
        raise ValueError(f"No records in peak window {start_hour}-{end_hour}.")
    candidates.sort(key=lambda r: (-float(r["density_ratio"]), float(r["center_hour"])))
    return candidates[0]


def _plot_profile(records: list[dict[str, Any]], output_path: Path) -> None:
    x = [float(r["center_hour"]) for r in records]
    y = [float(r["density_ratio"]) for r in records]

    morning_peak = _pick_peak(records, 6.0, 10.0)
    evening_peak = _pick_peak(records, 16.0, 20.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, color="#1f77b4", linewidth=2.2, marker="o", markersize=5, label="Density ratio")

    for peak, label, color in (
        (morning_peak, "Morning peak", "#d62728"),
        (evening_peak, "Evening peak", "#ff7f0e"),
    ):
        px = float(peak["center_hour"])
        py = float(peak["density_ratio"])
        ax.scatter([px], [py], color=color, s=75, zorder=4)
        ax.annotate(
            f"{label}\n{_hour_to_hhmm(px)} | {py:.3f}",
            xy=(px, py),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=9,
            color=color,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.85},
        )

    xticks = list(range(0, 25, 2))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{h:02d}:00" for h in xticks], rotation=0)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, max(1.1, max(y) * 1.15))
    ax.set_xlabel("Time of day", fontweight="bold")
    ax.set_ylabel("Density ratio", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    xlsx_path = Path(args.xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    rows = _load_sheet_rows(xlsx_path=xlsx_path, sheet_name=args.sheet_name)
    header_row = _find_header_row(rows)
    records = _extract_daily_profile(row_map=rows, header_row=header_row)
    output_path = Path(args.output_path)
    _plot_profile(records=records, output_path=output_path)
    print(f"Saved figure: {output_path.resolve()}")


if __name__ == "__main__":
    main()
