"""Shared helpers for analysis.plots_* submodules."""

import math
from pathlib import Path
from typing import Dict, List, Optional


def _normalize_orientation(value: str) -> str:
    if not value:
        return "horizontal"
    value = value.lower()
    if value in ("vertical", "vert", "v"):
        return "vertical"
    return "horizontal"


def _orientation_path(base_path: Path, orientation: str) -> Path:
    orientation = _normalize_orientation(orientation)
    return base_path.parent / orientation / base_path.name


def _filter_by_snr(
    entries: List[Dict[str, float]],
    snr_min: Optional[float],
    snr_max: Optional[float],
    snr_list: Optional[List[float]],
) -> List[Dict[str, float]]:
    if not entries:
        return []
    if snr_list:
        tol = 1e-6
        targets = snr_list
        return [
            e
            for e in entries
            if any(abs(e["snr_db"] - target) <= tol for target in targets)
        ]
    filtered = []
    for e in entries:
        snr_val = e["snr_db"]
        if snr_min is not None and snr_val < snr_min:
            continue
        if snr_max is not None and snr_val > snr_max:
            continue
        filtered.append(e)
    return filtered


def _filter_entries_by_symbols(
    entries: List[Dict[str, float]],
    target_symbols: int,
    snr_min: Optional[float],
    snr_max: Optional[float],
    snr_list: Optional[List[float]],
) -> List[Dict[str, float]]:
    """Filter rows to a single symbols target, sort by SNR, then apply SNR filter.

    Extracted from duplicate `_collect` closures previously in plots_compare.py
    and plots_velocity.py (round2 R-3).
    """
    filtered = [
        entry
        for entry in entries
        if not math.isnan(entry.get("symbols", math.nan))
        and int(round(entry["symbols"])) == target_symbols
    ]
    filtered.sort(key=lambda e: e["snr_db"])
    return _filter_by_snr(filtered, snr_min, snr_max, snr_list)
