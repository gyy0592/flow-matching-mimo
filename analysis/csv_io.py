"""CSV I/O + per-method row collection + result aggregation helpers."""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from analysis.metrics import (
    extract_ser,
    extract_x_nmse,
    final_nmse_db,
    _safe_metric_lookup,
)


RESULT_NAME_PATTERN = re.compile(
    r"numpilots(?P<pilots>\d+(?:\.\d+)?)_numsymbols(?P<symbols>\d+(?:\.\d+)?)"
)


def parse_metadata(path: Path) -> Tuple[float, float]:
    match = RESULT_NAME_PATTERN.search(path.name)
    pilots = math.nan
    symbols = math.nan
    if match:
        pilots = float(match.group("pilots"))
        symbols = float(match.group("symbols"))
        return pilots, symbols

    # Fallback for legacy baseline filenames such as results_Nt32_Nr64_30pilots_lr.pt.
    pilots_match = re.search(r"(?P<pilots>\d+)\s*pilots", path.name, re.IGNORECASE)
    if pilots_match:
        pilots = float(pilots_match.group("pilots"))
    return pilots, symbols


def collect_rows_for_method(
    method_name: str,
    directories: Iterable[Path],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    directories = list(directories)
    available_directories: List[Path] = []
    for directory in directories:
        if directory.exists():
            available_directories.append(directory)
        else:
            print(
                f"[warn] Directory missing for method '{method_name}': {directory}"
            )

    if not available_directories:
        return rows

    def _should_include(path: Path) -> bool:
        name = path.name.lower()
        if method_name == "langevin":
            return "reuse" in name
        if method_name == "single_langevin":
            return "reuse" not in name
        return True

    for directory in available_directories:
        # The formal launcher uses 30 pilots; existing NPZ files omit this metadata.
        npz_groups = {
            int(path.stem.split("_batch")[-1]): [path]
            for path in sorted(directory.glob("all_realizations_batch*.npz"))
        }
        sequential_symbols = set(npz_groups)
        for path in sorted(directory.glob("realization_*_batch*.npz")):
            symbols = int(path.stem.split("_batch")[-1])
            if symbols not in sequential_symbols:
                npz_groups.setdefault(symbols, []).append(path)
        result_groups = [
            [path]
            for path in sorted(directory.glob("*.pt"))
            if _should_include(path)
            and parse_metadata(path)[1] not in npz_groups
        ]
        result_groups.extend(npz_groups.values())
        if not result_groups:
            print(
                f"[warn] No .pt or .npz result files found for method '{method_name}' in: {directory}"
            )
            continue
        for paths in result_groups:
            path = paths[0]
            if path.suffix == ".npz":
                runs = []
                for run_path in paths:
                    with np.load(run_path) as data:
                        runs.append({key: data[key] for key in data.files})
                payload = {"snr_range": runs[0]["snr_range"]}
                for source_key, target_key in (
                    ("oracle_log_runs", "oracle_log"),
                    ("x_nmse_log_runs", "x_nmse_log"),
                    ("SER_velocity_runs", "SER_velocity"),
                ):
                    payload[target_key] = np.concatenate(
                        [run[source_key] for run in runs], axis=0
                    ).mean(axis=0)
                pilots, symbols = 30.0, float(path.stem.split("_batch")[-1])
            else:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                pilots, symbols = parse_metadata(path)
            snr = torch.as_tensor(payload["snr_range"], dtype=torch.float64)
            snr_len = snr.numel()
            nmse = final_nmse_db(payload["oracle_log"], snr_len)
            ser = extract_ser(payload)
            x_nmse = extract_x_nmse(payload, snr_len)
            for snr_idx in range(snr.shape[0]):
                nmse_value = float(nmse[min(snr_idx, nmse.numel() - 1)].item())
                row = {
                    "method": method_name,
                    "pilots": float(pilots),
                    "symbols": float(symbols),
                    "snr_db": float(snr[snr_idx].item()),
                    "nmse_db": float(nmse_value),
                }
                ser_value = _safe_metric_lookup(ser, snr_idx)
                if ser_value is not None:
                    row["ser"] = ser_value
                x_nmse_value = _safe_metric_lookup(x_nmse, snr_idx)
                if x_nmse_value is not None:
                    row["x_nmse"] = x_nmse_value
                rows.append(row)
    return rows


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_snr_shift(
    rows: List[Dict[str, float]],
    shift: float,
    excluded_methods: Optional[Iterable[str]] = None,
) -> None:
    if not shift:
        return
    excluded: set = set(excluded_methods or [])
    for row in rows:
        if row.get("method") in excluded:
            continue
        if "snr_db" in row and row["snr_db"] is not None:
            row["snr_db"] = float(row["snr_db"]) + shift


def _infer_tx_antennas(method_directories: Dict[str, List[Path]], default: int = 32) -> int:
    pattern = re.compile(r"Nt(\d+)", re.IGNORECASE)
    for directories in method_directories.values():
        for directory in directories:
            try:
                for candidate in directory.glob("*.pt"):
                    match = pattern.search(candidate.stem)
                    if match:
                        value = int(match.group(1))
                        if value > 0:
                            return value
            except FileNotFoundError:
                continue
    return default


def write_csv(
    csv_path: Path,
    method_directories: Dict[str, List[Path]],
    snr_shift: float,
) -> List[Dict[str, float]]:
    all_rows: List[Dict[str, float]] = []
    for method, directories in method_directories.items():
        rows = collect_rows_for_method(method, directories)
        if not rows:
            print(
                f"[warn] Skipping method '{method}' while building CSV because no valid result files were found."
            )
            continue
        all_rows.extend(rows)
    if not all_rows:
        return []
    _apply_snr_shift(all_rows, snr_shift)
    ensure_output_directory(csv_path)
    fieldnames = ["method", "pilots", "symbols", "snr_db", "nmse_db", "ser", "x_nmse"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            all_rows,
            key=lambda r: (r["method"], r["symbols"], r["snr_db"]),
        ):
            writer.writerow(row)
    return all_rows


def load_csv(csv_path: Path) -> List[Dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {
                "method": row["method"],
                "pilots": float(row["pilots"]) if row["pilots"] else math.nan,
                "symbols": float(row["symbols"]) if row["symbols"] else math.nan,
                "snr_db": float(row["snr_db"]) if row["snr_db"] else math.nan,
                "nmse_db": float(row["nmse_db"]) if row["nmse_db"] else math.nan,
                "ser": float(row["ser"]) if row.get("ser") else math.nan,
                "x_nmse": float(row["x_nmse"]) if row.get("x_nmse") else math.nan,
            }
            rows.append(parsed)
    return rows


def group_by_symbols(rows: List[Dict[str, float]], method: str) -> Dict[int, List[Dict[str, float]]]:
    grouped: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        symbols_value = row["symbols"]
        if math.isnan(symbols_value):
            continue
        symbols = int(round(symbols_value))
        grouped[symbols].append(row)
    for item in grouped.values():
        item.sort(key=lambda r: r["snr_db"])
    return grouped
