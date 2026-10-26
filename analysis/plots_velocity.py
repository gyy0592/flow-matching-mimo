"""Velocity vs Langevin paper plot."""

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from analysis.csv_io import ensure_output_directory
from analysis.plots_common import (
    _filter_entries_by_symbols,
    _normalize_orientation,
    _orientation_path,
)
from analysis.styling import (
    LANGEVIN_SYMBOL_COLORS,
    METHOD_DISPLAY_NAMES,
    VELOCITY_SYMBOL_COLORS,
    _style_for_variant,
)


def plot_velocity_vs_langevin_symbols(
    rows: List[Dict[str, float]],
    velocity_method: str,
    langevin_method: str,
    symbols_list: Iterable[int],
    out_path: Path,
    snr_min: Optional[float],
    snr_max: Optional[float],
    snr_list: Optional[List[float]],
    orientations: Iterable[str] = ("horizontal",),
    axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> None:
    if velocity_method not in METHOD_DISPLAY_NAMES or langevin_method not in METHOD_DISPLAY_NAMES:
        return

    symbols_list = list(symbols_list)
    if not symbols_list:
        return

    def _collect(entries: List[Dict[str, float]], target_symbols: int) -> List[Dict[str, float]]:
        return _filter_entries_by_symbols(entries, target_symbols, snr_min, snr_max, snr_list)

    velocity_entries = [row for row in rows if row["method"] == velocity_method]
    langevin_entries = [row for row in rows if row["method"] == langevin_method]
    if not velocity_entries or not langevin_entries:
        return

    def _render(ax_nmse, ax_ser):
        vel_nmse_lines = []
        lan_nmse_lines = []
        vel_ser_lines = []
        lan_ser_lines = []

        for idx, symbols in enumerate(symbols_list):
            vel_data = _collect(velocity_entries, symbols)
            lan_data = _collect(langevin_entries, symbols)
            if not vel_data and not lan_data:
                continue

            vel_color = VELOCITY_SYMBOL_COLORS.get(symbols)
            lan_color = LANGEVIN_SYMBOL_COLORS.get(symbols)

            vel_style = {"marker": "o", "linewidth": 2.0}
            if vel_color:
                vel_style["color"] = vel_color
            else:
                vel_style.update(_style_for_variant(velocity_method, idx, len(symbols_list)))

            lan_style = {"marker": "s", "linewidth": 2.0, "linestyle": "--", "markerfacecolor": "none"}
            if lan_color:
                lan_style["color"] = lan_color
            else:
                lan_style.update(_style_for_variant(langevin_method, idx, len(symbols_list)))

            velocity_display = METHOD_DISPLAY_NAMES.get(velocity_method, velocity_method)
            if velocity_method == "velocity_steps400":
                velocity_display = "OT Flow"
            velocity_label = f"{velocity_display} - D={symbols}"
            langevin_label = f"{METHOD_DISPLAY_NAMES.get(langevin_method, langevin_method)} - D={symbols}"

            if vel_data:
                snr_vals = [d["snr_db"] for d in vel_data]
                nmse_vals = [d["nmse_db"] for d in vel_data]
                line_nmse = ax_nmse.plot(snr_vals, nmse_vals, **vel_style)[0]
                line_nmse.set_label(velocity_label)
                vel_nmse_lines.append((line_nmse, velocity_label))
                ser_points = [
                    (d["snr_db"], d["ser"])
                    for d in vel_data
                    if not math.isnan(d.get("ser", math.nan))
                ]
                if ser_points:
                    snr_ser, ser_vals = zip(*ser_points)
                    line_ser = ax_ser.plot(snr_ser, ser_vals, **vel_style)[0]
                    line_ser.set_label(velocity_label)
                    vel_ser_lines.append((line_ser, velocity_label))

            if lan_data:
                snr_vals = [d["snr_db"] for d in lan_data]
                nmse_vals = [d["nmse_db"] for d in lan_data]
                line_nmse = ax_nmse.plot(snr_vals, nmse_vals, **lan_style)[0]
                line_nmse.set_label(langevin_label)
                lan_nmse_lines.append((line_nmse, langevin_label))
                ser_points = [
                    (d["snr_db"], d["ser"])
                    for d in lan_data
                    if not math.isnan(d.get("ser", math.nan))
                ]
                if ser_points:
                    snr_ser, ser_vals = zip(*ser_points)
                    line_ser = ax_ser.plot(snr_ser, ser_vals, **lan_style)[0]
                    line_ser.set_label(langevin_label)
                    lan_ser_lines.append((line_ser, langevin_label))

        ax_nmse.set_title("Channel NMSE Comparison")
        ax_nmse.set_ylabel("Final NMSE (dB)")
        ax_nmse.grid(True, linestyle="--", linewidth=0.5)

        ax_ser.set_title("SER Comparison")
        ax_ser.set_ylabel("SER")
        ax_ser.set_yscale("log", base=10)
        ax_ser.yaxis.set_major_locator(LogLocator(base=10.0))
        ax_ser.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_ser.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax_ser.grid(True, linestyle="--", linewidth=0.5, which="both")

        handles0 = [h for h, _ in vel_nmse_lines] + [h for h, _ in lan_nmse_lines]
        labels0 = [lbl for _, lbl in vel_nmse_lines] + [lbl for _, lbl in lan_nmse_lines]
        if handles0:
            ax_nmse.legend(handles0, labels0, loc="best")

        handles_ser = [h for h, _ in vel_ser_lines] + [h for h, _ in lan_ser_lines]
        labels_ser = [lbl for _, lbl in vel_ser_lines] + [lbl for _, lbl in lan_ser_lines]
        if handles_ser:
            ax_ser.legend(handles_ser, labels_ser, loc="best")

    # If axes are provided, render directly into them and return without saving new figs
    if axes is not None:
        ax_nmse, ax_ser = axes
        _render(ax_nmse, ax_ser)
        return

    orientations = tuple(orientations) if orientations else ("horizontal",)
    for orientation in orientations:
        normalized = _normalize_orientation(orientation)
        if normalized == "vertical":
            fig, axes = plt.subplots(2, 1, figsize=(6.0, 7.8), sharex=True)
            ax_nmse, ax_ser = axes
            ax_nmse.set_xlabel("")
            ax_ser.set_xlabel("SNR (dB)")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
            ax_nmse, ax_ser = axes
            ax_nmse.set_xlabel("SNR (dB)")
            ax_ser.set_xlabel("SNR (dB)")

        _render(ax_nmse, ax_ser)
        if normalized == "vertical":
            ser_leg = ax_ser.get_legend()
            if ser_leg is not None:
                ser_leg.remove()
        fig.tight_layout()
        save_path = _orientation_path(out_path, normalized)
        ensure_output_directory(save_path)
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
