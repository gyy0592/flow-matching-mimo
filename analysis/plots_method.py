"""Method-comparison paper plots: panels / pairwise / multi-symbol grid."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from analysis.csv_io import ensure_output_directory, group_by_symbols
from analysis.plots_common import (
    _filter_by_snr,
    _normalize_orientation,
    _orientation_path,
)
from analysis.styling import (
    METHOD_DISPLAY_NAMES,
    METHOD_STYLE_OVERRIDES,
    _style_for_variant,
)


def plot_method_panels(
    rows: List[Dict[str, float]],
    method: str,
    out_path: Path,
    snr_min: Optional[float],
    snr_max: Optional[float],
    snr_list: Optional[List[float]],
    overlay_methods: Optional[Iterable[str]] = None,
    apply_variant_style: bool = True,
    orientations: Iterable[str] = ("horizontal",),
) -> None:
    grouped = group_by_symbols(rows, method)
    if not grouped and not overlay_methods:
        return

    sorted_symbols = sorted(grouped)
    overlay_methods = list(overlay_methods or [])
    ensure_output_directory(out_path)

    def _render(ax_nmse, ax_ser) -> None:
        for idx, symbols in enumerate(sorted_symbols):
            data = _filter_by_snr(grouped[symbols], snr_min, snr_max, snr_list)
            if not data:
                continue
            snr_vals = [d["snr_db"] for d in data]
            nmse_vals = [d["nmse_db"] for d in data]
            if apply_variant_style:
                style_kwargs = _style_for_variant(method, idx, len(sorted_symbols), {"marker": "o"})
            else:
                style_kwargs = {"marker": "o"}
            ax_nmse.plot(snr_vals, nmse_vals, label=f"D={symbols}", **style_kwargs)

        ax_nmse.set_ylabel("Final NMSE (dB)")
        ax_nmse.set_title(f"{METHOD_DISPLAY_NAMES.get(method, method)} Channel NMSE")
        ax_nmse.grid(True, linestyle="--", linewidth=0.5)
        if ax_nmse.lines:
            ax_nmse.legend()

        for idx, symbols in enumerate(sorted_symbols):
            data = _filter_by_snr(grouped[symbols], snr_min, snr_max, snr_list)
            ser_points = [
                (d["snr_db"], d["ser"])
                for d in data
                if not math.isnan(d.get("ser", math.nan))
            ]
            if not ser_points:
                continue
            snr_vals, ser_vals = zip(*ser_points)
            if apply_variant_style:
                style_kwargs = _style_for_variant(method, idx, len(sorted_symbols), {"marker": "o"})
            else:
                style_kwargs = {"marker": "o"}
            ax_ser.plot(snr_vals, ser_vals, label=f"D={symbols}", **style_kwargs)

        ax_ser.set_ylabel("SER")
        ax_ser.set_title(f"{METHOD_DISPLAY_NAMES.get(method, method)} SER")
        ax_ser.grid(True, linestyle="--", linewidth=0.5, which="both")
        ax_ser.set_yscale("log", base=10)
        ax_ser.yaxis.set_major_locator(LogLocator(base=10.0))
        ax_ser.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_ser.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        if ax_ser.lines:
            ax_ser.legend()

        if overlay_methods:
            for overlay in overlay_methods:
                entries = [
                    row
                    for row in rows
                    if row["method"] == overlay
                ]
                if not entries:
                    continue
                entries.sort(key=lambda r: r["snr_db"])
                filtered = _filter_by_snr(entries, snr_min, snr_max, snr_list)
                if filtered:
                    snr_vals = [e["snr_db"] for e in filtered]
                    nmse_vals = [e["nmse_db"] for e in filtered]
                    style_kwargs = _style_for_variant(overlay, 0, 1, {"linestyle": "--", "linewidth": 1.3})
                    ax_nmse.plot(
                        snr_vals,
                        nmse_vals,
                        label=METHOD_DISPLAY_NAMES.get(overlay, overlay),
                        **style_kwargs,
                    )
                ser_filtered = [
                    e
                    for e in filtered
                    if not math.isnan(e.get("ser", math.nan))
                ]
                if ser_filtered:
                    snr_vals = [e["snr_db"] for e in ser_filtered]
                    ser_vals = [e["ser"] for e in ser_filtered]
                    style_kwargs = _style_for_variant(overlay, 0, 1, {"linestyle": "--", "linewidth": 1.3})
                    ax_ser.plot(
                        snr_vals,
                        ser_vals,
                        label=METHOD_DISPLAY_NAMES.get(overlay, overlay),
                        **style_kwargs,
                    )
            if ax_nmse.lines:
                ax_nmse.legend()
            if ax_ser.lines:
                ax_ser.legend()

    for orientation in orientations:
        normalized = _normalize_orientation(orientation)
        if normalized == "vertical":
            fig, axes = plt.subplots(2, 1, figsize=(6.0, 7.8), sharex=True)
            ax_nmse, ax_ser = axes
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
            ax_nmse, ax_ser = axes

        _render(ax_nmse, ax_ser)

        if normalized == "vertical":
            ax_nmse.set_xlabel("")
            ax_ser.set_xlabel("SNR (dB)")
            ser_legend = ax_ser.get_legend()
            if ser_legend is not None:
                ser_legend.remove()
        else:
            ax_nmse.set_xlabel("SNR (dB)")
            ax_ser.set_xlabel("SNR (dB)")

        fig.tight_layout()
        save_path = _orientation_path(out_path, normalized)
        ensure_output_directory(save_path)
        fig.savefig(save_path, dpi=220)
        plt.close(fig)


def plot_method_comparison(
    rows: List[Dict[str, float]],
    target_symbols: int,
    methods: Iterable[str],
    out_path: Path,
    snr_min: Optional[float],
    snr_max: Optional[float],
    snr_list: Optional[List[float]],
    include_ser: bool = True,
    orientations: Iterable[str] = ("horizontal",),
    axes: Optional[Tuple[plt.Axes, Optional[plt.Axes]]] = None,
) -> None:
    relevant = defaultdict(list)
    methods = list(methods)
    for row in rows:
        method = row["method"]
        if method not in methods:
            continue
        symbols_value = row["symbols"]
        if math.isnan(symbols_value):
            relevant[method].append(row)
            continue
        if int(round(symbols_value)) != target_symbols:
            continue
        relevant[method].append(row)
    for method_rows in relevant.values():
        method_rows.sort(key=lambda r: r["snr_db"])
    if len(relevant) < 2:
        return

    def _render(ax_nmse_local, ax_ser_local: Optional[plt.Axes]) -> None:
        for method in methods:
            if method not in relevant:
                continue
            data = _filter_by_snr(relevant[method], snr_min, snr_max, snr_list)
            if not data:
                continue
            snr = [d["snr_db"] for d in data]
            nmse = [d["nmse_db"] for d in data]
            style_kwargs = dict(marker="o")
            style_kwargs.update(METHOD_STYLE_OVERRIDES.get(method, {}))
            ax_nmse_local.plot(
                snr,
                nmse,
                label=METHOD_DISPLAY_NAMES.get(method, method),
                **style_kwargs,
            )

        ax_nmse_local.set_title(f"Channel NMSE (D={target_symbols})")
        ax_nmse_local.set_ylabel("Final NMSE (dB)")
        ax_nmse_local.grid(True, linestyle="--", linewidth=0.5)
        handles, _ = ax_nmse_local.get_legend_handles_labels()
        if handles:
            ax_nmse_local.legend(loc="best")

        if include_ser and ax_ser_local is not None:
            for method in methods:
                if method not in relevant:
                    continue
                data = [
                    d
                    for d in _filter_by_snr(relevant[method], snr_min, snr_max, snr_list)
                    if not math.isnan(d.get("ser", math.nan))
                ]
                if not data:
                    continue
                snr_vals = [d["snr_db"] for d in data]
                ser_vals = [d["ser"] for d in data]
                style_kwargs = dict(marker="o")
                style_kwargs.update(METHOD_STYLE_OVERRIDES.get(method, {}))
                ax_ser_local.plot(
                    snr_vals,
                    ser_vals,
                    label=METHOD_DISPLAY_NAMES.get(method, method),
                    **style_kwargs,
                )

            ax_ser_local.set_title(f"SER (D={target_symbols})")
            ax_ser_local.set_ylabel("SER")
            ax_ser_local.set_yscale("log", base=10)
            ax_ser_local.yaxis.set_major_locator(LogLocator(base=10.0))
            ax_ser_local.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax_ser_local.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax_ser_local.grid(True, linestyle="--", linewidth=0.5, which="both")
            handles_ser, _ = ax_ser_local.get_legend_handles_labels()
            if handles_ser and not handles:
                ax_ser_local.legend(loc="best")

    # If axes are provided, render directly into them and return (no saving here)
    if axes is not None:
        ax_nmse_local = axes[0]
        ax_ser_local = axes[1] if include_ser and len(axes) > 1 else None
        _render(ax_nmse_local, ax_ser_local)
        # Apply labels consistent with standalone mode
        ax_nmse_local.set_xlabel("SNR (dB)")
        if include_ser and ax_ser_local is not None:
            ax_ser_local.set_xlabel("SNR (dB)")
        return

    ensure_output_directory(out_path)

    if include_ser:
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
                ser_legend = ax_ser.get_legend()
                if ser_legend is not None:
                    ser_legend.remove()
            fig.tight_layout()
            save_path = _orientation_path(out_path, normalized)
            ensure_output_directory(save_path)
            fig.savefig(save_path, dpi=220)
            plt.close(fig)
    else:
        fig, ax_nmse = plt.subplots(figsize=(5.5, 4.2), sharex=False)
        _render(ax_nmse, None)
        ax_nmse.set_xlabel("SNR (dB)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)


# --- B-2 split helpers for plot_method_comparison_grid ----------------------

def _grid_setup(
    rows: List[Dict[str, float]],
    methods: Iterable[str],
    out_path: Path,
):
    """Compute symbol set, create figure+axes, build global baseline index.

    Returns (symbols_set, fig, axes, global_baselines) or None if nothing to plot.
    """
    symbols_set = sorted(
        {
            int(round(row["symbols"]))
            for row in rows
            if row["method"] in methods and not math.isnan(row["symbols"])
        }
    )
    if not symbols_set:
        return None

    ensure_output_directory(out_path)
    n_rows = len(symbols_set)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10, 3.2 * n_rows),
        sharex=False,
    )
    axes = np.atleast_2d(axes)

    global_baselines = {
        method: [
            entry
            for entry in rows
            if entry["method"] == method and math.isnan(entry["symbols"])
        ]
        for method in methods
    }
    return symbols_set, fig, axes, global_baselines


def _grid_panel(
    rows: List[Dict[str, float]],
    methods: Iterable[str],
    symbols: int,
    row_idx: int,
    axes,
    snr_min: Optional[float],
    snr_max: Optional[float],
    snr_list: Optional[List[float]],
    global_baselines: Dict[str, List[Dict[str, float]]],
    apply_variant_style: bool,
) -> None:
    """Draw the NMSE+SER panel pair for a single symbol-length row."""
    symbol_rows = {
        method: [
            entry
            for entry in rows
            if entry["method"] == method
            and not math.isnan(entry["symbols"])
            and int(round(entry["symbols"])) == symbols
        ]
        for method in methods
    }
    ax_nmse = axes[row_idx, 0]
    ax_ser = axes[row_idx, 1]

    for method in methods:
        entries = _filter_by_snr(symbol_rows.get(method, []), snr_min, snr_max, snr_list)
        if not entries:
            continue
        snr_vals = [entry["snr_db"] for entry in entries]
        nmse_vals = [entry["nmse_db"] for entry in entries]
        if apply_variant_style:
            style_kwargs = _style_for_variant(method, 0, 1, {"marker": "o"})
        else:
            style_kwargs = {"marker": "o"}
        ax_nmse.plot(
            snr_vals,
            nmse_vals,
            label=METHOD_DISPLAY_NAMES.get(method, method),
            **style_kwargs,
        )

    ax_nmse.set_ylabel("Final NMSE (dB)")
    ax_nmse.set_title(f"NMSE (D={symbols})")
    ax_nmse.grid(True, linestyle="--", linewidth=0.5)
    if row_idx == 0 and ax_nmse.lines:
        ax_nmse.legend(loc="best")

    for method in methods:
        entries = [
            entry
            for entry in _filter_by_snr(symbol_rows.get(method, []), snr_min, snr_max, snr_list)
            if not math.isnan(entry.get("ser", math.nan))
        ]
        if not entries:
            continue
        snr_vals = [entry["snr_db"] for entry in entries]
        ser_vals = [entry["ser"] for entry in entries]
        if apply_variant_style:
            style_kwargs = _style_for_variant(method, 0, 1, {"marker": "o"})
        else:
            style_kwargs = {"marker": "o"}
        ax_ser.plot(
            snr_vals,
            ser_vals,
            label=METHOD_DISPLAY_NAMES.get(method, method),
            **style_kwargs,
        )

    # Overlay global baselines (e.g., LMMSE) with dashed lines.
    for method, baseline_entries in global_baselines.items():
        if not baseline_entries:
            continue
        filtered = _filter_by_snr(baseline_entries, snr_min, snr_max, snr_list)
        if filtered:
            snr_vals = [entry["snr_db"] for entry in filtered]
            nmse_vals = [entry["nmse_db"] for entry in filtered]
            if apply_variant_style:
                style_kwargs = _style_for_variant(method, 0, 1, {"linestyle": "--", "linewidth": 1.3})
            else:
                style_kwargs = {"linestyle": "--", "linewidth": 1.3}
            ax_nmse.plot(
                snr_vals,
                nmse_vals,
                label=METHOD_DISPLAY_NAMES.get(method, method),
                **style_kwargs,
            )
        filtered_ser = [
            entry
            for entry in filtered
            if not math.isnan(entry.get("ser", math.nan))
        ]
        if filtered_ser:
            snr_vals = [entry["snr_db"] for entry in filtered_ser]
            ser_vals = [entry["ser"] for entry in filtered_ser]
            if apply_variant_style:
                style_kwargs = _style_for_variant(method, 0, 1, {"linestyle": "--", "linewidth": 1.3})
            else:
                style_kwargs = {"linestyle": "--", "linewidth": 1.3}
            ax_ser.plot(
                snr_vals,
                ser_vals,
                label=METHOD_DISPLAY_NAMES.get(method, method),
                **style_kwargs,
            )

    ax_ser.set_ylabel("SER")
    ax_ser.set_title(f"SER (D={symbols})")
    ax_ser.set_yscale("log", base=10)
    ax_ser.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_ser.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_ser.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_ser.grid(True, linestyle="--", linewidth=0.5, which="both")
    if row_idx == 0 and ax_ser.lines and not ax_nmse.lines:
        ax_ser.legend(loc="best")


def _grid_save(fig, axes, out_path: Path) -> None:
    """Apply x-labels, suptitle, tight_layout, save and close."""
    axes[-1, 0].set_xlabel("SNR (dB)")
    axes[-1, 1].set_xlabel("SNR (dB)")
    fig.suptitle("Method Comparison Across Symbol Lengths", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_method_comparison_grid(
    rows: List[Dict[str, float]],
    methods: Iterable[str],
    out_path: Path,
    snr_min: Optional[float],
    snr_max: Optional[float],
    snr_list: Optional[List[float]],
    apply_variant_style: bool = True,
) -> None:
    methods = list(methods)
    setup = _grid_setup(rows, methods, out_path)
    if setup is None:
        return
    symbols_set, fig, axes, global_baselines = setup

    for row_idx, symbols in enumerate(symbols_set):
        _grid_panel(
            rows,
            methods,
            symbols,
            row_idx,
            axes,
            snr_min,
            snr_max,
            snr_list,
            global_baselines,
            apply_variant_style,
        )

    _grid_save(fig, axes, out_path)

    # NOTE: The block below is a historical duplicate render kept verbatim to
    # preserve byte-equal paper figure output. Although it overwrites the
    # output file with an identical figure, removing it changes the figure
    # save sequence and is therefore considered a behavior change. Do not
    # delete without paper-output diff verification.
    ensure_output_directory(out_path)
    n_rows = len(symbols_set)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10, 3.2 * n_rows),
        sharex=False,
    )
    axes = np.atleast_2d(axes)

    global_baselines = {
        method: [
            entry
            for entry in rows
            if entry["method"] == method and math.isnan(entry["symbols"])
        ]
        for method in methods
    }

    for row_idx, symbols in enumerate(symbols_set):
        symbol_rows = {
            method: [
                entry
                for entry in rows
                if entry["method"] == method
                and not math.isnan(entry["symbols"])
                and int(round(entry["symbols"])) == symbols
            ]
            for method in methods
        }
        ax_nmse = axes[row_idx, 0]
        ax_ser = axes[row_idx, 1]

        for method in methods:
            entries = _filter_by_snr(symbol_rows.get(method, []), snr_min, snr_max, snr_list)
            if not entries:
                continue
            snr_vals = [entry["snr_db"] for entry in entries]
            nmse_vals = [entry["nmse_db"] for entry in entries]
            if apply_variant_style:
                style_kwargs = _style_for_variant(method, 0, 1, {"marker": "o"})
            else:
                style_kwargs = {"marker": "o"}
            ax_nmse.plot(
                snr_vals,
                nmse_vals,
                label=METHOD_DISPLAY_NAMES.get(method, method),
                **style_kwargs,
            )

        ax_nmse.set_ylabel("Final NMSE (dB)")
        ax_nmse.set_title(f"NMSE (D={symbols})")
        ax_nmse.grid(True, linestyle="--", linewidth=0.5)
        if row_idx == 0 and ax_nmse.lines:
            ax_nmse.legend(loc="best")

        for method in methods:
            entries = [
                entry
                for entry in _filter_by_snr(symbol_rows.get(method, []), snr_min, snr_max, snr_list)
                if not math.isnan(entry.get("ser", math.nan))
            ]
            if not entries:
                continue
            snr_vals = [entry["snr_db"] for entry in entries]
            ser_vals = [entry["ser"] for entry in entries]
            if apply_variant_style:
                style_kwargs = _style_for_variant(method, 0, 1, {"marker": "o"})
            else:
                style_kwargs = {"marker": "o"}
            ax_ser.plot(
                snr_vals,
                ser_vals,
                label=METHOD_DISPLAY_NAMES.get(method, method),
                **style_kwargs,
            )

        # Overlay global baselines (e.g., LMMSE) with dashed lines.
        for method, baseline_entries in global_baselines.items():
            if not baseline_entries:
                continue
            filtered = _filter_by_snr(baseline_entries, snr_min, snr_max, snr_list)
            if filtered:
                snr_vals = [entry["snr_db"] for entry in filtered]
                nmse_vals = [entry["nmse_db"] for entry in filtered]
                style_kwargs = _style_for_variant(method, 0, 1, {"linestyle": "--", "linewidth": 1.3})
                ax_nmse.plot(
                    snr_vals,
                    nmse_vals,
                    label=METHOD_DISPLAY_NAMES.get(method, method),
                    **style_kwargs,
                )
            filtered_ser = [
                entry
                for entry in filtered
                if not math.isnan(entry.get("ser", math.nan))
            ]
            if filtered_ser:
                snr_vals = [entry["snr_db"] for entry in filtered_ser]
                ser_vals = [entry["ser"] for entry in filtered_ser]
                style_kwargs = _style_for_variant(method, 0, 1, {"linestyle": "--", "linewidth": 1.3})
                ax_ser.plot(
                    snr_vals,
                    ser_vals,
                    label=METHOD_DISPLAY_NAMES.get(method, method),
                    **style_kwargs,
                )

        ax_ser.set_ylabel("SER")
        ax_ser.set_title(f"SER (D={symbols})")
        ax_ser.set_yscale("log", base=10)
        ax_ser.yaxis.set_major_locator(LogLocator(base=10.0))
        ax_ser.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_ser.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax_ser.grid(True, linestyle="--", linewidth=0.5, which="both")
        if row_idx == 0 and ax_ser.lines and not ax_nmse.lines:
            ax_ser.legend(loc="best")

    axes[-1, 0].set_xlabel("SNR (dB)")
    axes[-1, 1].set_xlabel("SNR (dB)")

    fig.suptitle("Method Comparison Across Symbol Lengths", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
