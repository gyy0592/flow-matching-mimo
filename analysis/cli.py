"""Command-line entry point for paper-style analysis."""

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from analysis.csv_io import _infer_tx_antennas, write_csv
from analysis.plots_compare import plot_compare_triptych_reuse_code
from analysis.plots_method import (
    plot_method_comparison,
    plot_method_comparison_grid,
    plot_method_panels,
)
from analysis.plots_velocity import plot_velocity_vs_langevin_symbols


SCRIPT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_relative_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (SCRIPT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def parse_method_args(extra_args: List[str]) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for value in extra_args:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got '{value}'")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        resolved = _resolve_relative_path(path)
        if resolved is None:
            continue
        mapping.setdefault(name, []).append(resolved)
    return mapping


def _extract_step_suffix(directory: Optional[str]) -> Optional[str]:
    if not directory:
        return None
    last_component = Path(directory).name
    tokens = re.findall(r"\d+", last_component)
    if not tokens:
        return None
    return "_" + "_".join(tokens)


def _apply_suffix(path: Path, suffix: Optional[str]) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSV summaries and paper-style plots."
    )
    default_velocity_dir = SCRIPT_ROOT / "results_velocity_seed4321" / os.environ.get(
        "VELOCITY_STEPS_SUBDIR", "steps200"
    )
    parser.add_argument("--velocity-dir", default=str(default_velocity_dir))
    parser.add_argument(
        "--velocity-method-name",
        default="velocity",
        help="Method key to associate with the primary velocity directory.",
    )
    parser.add_argument(
        "--velocity-steps-subdir",
        default=None,
        help="Convenience flag to set velocity directory to results_velocity_seed4321/<subdir>.",
    )
    parser.add_argument("--baseline-dir", default=str(SCRIPT_ROOT / "results_barry_seed4321"))
    parser.add_argument(
        "--lmmse-dir",
        default=None,
        help="Directory containing LMMSE baseline .pt files (e.g., results/results_ml_baseline_3gpp_3gpp).",
    )
    parser.add_argument(
        "--single-langevin-dir",
        default=None,
        help="Directory containing single-step Langevin baseline .pt files.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional method entries in the form name=path/to/results_dir",
    )
    parser.add_argument("--snr-min", type=float, default=None, help="Minimum SNR to show in plots (CSV always stores all points).")
    parser.add_argument("--snr-max", type=float, default=None, help="Maximum SNR to show in plots (CSV always stores all points).")
    parser.add_argument(
        "--plot-snr",
        type=float,
        nargs="+",
        default=None,
        help="Explicit SNR values to plot. Overrides --snr-min/--snr-max when provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_ROOT / "output"),
        help="Directory where plots (and default CSV) will be written.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Full path to the CSV summary file. Defaults to <output-dir>/paper_metrics_summary.csv.",
    )
    parser.add_argument(
        "--comparison-symbols",
        type=int,
        default=50,
        help="Data length D to use in the method comparison plot.",
    )
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--comparison-methods",
        nargs="*",
        default=["velocity", "langevin", "single_langevin", "lmmse"],
        help="Order of methods to show in the comparison plot.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip generating method comparison plots.",
    )
    args = parser.parse_args()

    output_dir = _resolve_relative_path(args.output_dir)
    if output_dir is None:
        output_dir = (SCRIPT_ROOT / "output").resolve()
    args.output_dir = output_dir

    if args.velocity_steps_subdir:
        step_dir = Path(args.velocity_steps_subdir).expanduser()
        if step_dir.is_absolute():
            args.velocity_dir = step_dir.resolve()
        else:
            args.velocity_dir = (SCRIPT_ROOT / "results_velocity_seed4321" / step_dir).resolve()
    else:
        resolved_velocity = _resolve_relative_path(args.velocity_dir)
        args.velocity_dir = resolved_velocity

    args.baseline_dir = _resolve_relative_path(args.baseline_dir)
    args.lmmse_dir = _resolve_relative_path(args.lmmse_dir)
    args.single_langevin_dir = _resolve_relative_path(args.single_langevin_dir)

    if args.csv_path:
        args.csv_path = _resolve_relative_path(args.csv_path)
    else:
        args.csv_path = output_dir / "paper_metrics_summary.csv"

    method_dirs: Dict[str, List[Path]] = {}
    primary_velocity_name: Optional[str] = None
    if args.velocity_dir:
        primary_velocity_name = args.velocity_method_name
        method_dirs[primary_velocity_name] = [Path(args.velocity_dir)]
    if args.baseline_dir:
        method_dirs["langevin"] = [Path(args.baseline_dir)]
    if args.single_langevin_dir:
        method_dirs["single_langevin"] = [Path(args.single_langevin_dir)]
    if args.lmmse_dir:
        method_dirs["lmmse"] = [Path(args.lmmse_dir)]
    method_dirs.update(parse_method_args(args.extra))

    velocity_suffix = _extract_step_suffix(args.velocity_dir)
    overlay_methods = []
    if "lmmse" in method_dirs:
        overlay_methods.append("lmmse")

    if not args.write_csv and not args.plot:
        raise RuntimeError(
            "No action requested. Use --write-csv, --plot, or both to generate fresh outputs."
        )

    tx_antennas = _infer_tx_antennas(method_dirs)
    snr_shift = 10.0 * math.log10(tx_antennas) if tx_antennas > 0 else 0.0

    rows: List[Dict[str, float]] = write_csv(args.csv_path, method_dirs, snr_shift)
    if not rows:
        raise RuntimeError(
            "No data found while writing CSV. Verify the result directories contain .pt files."
        )

    if args.plot:
        overlay_for_velocity: List[str] = []
        if primary_velocity_name:
            plot_method_panels(
                rows,
                primary_velocity_name,
                _apply_suffix(args.output_dir / "paper_velocity_metrics.png", velocity_suffix),
                args.snr_min,
                args.snr_max,
                args.plot_snr,
                overlay_methods=overlay_for_velocity,
                apply_variant_style=False,
                orientations=("horizontal", "vertical"),
            )
        plot_method_panels(
            rows,
            "langevin",
            args.output_dir / "paper_langevin_metrics.png",
            args.snr_min,
            args.snr_max,
            args.plot_snr,
            overlay_methods=overlay_methods,
            apply_variant_style=False,
            orientations=("horizontal", "vertical"),
        )
        if "single_langevin" in method_dirs:
            plot_method_panels(
                rows,
                "single_langevin",
                args.output_dir / "paper_single_langevin_metrics.png",
                args.snr_min,
                args.snr_max,
                args.plot_snr,
                overlay_methods=overlay_methods,
                orientations=("horizontal", "vertical"),
            )
        if not args.skip_comparison:
            comparison_methods = [
                method for method in args.comparison_methods if method in method_dirs
            ]
            if "velocity_steps200" in method_dirs and "velocity_steps200" not in comparison_methods:
                comparison_methods.insert(0, "velocity_steps200")
            if primary_velocity_name and primary_velocity_name not in comparison_methods:
                comparison_methods = [primary_velocity_name] + comparison_methods
            nmse_only_methods = [
                method
                for method in comparison_methods
                if method in method_dirs
            ]
            grid_methods: List[str] = []
            if primary_velocity_name and primary_velocity_name in method_dirs:
                grid_methods.append(primary_velocity_name)
            if "velocity_steps200" in method_dirs:
                grid_methods.append("velocity_steps200")
            if "langevin" in method_dirs:
                grid_methods.append("langevin")
            # Plot NMSE comparison (relaxed: allow even 1 method)
            if len(nmse_only_methods) >= 1:
                print(f"Plotting NMSE comparison for {len(nmse_only_methods)} method(s)...")
                plot_method_comparison(
                    rows,
                    args.comparison_symbols,
                    nmse_only_methods,
                    args.output_dir / "paper_method_comparison.png",
                    args.snr_min,
                    args.snr_max,
                    args.plot_snr,
                    include_ser=False,
                )
            else:
                print(
                    "[warn] Skipping paper_method_comparison plot because no methods were available."
                )

            # Plot grid comparison (relaxed: allow even 1 method)
            if len(grid_methods) >= 1:
                plot_method_comparison_grid(
                    rows,
                    grid_methods,
                    args.output_dir / "paper_method_comparison_grid.png",
                    args.snr_min,
                    args.snr_max,
                    args.plot_snr,
                )
            else:
                print(
                    "[warn] Skipping paper_method_comparison_grid plot because primary velocity or Langevin data is missing."
                )

            velocity_compare_method = None
            if "velocity_steps400" in method_dirs:
                velocity_compare_method = "velocity_steps400"
            elif primary_velocity_name == "velocity_steps400":
                velocity_compare_method = primary_velocity_name

            # Render flow-matching results when comparison data is absent.
            if velocity_compare_method:
                langevin_method = "langevin" if "langevin" in method_dirs else None

                if langevin_method:
                    # Render the comparison when Langevin data is available.
                    plot_velocity_vs_langevin_symbols(
                        rows,
                        velocity_compare_method,
                        langevin_method,
                        symbols_list=[10, 20, 50, 100],
                        out_path=args.output_dir / "paper_velocity_vs_langevin.png",
                        snr_min=args.snr_min,
                        snr_max=args.snr_max,
                        snr_list=args.plot_snr,
                        orientations=("horizontal", "vertical"),
                    )

                # Compose the three-panel figure with the available method data.
                try:
                    plot_compare_triptych_reuse_code(
                        rows,
                        velocity_compare_method,
                        langevin_method or velocity_compare_method,  # Use available method data.
                        nmse_only_methods if nmse_only_methods else [velocity_compare_method],
                        args.comparison_symbols,
                        args.output_dir / "paper_compare.png",
                        args.snr_min,
                        args.snr_max,
                        args.plot_snr,
                    )
                except Exception as e:
                    print(f"[warn] Failed to build paper_compare.png: {e}")
            else:
                print(
                    "[warn] Skipping velocity vs Langevin comparison plot because velocity_steps400 data is unavailable."
                )


if __name__ == "__main__":
    main()
