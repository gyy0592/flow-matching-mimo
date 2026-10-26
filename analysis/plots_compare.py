"""Triptych compare paper plots."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

from analysis.csv_io import ensure_output_directory
from analysis.plots_method import plot_method_comparison
from analysis.plots_velocity import plot_velocity_vs_langevin_symbols


def compose_compare_from_pngs(output_dir: Path) -> None:
    """Compose paper_compare.png from existing method-comparison panels.

    Use the method comparison on the left and the two flow/Langevin panels
    in the middle and right. Missing source images leave composition unchanged.
    """
    try:
        nmse_path = output_dir / "paper_method_comparison.png"
        vel_h_path = output_dir / "horizontal" / "paper_velocity_vs_langevin.png"
        if not (nmse_path.exists() and vel_h_path.exists()):
            print(f"[warn] compose_compare_from_pngs: missing source images: {nmse_path} or {vel_h_path}")
            return
        img_left = plt.imread(str(nmse_path))
        img_vel = plt.imread(str(vel_h_path))
        h, w = img_vel.shape[:2]
        mid = w // 2
        img_mid = img_vel[:, :mid]
        img_right = img_vel[:, mid:]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        axes[0].imshow(img_left)
        axes[1].imshow(img_mid)
        axes[2].imshow(img_right)
        for ax in axes:
            ax.axis('off')
        out_path = output_dir / "paper_compare.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"[compose] Wrote {out_path}")
    except Exception as e:
        print(f"[warn] compose_compare_from_pngs failed: {e}")


def plot_compare_triptych_reuse_code(
    rows: List[Dict[str, float]],
    velocity_method: str,
    langevin_method: str,
    comparison_methods: Iterable[str],
    target_symbols: int,
    out_path: Path,
    snr_min: Optional[float],
    snr_max: Optional[float],
    snr_list: Optional[List[float]],
) -> None:
    """Assemble 1x3 by reusing the exact plotting code paths used by
    plot_method_comparison (NMSE only) and plot_velocity_vs_langevin_symbols.
    Legends retain the positions and ordering of the individual plots."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    # Left: method comparison NMSE (same call path: include_ser=False)
    plot_method_comparison(
        rows,
        target_symbols,
        list(comparison_methods),
        out_path,  # unused when axes provided
        snr_min,
        snr_max,
        snr_list,
        include_ser=False,
        axes=(axes[0], None),
    )

    # Middle & Right: velocity vs Langevin (same call path)
    plot_velocity_vs_langevin_symbols(
        rows,
        velocity_method,
        langevin_method,
        symbols_list=[10, 20, 50, 100],
        out_path=out_path,  # unused when axes provided
        snr_min=snr_min,
        snr_max=snr_max,
        snr_list=snr_list,
        orientations=("horizontal",),
        axes=(axes[1], axes[2]),
    )

    ensure_output_directory(out_path)
    fig.savefig(out_path, dpi=220)
    # Also save EPS for publication
    eps_path = out_path.with_suffix('.eps')
    fig.savefig(eps_path, format='eps')
    plt.close(fig)
