"""Color/style constants and helpers for paper-style plots."""

from typing import Dict, Optional

from colorsys import hls_to_rgb, rgb_to_hls
from matplotlib.colors import to_hex, to_rgb


METHOD_DISPLAY_NAMES = {
    "velocity": "Velocity",
    "velocity_steps200": "OT Flow (200 steps)",
    "velocity_steps400": "OT Flow (400 steps)",
    "langevin": "Langevin",
    "lmmse": "LMMSE",
    "single_langevin": "Single Langevin",
}

METHOD_STYLE_OVERRIDES = {
    "velocity": {"color": "#1C4BAA"},
    "velocity_steps200": {"color": "#6CB4FF", "linestyle": "-", "marker": "o"},
    "velocity_steps400": {"color": "#1C4BAA", "linestyle": "-", "marker": "o"},
    "single_langevin": {
        "color": "#D62728",
        "linestyle": "--",
        "marker": "o",
        "markerfacecolor": "none",
    },
    "langevin": {
        "color": "#FF7F0E",
        "linestyle": "--",
        "marker": "s",
        "markerfacecolor": "none",
    },
    "lmmse": {"color": "#2CA02C", "linestyle": "-", "marker": "s"},
}

VELOCITY_SYMBOL_COLORS = {
    10: "#2FD7C4",
    20: "#7ABEFF",
    50: "#1D6FE4",
    100: "#0C2E6E",
}

LANGEVIN_SYMBOL_COLORS = {
    10: "#FFD449",
    20: "#FF9F1C",
    50: "#FF6B1A",
    100: "#C1121F",
}


def _adjust_color_lightness(color: str, lighten_amount: float) -> str:
    """Lighten a hex color by the given amount (0=no change, 1=white)."""
    r, g, b = to_rgb(color)
    h, l, s = rgb_to_hls(r, g, b)
    lighten_amount = max(0.0, min(1.0, lighten_amount))
    l = l + (1.0 - l) * lighten_amount
    l = max(0.0, min(1.0, l))
    r_new, g_new, b_new = hls_to_rgb(h, l, s)
    return to_hex((r_new, g_new, b_new))


def _style_for_variant(
    method: str,
    variant_index: int,
    total_variants: int,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    style = dict(METHOD_STYLE_OVERRIDES.get(method, {}))
    if extra:
        for key, value in extra.items():
            style.setdefault(key, value)
    if "color" in style and total_variants > 1:
        step = variant_index / max(total_variants - 1, 1)
        style["color"] = _adjust_color_lightness(style["color"], 0.25 * step)
    return style
