"""Load optional defaults from configs/runtime.yaml.

Missing configuration files use built-in defaults. Explicit command-line
arguments take precedence over the defaults returned by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Defaults used when the optional runtime YAML is absent.
_BUILTIN_DEFAULTS: Dict[str, Any] = {
    "seed": 4321,
    "results_dir": "./results_velocity_seed4321",
    "baseline_dir": "./results_barry_seed4321",
    "default_step_subdir": "steps400",
    "num_realizations": 5,
    "batch_size_x_list": [50],
    "pilots_list": [10, 20, 50, 100],
    "device": "cuda",
}


def _default_yaml_path() -> Path:
    """Resolve configs/runtime.yaml relative to this module's project root."""
    return Path(__file__).resolve().parent.parent / "configs" / "runtime.yaml"


def load_runtime_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Merge optional YAML values over built-in defaults.

    Use configs/runtime.yaml when path is omitted. Return defaults when the
    file is absent; report parsing or mapping errors with ValueError.
    """
    cfg: Dict[str, Any] = dict(_BUILTIN_DEFAULTS)  # Copy defaults before merging.
    yaml_path = Path(path) if path is not None else _default_yaml_path()

    if not yaml_path.is_file():
        return cfg  # Use defaults when the optional file is absent.

    try:
        import yaml  # Import only when a YAML file needs to be loaded.
    except ImportError as exc:
        raise ValueError(
            f"Cannot load {yaml_path}: install PyYAML with pip install pyyaml"
        ) from exc

    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise ValueError(f"Cannot parse {yaml_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(
            f"{yaml_path} must contain a mapping, got {type(loaded).__name__}"
        )

    cfg.update(loaded)
    return cfg


__all__ = ["load_runtime_config"]
