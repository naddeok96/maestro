"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    if "defaults" in cfg:
        merged: Dict[str, Any] = {}
        for default in cfg.pop("defaults"):
            default_path = (path.parent / default).resolve()
            merged = _merge_dicts(merged, load_config(default_path))
        cfg = _merge_dicts(merged, cfg)
    return cfg
