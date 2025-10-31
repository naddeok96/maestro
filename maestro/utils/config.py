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

    def _resolve_yaml_like(p: Path) -> Path:
        """
        Resolve default include entries that may be:
          - relative paths without extension (e.g., '../defaults')
          - directories (e.g., '../defaults/')
          - explicit files (e.g., '../defaults.yaml' or '../defaults.yml')
        We normalize to an existing *.yaml or *.yml file.
        """

        p = p.resolve()
        if p.is_file():
            return p
        # If it's a directory, try defaults.yaml then defaults.yml inside that directory
        if p.is_dir():
            for candidate in ("defaults.yaml", "defaults.yml"):
                candidate_path = (p / candidate).resolve()
                if candidate_path.is_file():
                    return candidate_path
        # If no suffix, try adding .yaml then .yml
        if not p.suffix:
            yaml_p = p.with_suffix(".yaml")
            yml_p = p.with_suffix(".yml")
            if yaml_p.is_file():
                return yaml_p.resolve()
            if yml_p.is_file():
                return yml_p.resolve()
        # Last chance: if it ends with .yaml or .yml but doesn’t exist, raise
        raise FileNotFoundError(f"Could not resolve defaults include: {p}")

    if "defaults" in cfg:
        merged: Dict[str, Any] = {}
        defaults_list = cfg.pop("defaults")
        if not isinstance(defaults_list, (list, tuple)):
            raise TypeError(f"'defaults' must be a list, got: {type(defaults_list)}")
        for default in defaults_list:
            # allow simple strings only (keep behavior simple and predictable)
            if not isinstance(default, str):
                raise TypeError(f"defaults entries must be strings, got: {type(default)} in {path}")
            # resolve relative to current file’s parent
            raw = (path.parent / default)
            # normalize to an actual yaml file
            default_path = _resolve_yaml_like(raw)
            merged = _merge_dicts(merged, load_config(default_path))
        cfg = _merge_dicts(merged, cfg)

    return cfg
