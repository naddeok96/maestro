"""Helpers for constructing a shared label space across heterogeneous datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

DEFAULT_SYNONYMS: Dict[str, str] = {
    "aeroplane": "airplane",
    "motorbike": "motorcycle",
    "diningtable": "dining table",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv",
    "pedestrian": "person",
    "cyclist": "person",
}


def _normalise_text(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def normalize(name: str, synonyms: Dict[str, str] | None = None) -> str:
    table = dict(DEFAULT_SYNONYMS)
    if synonyms:
        table.update({k.lower(): v for k, v in synonyms.items()})
    key = _normalise_text(name)
    return table.get(key, key)


def canonical_from_lists(
    small_lists: List[List[str]],
    big_lists: List[List[str]] | None,
    mode: str = "overlap_only",
    synonyms: Dict[str, str] | None = None,
) -> List[str]:
    base: Set[str] = {normalize(name, synonyms) for names in small_lists for name in names}
    if mode == "union_full" and big_lists:
        base.update(normalize(name, synonyms) for names in big_lists for name in names)
    return sorted(base)


def id_map(src_names: Iterable[str], canonical: List[str], synonyms: Dict[str, str] | None = None) -> Dict[int, int]:
    lookup = {normalize(name, synonyms): idx for idx, name in enumerate(canonical)}
    mapping: Dict[int, int] = {}
    for idx, name in enumerate(src_names):
        normalised = normalize(name, synonyms)
        if normalised in lookup:
            mapping[idx] = lookup[normalised]
    return mapping


def dump_canonical(path: Path, canonical: List[str]) -> None:
    path.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
