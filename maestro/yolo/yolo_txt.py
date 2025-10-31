"""Utilities for manipulating YOLO label text files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def rewrite_label(src_txt: Path, dst_txt: Path, id_map: Dict[int, int]) -> bool:
    """Rewrite ``src_txt`` replacing class identifiers according to ``id_map``.

    Parameters
    ----------
    src_txt:
        Path to the original YOLO label file.
    dst_txt:
        Destination file that will receive the rewritten labels.
    id_map:
        Mapping from source class identifiers to canonical identifiers.

    Returns
    -------
    bool
        ``True`` if at least one bounding box was retained, ``False`` otherwise.
    """

    if not src_txt.exists():
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        dst_txt.write_text("", encoding="utf-8")
        return False

    kept: list[str] = []
    for raw in src_txt.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        if cid not in id_map:
            continue
        parts[0] = str(id_map[cid])
        kept.append(" ".join(parts))

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(kept), encoding="utf-8")
    return bool(kept)
