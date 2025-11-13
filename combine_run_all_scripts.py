#!/usr/bin/env python3
"""Bundle every code/config file touched by scripts/run_all.sh into one Markdown file.

This helper mirrors `combine_repo.py` but automatically detects the files that the
publication pipeline interacts with (Python entrypoints, shell helpers, YAML configs,
etc.).  The resulting Markdown can be shared with another agent without exposing the
entire repository.  It also pulls in the Maestro package pieces that power the
baseline/student pipelines so downstream users see the supporting code.
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".md": "markdown",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".mm": "objectivec",
    ".m": "objectivec",
    ".swift": "swift",
    ".sql": "sql",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".xml": "xml",
    ".toml": "toml",
    ".ini": "ini",
    ".dart": "dart",
    ".lua": "lua",
    ".pl": "perl",
    ".makefile": "make",
    ".mk": "make",
    ".dockerfile": "dockerfile",
    ".gradle": "groovy",
    ".groovy": "groovy",
    ".bat": "batch",
    ".env": "",
    ".txt": "",
    ".conf": "",
    ".cfg": "",
}


BINARY_EXTS_COMMON = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".tar",
    ".jar",
    ".war",
    ".class",
    ".o",
    ".obj",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".iso",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".mp3",
    ".wav",
    ".flac",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
}


ALLOWED_EXTS = (".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".md", ".cfg", ".ini")
PATH_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_\-./]+?\.(?:py|sh|ya?ml|json|txt|md|cfg|ini))(?=$|[^A-Za-z0-9_.-])"
)
DEFAULT_EXTRA_PATTERNS: List[str] = [
    "configs/publication/*.yaml",
    "maestro/__init__.py",
    "maestro/datasets/**/*.py",
    "maestro/students/**/*.py",
    "maestro/utils/__init__.py",
    "maestro/utils/logging.py",
    "maestro/utils/seeding.py",
]


def guess_lang(path: Path) -> str:
    name = path.name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "make"
    ext = path.suffix.lower()
    return LANG_BY_EXT.get(ext, "")


def looks_binary(sample: bytes) -> bool:
    if b"\x00" in sample:
        return True
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    nontext = sample.translate(None, text_chars)
    return len(nontext) / max(1, len(sample)) > 0.30


def is_binary_file(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTS_COMMON:
        return True
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
        return looks_binary(chunk)
    except Exception:
        return True


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.0f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def _token_variants(token: str) -> List[str]:
    base = token.strip()
    candidates = {
        base,
        base.lstrip("./"),
        base.lstrip("-"),
        base.lstrip("./-"),
    }
    return [cand for cand in candidates if cand]


def resolve_reference(
    token: str, run_script: Path, root: Path
) -> Tuple[str, Path] | None:
    seen_paths: set[Path] = set()
    for variant in _token_variants(token):
        path_obj = Path(variant)
        search_paths: Iterable[Path]
        if path_obj.is_absolute():
            search_paths = (path_obj,)
        else:
            search_paths = (
                (root / path_obj),
                (run_script.parent / path_obj),
            )

        for candidate in search_paths:
            try:
                resolved = candidate.resolve()
            except FileNotFoundError:
                continue
            if resolved in seen_paths or not resolved.is_file():
                continue
            seen_paths.add(resolved)
            try:
                rel = resolved.relative_to(root)
            except ValueError:
                continue
            return rel.as_posix(), resolved
    return None


def collect_run_all_files(
    run_script: Path, root: Path
) -> Tuple[Dict[str, Path], List[str]]:
    text = run_script.read_text(encoding="utf-8")
    matches = {m.group("path") for m in PATH_PATTERN.finditer(text)}

    files: Dict[str, Path] = {}
    missing: List[str] = []
    for token in sorted(matches):
        if not token.lower().endswith(tuple(ext.lower() for ext in ALLOWED_EXTS)):
            continue
        resolved = resolve_reference(token, run_script, root)
        if resolved is None:
            missing.append(token)
            continue
        rel, abs_path = resolved
        files[rel] = abs_path

    rel_run_script = run_script.relative_to(root).as_posix()
    files[rel_run_script] = run_script
    return files, missing


def expand_patterns(root: Path, patterns: Sequence[str]) -> Tuple[Dict[str, Path], List[str]]:
    files: Dict[str, Path] = {}
    missing: List[str] = []
    for pattern in patterns:
        matches = [p for p in root.glob(pattern) if p.is_file()]
        if not matches:
            missing.append(pattern)
            continue
        for abs_path in matches:
            rel = abs_path.relative_to(root).as_posix()
            files[rel] = abs_path
    return files, missing


def write_bundle(
    files: Sequence[Tuple[str, Path, int]],
    out_path: Path,
    *,
    header: str,
    root: Path,
    include_binary: str,
    max_file_bytes: int,
    newline_between: int,
) -> Tuple[int, List[str], List[str], List[str]]:
    skipped_large: List[str] = []
    skipped_binary: List[str] = []
    failed_read: List[str] = []
    written = 0
    nl = "\n" * max(1, newline_between)

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{header}\n\n")
        handle.write(f"_Root: `{root}`_\n\n")
        handle.write("<details>\n<summary>Table of contents</summary>\n\n")
        for rel_path, _, _ in files:
            toc_anchor = rel_path.replace("/", "").replace(".", "")
            handle.write(f"- [{rel_path}](#file-{toc_anchor})\n")
        handle.write("\n</details>\n\n")

        for rel_path, abs_path, size in files:
            anchor = f"file-{rel_path.replace('/', '').replace('.', '')}"
            handle.write("---\n\n")
            handle.write(f"## `{rel_path}` <a id=\"{anchor}\"></a>\n\n")
            handle.write(f"- Size: {human_size(size)}\n\n")

            if size > max_file_bytes:
                handle.write(
                    f"> ⚠️ Skipped: file exceeds max-file-bytes ({max_file_bytes}).\n\n"
                )
                skipped_large.append(rel_path)
                handle.write(nl)
                continue

            if is_binary_file(abs_path):
                skipped_binary.append(rel_path)
                if include_binary == "skip":
                    handle.write("> 🧱 Skipped binary file.\n\n")
                elif include_binary == "note":
                    handle.write("> 🧱 Binary file (content omitted).\n\n")
                elif include_binary == "base64":
                    try:
                        data = abs_path.read_bytes()
                        encoded = base64.b64encode(data).decode("ascii")
                        handle.write("```base64\n")
                        handle.write(encoded)
                        handle.write("\n```\n\n")
                    except Exception as exc:
                        handle.write(f"> ❌ Failed to read binary file: {exc}\n\n")
                        failed_read.append(rel_path)
                handle.write(nl)
                continue

            lang = guess_lang(abs_path)
            try:
                text = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    text = abs_path.read_text(encoding="latin-1")
                except Exception as exc:
                    handle.write(f"> ❌ Failed to read text file: {exc}\n\n")
                    failed_read.append(rel_path)
                    handle.write(nl)
                    continue
            except Exception as exc:
                handle.write(f"> ❌ Failed to read file: {exc}\n\n")
                failed_read.append(rel_path)
                handle.write(nl)
                continue

            fence = lang if lang is not None else ""
            handle.write(f"```{fence}\n")
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.write("```\n\n")
            handle.write(nl)
            written += 1

        handle.write("\n---\n\n")
        handle.write("### Summary\n\n")
        handle.write(f"- Files included: **{written}**\n")
        if skipped_large:
            handle.write(f"- Skipped (too large): {len(skipped_large)}\n")
        if skipped_binary:
            handle.write(f"- Binary files encountered: {len(skipped_binary)}\n")
        if failed_read:
            handle.write(f"- Failed to read: {len(failed_read)}\n")

    return written, skipped_large, skipped_binary, failed_read


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine the code paths referenced by scripts/run_all.sh."
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--run-script",
        default="scripts/run_all.sh",
        help="Path to the orchestrating shell script (default: scripts/run_all.sh).",
    )
    parser.add_argument(
        "--out",
        default="combined_run_all_scripts.md",
        help="Output Markdown path (default: combined_run_all_scripts.md).",
    )
    parser.add_argument(
        "--header",
        default="# 🚀 Run-All Pipeline Bundle",
        help="Top-level header text for the Markdown output.",
    )
    parser.add_argument(
        "--include-binary",
        choices=["skip", "note", "base64"],
        default="note",
        help="How to handle binary files (default: note).",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=1_000_000,
        help="Skip files larger than this many bytes (default: 1,000,000).",
    )
    parser.add_argument(
        "--newline-between",
        type=int,
        default=1,
        help="Blank lines between files in the output (default: 1).",
    )
    parser.add_argument(
        "--sort",
        choices=["path", "size"],
        default="path",
        help="Order files by path or size (default: path).",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional glob pattern(s) to include.",
    )
    parser.add_argument(
        "--skip-default-extras",
        action="store_true",
        help="Do not automatically include Maestro datasets/students/utils.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    run_script = (root / args.run_script).resolve()
    if not run_script.is_file():
        raise SystemExit(f"Run script not found: {run_script}")

    files_map, missing_tokens = collect_run_all_files(run_script, root)

    if not args.skip_default_extras:
        default_map, default_missing = expand_patterns(root, DEFAULT_EXTRA_PATTERNS)
        files_map.update(default_map)
    else:
        default_missing = []

    if args.extra:
        extra_map, extra_missing = expand_patterns(root, args.extra)
        files_map.update(extra_map)
    else:
        extra_missing = []

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files: List[Tuple[str, Path, int]] = []
    for rel_path, abs_path in files_map.items():
        if abs_path.resolve() == output_path:
            continue
        try:
            size = abs_path.stat().st_size
        except FileNotFoundError:
            continue
        files.append((rel_path, abs_path, size))

    if not files:
        raise SystemExit("No files collected from run_all references.")

    if args.sort == "size":
        files.sort(key=lambda entry: entry[2])
    else:
        files.sort(key=lambda entry: entry[0])

    written, skipped_large, skipped_binary, failed_read = write_bundle(
        files,
        output_path,
        header=args.header,
        root=root,
        include_binary=args.include_binary,
        max_file_bytes=args.max_file_bytes,
        newline_between=args.newline_between,
    )

    if missing_tokens:
        print(
            f"[combine_run_all_scripts] Warning: could not resolve {len(missing_tokens)} reference(s):"
        )
        for token in missing_tokens:
            print(f"  - {token}")

    if default_missing:
        print(
            f"[combine_run_all_scripts] Warning: no matches for {len(default_missing)} default pattern(s):"
        )
        for pattern in default_missing:
            print(f"  - {pattern}")

    if extra_missing:
        print(
            f"[combine_run_all_scripts] Warning: no matches for {len(extra_missing)} extra pattern(s):"
        )
        for pattern in extra_missing:
            print(f"  - {pattern}")

    print(
        f"✅ wrote {output_path} with {written} files "
        f"(skipped_large={len(skipped_large)}, skipped_binary={len(skipped_binary)}, failed_read={len(failed_read)})"
    )


if __name__ == "__main__":
    main()
