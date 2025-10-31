#!/usr/bin/env python
"""Generate publication tables with statistical significance estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from maestro.stats import paired_bootstrap, permutation_test


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_ci(mean: float, ci: Tuple[float, float]) -> str:
    return f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _summarise_metric(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    summary = df.groupby(list(group_cols))
    result = summary["value"].agg(["mean", "std", "count"]).reset_index()
    result.rename(columns={"mean": "mean_value", "std": "std_value"}, inplace=True)
    return result


def _bootstrap_deltas(df: pd.DataFrame, pivot_cols: Iterable[str]) -> pd.DataFrame:
    pivot = df.pivot_table(index="seed", columns=list(pivot_cols), values="value")
    if pivot.empty:
        return pd.DataFrame()
    best_method = pivot.mean(axis=0).idxmax()
    results = []
    baseline = pivot[best_method].dropna()
    for column in pivot.columns:
        aligned = pivot[[best_method, column]].dropna()
        if aligned.empty:
            continue
        diff, ci = paired_bootstrap(aligned[best_method].to_numpy(), aligned[column].to_numpy())
        p_value = permutation_test(aligned[best_method].to_numpy(), aligned[column].to_numpy())
        results.append({"comparison": column, "delta": diff, "ci_low": ci[0], "ci_high": ci[1], "p_value": p_value})
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["baseline"] = " vs ".join(str(x) for x in best_method)
    return result_df


def _write_outputs(df: pd.DataFrame, tex_path: Path, csv_path: Path, caption: str, label: str) -> None:
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, float_format="%.3f", caption=caption, label=label, escape=False)
    tex_path.write_text(latex, encoding="utf-8")


def generate_table(
    name: str,
    raw_path: Path,
    outdir: Path,
    group_cols: Iterable[str | None],
    caption: str,
    label: str,
    metric_filter: str | None = None,
) -> Dict[str, object]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw CSV for table '{name}': {raw_path}")
    df = pd.read_csv(raw_path)
    if metric_filter is not None:
        df = df[df["metric"] == metric_filter]
    resolved_groups: list[str] = []
    for idx, col in enumerate(group_cols):
        if col is None:
            continue
        if col in df.columns:
            resolved_groups.append(col)
        elif idx == 0:
            raise KeyError(f"Required column '{col}' missing from {raw_path}")
    if not resolved_groups:
        raise KeyError(f"No valid grouping columns found in {raw_path}")
    summary = _summarise_metric(df, resolved_groups)
    def _ci_from_row(row: pd.Series) -> str:
        mean = float(row["mean_value"])
        std_val = float(row["std_value"]) if not pd.isna(row["std_value"]) else 0.0
        count = float(row["count"])
        stderr = std_val / np.sqrt(max(1.0, count))
        ci = (mean - 1.96 * stderr, mean + 1.96 * stderr)
        return _format_ci(mean, ci)

    summary["ci"] = summary.apply(_ci_from_row, axis=1)
    if "seed" in df.columns and len(resolved_groups) >= 2:
        delta = _bootstrap_deltas(df, resolved_groups[:2])
    else:
        delta = pd.DataFrame()
    table_path = outdir / f"table_{name}.csv"
    tex_path = outdir / f"table_{name}.tex"
    display = summary.drop(columns=["std_value", "count"]).rename(columns={"mean_value": "mean"})
    _write_outputs(display, tex_path, table_path, caption, label)
    delta_path = outdir / f"table_{name}_significance.json"
    delta_path.write_text(delta.to_json(orient="records", indent=2), encoding="utf-8")
    return {
        "summary": str(table_path),
        "latex": str(tex_path),
        "significance": str(delta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tables for publication")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="Override raw data directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tables_dir = args.out / "tables"
    _ensure_directory(tables_dir)
    raw_dir = args.raw if args.raw else args.out / "raw_data"

    manifest: Dict[str, Dict[str, object]] = {}
    configs = [
        ("main", raw_dir / "main_results.csv", ("task", "method"), "Main results (Macro-Acc, AUC)", "tab:main_results"),
        ("lofo", raw_dir / "lofo.csv", ("held_out", "method"), "Cross-task transfer (LOFO)", "tab:lofo"),
        ("baselines", raw_dir / "baselines.csv", ("task", "method"), "Baseline comparisons", "tab:baselines"),
        ("ablations", raw_dir / "ablations.csv", ("component", "method",), "Ablation study", "tab:ablations"),
    ]

    for name, path, group_cols, caption, label in configs:
        try:
            manifest[name] = generate_table(
                name,
                path,
                tables_dir,
                group_cols,
                caption,
                label,
            )
        except FileNotFoundError as exc:
            if not args.dry_run:
                raise
            print(f"[warn] {exc}")

    manifest_path = tables_dir / "tables_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

