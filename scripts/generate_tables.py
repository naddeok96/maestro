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


def _format_with_significance(mean: float, ci: Tuple[float, float], p: float) -> str:
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{mean:.3f}{stars} [{ci[0]:.3f}, {ci[1]:.3f}]"


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
    pivot_cols: Tuple[str, ...] = tuple(resolved_groups[:2])
    significance_lookup: Dict[Tuple[object, ...], float] = {}
    if "seed" in df.columns and pivot_cols:
        delta = _bootstrap_deltas(df, pivot_cols)
        if not delta.empty and "comparison" in delta.columns and "p_value" in delta.columns:
            for _, delta_row in delta.iterrows():
                comparison = delta_row["comparison"]
                if isinstance(comparison, tuple):
                    key = comparison
                elif isinstance(comparison, (list, np.ndarray, pd.Index)):
                    key = tuple(comparison)
                else:
                    key = (comparison,)
                significance_lookup[key] = float(delta_row["p_value"])
    else:
        delta = pd.DataFrame()
    def _ci_from_row(row: pd.Series) -> str:
        mean = float(row["mean_value"])
        std_val = float(row["std_value"]) if not pd.isna(row["std_value"]) else 0.0
        count = float(row["count"])
        stderr = std_val / np.sqrt(max(1.0, count))
        ci = (mean - 1.96 * stderr, mean + 1.96 * stderr)
        if significance_lookup:
            key = tuple(row[col] for col in pivot_cols)
            if key in significance_lookup:
                return _format_with_significance(mean, ci, significance_lookup[key])
        return _format_ci(mean, ci)

    summary["ci"] = summary.apply(_ci_from_row, axis=1)
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
    bpath = raw_dir / "baselines.csv"
    if bpath.exists():
        t4 = pd.read_csv(bpath)
        if {"method", "final_macro"}.issubset(t4.columns):
            t4_long = t4.rename(columns={"final_macro": "value"}).copy()
            t4_long["task"] = "aggregate"
            t4_long["metric"] = "macro_acc"
            bpath.write_text(t4_long.to_csv(index=False))

    ablation_path = raw_dir / "ablation.csv"
    if ablation_path.exists():
        abl = pd.read_csv(ablation_path)
        if "component" not in abl.columns and "flags" in abl.columns:
            def _component_from_flags(s: str) -> str:
                stripped = str(s).strip()
                if stripped == "{}":
                    return "full"
                for key in [
                    "drop_grad_cosine",
                    "drop_progress_block",
                    "drop_model_block",
                    "drop_data_block",
                ]:
                    if key in stripped:
                        return key
                return stripped

            abl["component"] = abl["flags"].apply(_component_from_flags)
        if "value" not in abl.columns and "macro_accuracy" in abl.columns:
            abl = abl.rename(columns={"macro_accuracy": "value"})
        ablation_path.write_text(abl.to_csv(index=False))

    configs = [
        ("main", raw_dir / "main_results.csv", ("task",), "Main results (Macro-Acc)", "tab:main_results"),
        ("lofo", raw_dir / "lofo.csv", ("task",), "Cross-task transfer (LOFO)", "tab:lofo"),
        ("baselines", raw_dir / "baselines.csv", ("task", "method"), "Baseline comparisons", "tab:baselines"),
        ("ablations", raw_dir / "ablation.csv", ("component",), "Ablation study", "tab:ablations"),
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

