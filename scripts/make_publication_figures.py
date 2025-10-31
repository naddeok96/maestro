#!/usr/bin/env python
"""Generate all publication figures from consolidated CSV artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    return pd.read_csv(path)


def _aggregate_curve(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["task", "method", "step", "metric"])
    stats = grouped["value"].agg(["mean", "std", "count"]).reset_index()
    stats.rename(columns={"mean": "mean_value", "std": "std_value"}, inplace=True)
    return stats


def fig1_learning_curves(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    aggregated = _aggregate_curve(df)
    for task in sorted(aggregated["task"].unique()):
        subset = aggregated[(aggregated["task"] == task) & (aggregated["metric"].isin(["macro_acc", "macro_mAP"]))]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, method_df in subset.groupby("method"):
            ax.plot(method_df["step"], method_df["mean_value"], label=method)
            std = method_df["std_value"].fillna(0.0)
            ax.fill_between(method_df["step"], method_df["mean_value"] - std, method_df["mean_value"] + std, alpha=0.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Macro metric")
        ax.set_title(f"Learning curves – {task}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = outdir / f"fig1_learning_{task}.{ext}"
            fig.savefig(path, dpi=200)
            outputs.append(path)
        plt.close(fig)
    return outputs


def fig2_markov_diagnostics(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    sorted_df = df.sort_values("r2", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sorted_df["feature"], sorted_df["r2"], color="#1f77b4")
    ax.set_ylabel("R²")
    ax.set_title("Markov diagnostics")
    ax.set_xticklabels(sorted_df["feature"], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = outdir / f"fig2_markov_r2.{ext}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    return outputs


def fig3_n_invariance(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for train_n, group in df.groupby("train_N"):
        ax.plot(group["test_N"], group["macro_metric"], marker="o", label=f"train N={train_n}")
    ax.set_xlabel("Test N")
    ax.set_ylabel("Macro metric")
    ax.set_title("N-invariance")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = outdir / f"fig3_n_invariance.{ext}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    return outputs


def fig4_ood_heatmap(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    pivot = df.pivot_table(index="noise", columns="imbalance", values="macro_metric", aggfunc=np.mean)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Imbalance")
    ax.set_ylabel("Noise")
    ax.set_title("OOD robustness")
    fig.colorbar(cax, ax=ax, label="Macro metric")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = outdir / f"fig4_ood_heatmap.{ext}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    return outputs


def fig5_ablation(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    ordered = df.sort_values("delta_macro")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(ordered["component"], ordered["delta_macro"], color="#d62728")
    ax.set_xlabel("Δ macro metric vs full")
    ax.set_title("Ablation study")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = outdir / f"fig5_ablation.{ext}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication figures")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--raw", type=Path, help="Override raw data directory")
    parser.add_argument("--dry-run", action="store_true", help="Skip missing figures for CI")
    args = parser.parse_args()

    outdir = args.out / "figures"
    _ensure_directory(outdir)
    raw_dir = args.raw if args.raw else args.out / "raw_data"

    generated = {}
    try:
        curves_df = _load_csv(raw_dir / "learning_curves.csv")
        generated["fig1"] = [str(p) for p in fig1_learning_curves(curves_df, outdir)]
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    try:
        markov_df = pd.read_json(raw_dir / "markov_diag.jsonl", lines=True)
        generated["fig2"] = [str(p) for p in fig2_markov_diagnostics(markov_df, outdir)]
    except (FileNotFoundError, ValueError) as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    try:
        n_df = _load_csv(raw_dir / "n_invariance.csv")
        generated["fig3"] = [str(p) for p in fig3_n_invariance(n_df, outdir)]
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    try:
        ood_df = _load_csv(raw_dir / "ood_grid.csv")
        generated["fig4"] = [str(p) for p in fig4_ood_heatmap(ood_df, outdir)]
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    try:
        ablation_df = _load_csv(raw_dir / "ablation.csv")
        generated["fig5"] = [str(p) for p in fig5_ablation(ablation_df, outdir)]
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    manifest_path = outdir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(json.dumps(generated, indent=2))


if __name__ == "__main__":
    main()

