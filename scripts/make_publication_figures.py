#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for csv in root.rglob("metrics.csv"):
        try:
            df = pd.read_csv(csv)
            df["run_dir"] = str(csv.parent)
            rows.append(df)
        except Exception:
            pass
    if not rows:
        raise FileNotFoundError(f"No metrics.csv under {root}")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()

    df = find_metrics(args.root)
    gb = df.groupby(["episode"]).agg({"return": "mean"}).reset_index()
    plt.figure()
    plt.plot(gb["episode"], gb["return"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Main Learning Curve (mean over runs)")
    fig1 = args.root / "fig1_learning_curves.png"
    plt.savefig(fig1, dpi=150)
    plt.close()

    last = df.sort_values("episode").groupby("run_dir").tail(1)
    table = last[
        ["run_dir", "return", "macro_accuracy", "avg_eta", "avg_u", "avg_usage"]
    ]
    table_path = args.root / "table1_main_results.csv"
    table.to_csv(table_path, index=False)

    print(json.dumps({"figures": [str(fig1)], "tables": [str(table_path)]}, indent=2))


if __name__ == "__main__":
    main()
