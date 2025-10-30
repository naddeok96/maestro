"""Generate plots and tables from experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    csv_path = args.run_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv in {args.run_dir}")
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df["episode"], df["return"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning curve")
    figure_path = args.run_dir / "learning_curve.png"
    plt.savefig(figure_path, dpi=150)
    print({"figure": str(figure_path)})


if __name__ == "__main__":
    main()
