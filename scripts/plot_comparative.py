"""Plot comparative baselines and aggregate Table 4 metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def _resolve_metric_paths(entries: Iterable[str]) -> Dict[str, List[Path]]:
    resolved: Dict[str, List[Path]] = defaultdict(list)
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Malformed --run entry '{entry}'. Expected format method=path[,path2,...]"
            )
        method, raw_paths = entry.split("=", maxsplit=1)
        for part in raw_paths.split(","):
            path = Path(part)
            if path.is_file():
                resolved[method].append(path)
                continue
            if path.is_dir():
                csvs = sorted(path.rglob("metrics.csv"))
                if not csvs:
                    raise FileNotFoundError(f"No metrics.csv found under {path}")
                resolved[method].extend(csvs)
                continue
            matches = list(Path().glob(part))
            if not matches:
                raise FileNotFoundError(f"Could not resolve path '{part}'")
            for match in matches:
                resolved[method].append(match)
    return resolved


def _load_runs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = Path(path).parent.name
        frame["source"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "num_steps" not in df.columns:
        raise ValueError("metrics.csv is missing 'num_steps' column")
    df = df.sort_values(["seed", "episode"])
    df["steps"] = df.groupby("seed")["num_steps"].cumsum()
    return df


def _aggregate_learning_curve(df: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        df.groupby(["seed", "steps"])["macro_accuracy"].last().reset_index()
    )
    summary = (
        per_seed.groupby("steps")["macro_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return summary


def _compute_auc(steps: np.ndarray, values: np.ndarray) -> float:
    if len(steps) == 0:
        return 0.0
    return float(np.trapz(values, steps))


def _summarise_table(df: pd.DataFrame, method: str) -> Dict[str, float]:
    per_seed = []
    for seed, seed_frame in df.groupby("seed"):
        seed_frame = seed_frame.sort_values("steps")
        final_macro = float(seed_frame["macro_accuracy"].iloc[-1])
        auc = _compute_auc(seed_frame["steps"].to_numpy(), seed_frame["macro_accuracy"].to_numpy())
        avg_eta = float(seed_frame["avg_eta"].mean()) if "avg_eta" in seed_frame else 0.0
        avg_usage = (
            float(seed_frame["avg_usage"].mean()) if "avg_usage" in seed_frame else 0.0
        )
        avg_u = float(seed_frame["avg_u"].mean()) if "avg_u" in seed_frame else 0.0
        per_seed.append(
            {
                "seed": seed,
                "final_macro": final_macro,
                "auc": auc,
                "avg_eta": avg_eta,
                "avg_u": avg_u,
                "avg_usage": avg_usage,
            }
        )
    table = pd.DataFrame(per_seed)
    return {
        "method": method,
        "final_macro": table["final_macro"].mean() if not table.empty else 0.0,
        "macro_std": table["final_macro"].std(ddof=0) if len(table) > 1 else 0.0,
        "sample_efficiency_auc": table["auc"].mean() if not table.empty else 0.0,
        "avg_eta": table["avg_eta"].mean() if not table.empty else 0.0,
        "avg_u": table["avg_u"].mean() if not table.empty else 0.0,
        "avg_usage": table["avg_usage"].mean() if not table.empty else 0.0,
    }


def plot_learning_curves(curves: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    for method, summary in curves.items():
        steps = summary["steps"].to_numpy()
        mean = summary["mean"].to_numpy()
        std = summary["std"].fillna(0.0).to_numpy()
        label = method.replace("_", " ")
        plt.plot(steps, mean, label=label)
        if np.any(std > 0):
            plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Environment steps")
    plt.ylabel("Macro accuracy")
    plt.title("Comparative Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "comparative_learning_curves.png"
    pdf_path = output_dir / "comparative_learning_curves.pdf"
    plt.tight_layout()
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot comparative results")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help="Method=path1[,path2,...] entries. Directories are searched recursively",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = _resolve_metric_paths(args.runs)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/comparative_plots") / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    curves: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, float]] = []

    for method, paths in tqdm(resolved.items(), desc="Aggregating"):
        df = _load_runs(paths)
        prepared = _prepare_time_series(df)
        curves[method] = _aggregate_learning_curve(prepared)
        rows.append(_summarise_table(prepared, method))

    table = pd.DataFrame(rows)
    table_path = output_dir / "table4_metrics.csv"
    table.sort_values("method").to_csv(table_path, index=False)
    plot_learning_curves(curves, output_dir)

    print("Saved learning curves and table to", output_dir)
    print(table.sort_values("method").to_string(index=False))


if __name__ == "__main__":
    main()
