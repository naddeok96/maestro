"""Convenience CLI to meta-train a MAESTRO teacher with PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_meta_train import run_meta_training


DEFAULT_CONFIG = Path("configs/meta_train/small_cpu_debug.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-train a MAESTRO teacher and save the checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the meta-training config (default: small CPU debug)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory under which <run_id>/policy.pt will be written (default: outputs)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional random seed override"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume training from a checkpoint"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print config and resolved paths only"
    )
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Run the teacher in eval mode for deterministic rollouts",
    )
    # --- Brief imitation warm-start (pass-through to run_meta_training) ---
    parser.add_argument(
        "--bc-warm-start",
        action="store_true",
        help="Run a short behavior-cloning warm start before PPO (default: off)",
    )
    parser.add_argument(
        "--bc-episodes",
        type=int,
        default=2,
        help="Episodes to collect for BC warm start (default: 2)",
    )
    parser.add_argument(
        "--bc-baseline",
        type=str,
        default="uniform",
        choices=["uniform", "easy_to_hard", "greedy", "linucb"],
        help="Baseline scheduler used to synthesize BC targets (default: uniform)",
    )
    parser.add_argument(
        "--bc-usage",
        type=float,
        default=0.4,
        help="Target usage fraction when creating BC targets; must be in (0, 1) (default: 0.4)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=2,
        help="Supervised epochs to fit policy heads during BC warm start (default: 2)",
    )
    args = parser.parse_args()

    output_dir = run_meta_training(
        args.config,
        dry_run=args.dry_run,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
        deterministic_eval=args.deterministic_eval,
        bc_warm_start_flag=args.bc_warm_start,
        bc_episodes=args.bc_episodes,
        bc_baseline=args.bc_baseline,
        bc_usage=args.bc_usage,
        bc_epochs=args.bc_epochs,
    )

    if args.dry_run:
        print(
            f"[dry-run] Would write checkpoints and logs under: {output_dir.resolve()}"
        )
    else:
        checkpoint = output_dir / "policy.pt"
        print(f"[✓] Teacher checkpoint saved to {checkpoint.resolve()}")


if __name__ == "__main__":
    main()
