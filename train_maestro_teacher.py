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
    args = parser.parse_args()

    output_dir = run_meta_training(
        args.config,
        dry_run=args.dry_run,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
        deterministic_eval=args.deterministic_eval,
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
