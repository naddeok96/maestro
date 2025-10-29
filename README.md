# MAESTRO Experiments Codebase

This repository contains a reference implementation of the **MAESTRO** experiments
from the paper *Learning to Teach for Distributional Robustness*.  The codebase is
organised so that a single reinforcement-learning "teacher" policy can be
meta-trained on inexpensive synthetic episodes and then reused for a wide variety
of curriculum-learning scenarios.

The implementation focuses on three guarantees highlighted in the paper:

* **Markov teacher--student interaction.**  The environment exposes a grouped
  observation consisting of dataset-level embeddings, model-complexity features,
  and training-progress indicators that make the interaction Markov under SGD
  dynamics with a budgeted cost.
* **Number-of-datasets invariance.**  Dataset descriptors are processed through a
  DeepSets encoder with shared per-dataset policy heads, ensuring that the policy
  produces valid mixture weights for any number and any permutation of datasets.
* **Task/architecture portability.**  Synthetic classification, NER, and
  detection students conform to a common API so that new students and datasets
  can be registered without touching the core environment.

## Quickstart

1. Create a virtual environment (conda or venv) and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the CPU debug meta-training configuration (finishes in under a minute on
   a laptop):

```bash
python scripts/run_meta_train.py --config configs/meta_train/small_cpu_debug.yaml
```

3. Evaluate the resulting teacher checkpoints and build the Markov diagnostics:

```bash
python scripts/run_eval.py --config configs/meta_train/lofo_classification.yaml
python scripts/run_markov_diag.py --config configs/meta_train/small_cpu_debug.yaml
```

4. Generate the paper figures and tables from the saved CSV/JSON artefacts:

```bash
python scripts/plot_make_figures.py --run-dir outputs/debug_run
```

All scripts support the `--dry-run` flag to verify configuration loading without
starting heavy computation.

## Repository layout

The high-level directory structure matches the detailed specification in the
paper's prompt and is reproduced below for convenience:

```
maestro/
  configs/                    # Hydra-style YAML configs for tasks, training, eval
  data/                       # Synthetic data generation notes
  maestro/
    envs/                     # Gymnasium environment, probes, observation logic
    policy/                   # DeepSets encoder, PPO implementation
    students/                 # Student models implementing the common API
    datasets/                 # Synthetic dataset factories and registry
    baselines/                # Uniform/easy-to-hard/greedy/bandit schedulers
    eval/                     # Diagnostics: Markovity, transfer, N-invariance
    utils/                    # Common helpers (seeding, EMA, logging, etc.)
  scripts/                    # CLI entry points for experiments and plotting
  tests/                      # Pytest suite used in CI
```

The environment outputs grouped observations with the following components:

* `g_data` -- DeepSets summary of the per-dataset descriptors `z_{t,i}` which
  include NLL moments, uncertainty, gradient agreement, diversity, and size.
* `g_model` -- Model-complexity statistics such as log parameter count and
  effective depth.
* `g_progress` -- Training-progress indicators such as remaining budget,
  current learning rate, validation loss trends, and gradient similarity.

The per-dataset descriptors themselves are exposed to the policy so that the
shared mixture head can emit logits for each dataset while remaining
permutation-equivariant.

## Testing and quality

The repository ships with a comprehensive pytest suite that is executed in CI.
To run the tests locally use:

```bash
pytest
```

Static type checking (mypy) and code formatting (black/isort/flake8) are driven
through pre-commit hooks defined in `.pre-commit-config.yaml`.

## Reproducibility notes

* Deterministic seeds and frozen random projections are used for probe
  gradients and dataset sampling.  Running the same configuration twice will
  produce identical metrics.
* All experiment scripts emit TensorBoard logs, CSV metrics, checkpoints, and
  JSON diagnostics under `outputs/<run_id>/`.
* FLOPs estimation gracefully falls back to timing-based proxies when external
  libraries (ptflops/fvcore) are unavailable, ensuring portability.

For a more detailed overview of the grouped observation design, CMDP handling,
metrics, and supported students/datasets please refer to the inline module
documentation in `maestro/envs/` and `maestro/students/`.
