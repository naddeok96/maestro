import pytest
import torch
from torch.utils.data import DataLoader

from maestro.datasets import build_from_config
from maestro.datasets.collate import detection_collate


def test_detection_eval_loader_collate():
    try:
        dataset_specs = build_from_config("configs/tasks/detection.yaml", seed=0)
    except RuntimeError as exc:  # pragma: no cover - depends on local data cache
        if "download_datasets.sh" in str(exc):
            pytest.skip(str(exc))
        raise
    detection_spec = dataset_specs[0]
    loader = DataLoader(detection_spec.val, batch_size=4, collate_fn=detection_collate)

    images, targets = next(iter(loader))

    assert images.dim() == 4
    assert isinstance(targets, list)
    assert len(targets) == 4
    assert all(isinstance(t, torch.Tensor) and t.ndim == 2 for t in targets)
