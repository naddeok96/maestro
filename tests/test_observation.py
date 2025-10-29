from __future__ import annotations

import numpy as np

from maestro.datasets.synthetic import SyntheticCurriculum
from maestro.envs.observation import ObservationBuilder
from maestro.students.registry import load_students


def test_observation_invariance():
    curriculum = SyntheticCurriculum(seed=0)
    students = load_students(curriculum)
    builder = ObservationBuilder(students, curriculum, probes_per_step=3, rng=np.random.default_rng(0))
    builder.reset()
    probes = builder.build()
    names = [p.student_name for p in probes]
    assert names == sorted(names)


def test_support_updates():
    curriculum = SyntheticCurriculum(seed=1)
    students = load_students(curriculum)
    builder = ObservationBuilder(students, curriculum, probes_per_step=2, rng=np.random.default_rng(1))
    builder.reset()
    before = list(builder.build())
    builder.update_with_outcome(0, accuracy=0.9, cost=0.1)
    after = list(builder.build())
    assert after[0].support >= before[0].support
    assert after[0].estimated_accuracy != before[0].estimated_accuracy
