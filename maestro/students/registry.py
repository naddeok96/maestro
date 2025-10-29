"""Registry of available student models."""
from __future__ import annotations

from typing import Iterable, List

from maestro.datasets.synthetic import SyntheticCurriculum
from maestro.students.base import BaseStudent, StudentModel


def load_students(curriculum: SyntheticCurriculum) -> List[StudentModel]:
    """Return a canonical ordered set of student models."""

    difficulty_stats = curriculum.describe()
    avg_difficulty = difficulty_stats["avg_difficulty"]
    return [
        BaseStudent(name="novice", skill=avg_difficulty - 0.3, assessment_cost=0.1, noise=0.2),
        BaseStudent(name="intermediate", skill=avg_difficulty, assessment_cost=0.2, noise=0.15),
        BaseStudent(name="expert", skill=avg_difficulty + 0.3, assessment_cost=0.3, noise=0.1),
    ]
