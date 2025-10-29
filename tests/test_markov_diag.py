from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from maestro.eval.markov_diag import Transition, compute_markov_diagnostics


def test_markov_r2_bounds():
    transitions = []
    for i in range(5):
        state = {
            "g_data": np.ones(8) * i,
            "g_model": np.ones(6) * 0.1,
            "g_progress": np.ones(11) * 0.2,
        }
        next_state = {
            "g_data": np.ones(8) * (i + 1),
            "g_model": np.ones(6) * 0.1,
            "g_progress": np.ones(11) * 0.2,
        }
        transitions.append(Transition(state=state, action={}, next_state=next_state))
    result = compute_markov_diagnostics(transitions)
    assert -1.0 <= result["r2"] <= 1.0
