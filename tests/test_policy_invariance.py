from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from maestro.policy.deepsets import DeepSetsEncoder
from maestro.policy.policy_heads import PolicyHeads


def test_mixture_invariance_under_permutation():
    encoder = DeepSetsEncoder(input_dim=8, phi_dim=16, rho_dim=16)
    descriptors = torch.randn(3, 8)
    summary, encoded = encoder(descriptors)
    rest = torch.randn(25)
    global_context = torch.cat([summary, rest], dim=0)
    policy_heads = PolicyHeads(descriptor_dim=encoded.size(1), context_dim=global_context.size(0))
    mixture, _, _ = policy_heads(encoded, global_context)

    perm = torch.randperm(3)
    summary_perm, encoded_perm = encoder(descriptors[perm])
    global_context_perm = torch.cat([summary_perm, rest], dim=0)
    mixture_perm, _, _ = policy_heads(encoded_perm, global_context_perm)
    assert torch.allclose(mixture[perm], mixture_perm, atol=1e-5)
