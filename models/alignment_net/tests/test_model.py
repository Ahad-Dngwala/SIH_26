"""MIP Section 11.1: input/output shape check + a fixed-seed forward
pass with a known expected output range, so a refactor that silently
breaks the architecture gets caught here instead of at training time.

Run with: python -m pytest models/alignment_net/tests/ -v
"""

import torch

from models.alignment_net.model import AlignmentNet


def test_output_shape():
    model = AlignmentNet()
    x = torch.randn(4, 9, 200)
    out = model(x)
    assert out.shape == (4, 4)


def test_to_angles_shape_and_yaw_range():
    model = AlignmentNet()
    x = torch.randn(4, 9, 200)
    angles = model.to_angles(model(x))
    assert angles.shape == (4, 3)
    yaw = angles[:, 2]
    # atan2 output range - if this ever falls outside it, to_angles()
    # broke, not the model's training.
    assert torch.all(yaw >= -torch.pi) and torch.all(yaw <= torch.pi)


def test_fixed_seed_forward_pass():
    """An untrained but seeded model should produce finite, bounded
    output - not NaN/Inf, and not wildly out of range for a network
    with no activation clamp on its final layer. Catches silent
    breakage (e.g. a bad reshape, a dropped ReLU) from refactors,
    per Section 11.1 - this is a sanity check, not a trained-accuracy
    check.
    """
    torch.manual_seed(0)
    model = AlignmentNet()
    x = torch.zeros(1, 9, 200)
    out = model(x)

    assert out.shape == (1, 4)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 50.0  # untrained FC(32->4) on zero input stays small
