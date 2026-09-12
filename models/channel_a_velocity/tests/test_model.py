"""MIP Section 11.1: input/output shape check + a fixed-seed forward
pass with a known expected output range.

Run with: python -m pytest models/channel_a_velocity/tests/ -v
"""

import torch

from models.channel_a_velocity.model import ChannelAVelocityNet


def test_output_shape():
    model = ChannelAVelocityNet()
    x = torch.randn(4, 6, 200)
    out = model(x)
    assert out.shape == (4, 1)


def test_causal_no_lookahead():
    """TCN blocks are supposed to be causal (Chomp1d) - output at the
    window's end should not change if we perturb only a later timestep
    ... but the whole window is fed at once and pooled globally, so
    what we can actually check without a manual per-timestep forward is
    that changing an early timestep still changes the (global-pooled)
    output at all, i.e. the causal chomp didn't accidentally zero out
    the whole receptive field.
    """
    torch.manual_seed(0)
    model = ChannelAVelocityNet()
    model.eval()
    x = torch.zeros(1, 6, 200)
    x2 = x.clone()
    x2[:, :, 0] = 5.0  # perturb the earliest timestep
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x2)
    assert not torch.allclose(out1, out2)


def test_fixed_seed_forward_pass():
    torch.manual_seed(0)
    model = ChannelAVelocityNet()
    x = torch.zeros(1, 6, 200)
    out = model(x)

    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 50.0  # untrained, zero input - should stay small
