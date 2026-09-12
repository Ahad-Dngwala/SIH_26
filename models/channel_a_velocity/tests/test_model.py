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
    """The model now reads out the causal feature at the window's last
    timestep directly (no more global pooling across all 200
    positions), so this can check real causality: with dilations
    (1, 2, 4, 8) and kernel_size=3, the receptive field feeding
    position -1 is 2*(1+2+4+8) + 1 = 31 samples, i.e. positions 169-199.

    - Perturbing a timestep INSIDE that receptive field (e.g. 180)
      must change the output.
    - Perturbing a timestep OUTSIDE it (e.g. 0, the earliest sample)
      must NOT change the output at all - if it does, the chomp is
      leaking future/out-of-range context into a position that should
      be blind to it.
    """
    torch.manual_seed(0)
    model = ChannelAVelocityNet()
    model.eval()
    x = torch.zeros(1, 6, 200)

    x_in_rf = x.clone()
    x_in_rf[:, :, 180] = 5.0  # inside the 31-sample receptive field of position -1
    x_out_rf = x.clone()
    x_out_rf[:, :, 0] = 5.0  # outside it

    with torch.no_grad():
        out_base = model(x)
        out_in_rf = model(x_in_rf)
        out_out_rf = model(x_out_rf)

    assert not torch.allclose(out_base, out_in_rf)
    assert torch.allclose(out_base, out_out_rf)


def test_fixed_seed_forward_pass():
    torch.manual_seed(0)
    model = ChannelAVelocityNet()
    x = torch.zeros(1, 6, 200)
    out = model(x)

    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 50.0  # untrained, zero input - should stay small
