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


def test_full_window_reaches_last_timestep():
    """With dilations (8, 16, 32, 64) and kernel_size=3, the receptive
    field feeding position -1 is 2*(8+16+32+64) + 1 = 241 samples -
    wider than the 200-sample window, so every position in the window
    (including the very first) should influence the model's output.
    This is a regression guard: the previous dilation schedule
    (1, 2, 4, 8) only gave position -1 a 31-sample receptive field, so
    the last-timestep readout was silently discarding the earlier
    ~1.7s of the window - if a future change shrinks the dilations
    back down, this should catch it.
    """
    torch.manual_seed(0)
    model = ChannelAVelocityNet()
    model.eval()
    x = torch.zeros(1, 6, 200)
    x_perturbed = x.clone()
    x_perturbed[:, :, 0] = 5.0  # earliest timestep in the window

    with torch.no_grad():
        out_base = model(x)
        out_perturbed = model(x_perturbed)

    assert not torch.allclose(out_base, out_perturbed)


def test_fixed_seed_forward_pass():
    torch.manual_seed(0)
    model = ChannelAVelocityNet()
    x = torch.zeros(1, 6, 200)
    out = model(x)

    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 50.0  # untrained, zero input - should stay small
