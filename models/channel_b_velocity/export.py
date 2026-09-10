"""Export the trained Channel B model to ONNX + quantized int8 - Section 4.6.

Usage:
    python models/channel_b_velocity/export.py --checkpoint <path to .ckpt>
"""

import argparse

import torch

from models.channel_b_velocity.model import ChannelBVelocityNet
from models.common.export_utils import export_to_onnx, quantize_dynamic_int8


def main(checkpoint_path: str, output_dir: str = "models/channel_b_velocity/exported"):
    model = ChannelBVelocityNet()
    # TODO(Layer 1): load weights from the Lightning checkpoint, strip
    # the `model.` key prefix train.py's LightningModule adds.
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    dummy_input = torch.randn(1, 3, 400)  # batch=1, per Section 4.6 step 1

    float32_path = f"{output_dir}/channel_b_fp32.onnx"
    quantized_path = f"{output_dir}/channel_b.onnx"  # naming per Section 4.6 step 4

    export_to_onnx(model, dummy_input, float32_path)
    quantize_dynamic_int8(float32_path, quantized_path)
    # TODO(Layer 1): call verify_quantized_accuracy() with a real
    # val-RMSE metric function before trusting quantized_path.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.checkpoint)
