"""Export the trained alignment net to ONNX + quantized int8 - Section 4.6.

Usage:
    python models/alignment_net/export.py --checkpoint <path to .ckpt>
"""

import argparse

import torch

from models.alignment_net.model import AlignmentNet
from models.common.export_utils import export_to_onnx, quantize_dynamic_int8


def main(checkpoint_path: str, output_dir: str = "models/alignment_net/exported"):
    model = AlignmentNet()
    # TODO(Layer 1): load weights from the Lightning checkpoint - key
    # prefix depends on how train.py's LightningModule wraps the model
    # (currently `model.<param>`), strip it here before load_state_dict.
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    dummy_input = torch.randn(1, 9, 200)  # batch=1, per Section 4.6 step 1

    float32_path = f"{output_dir}/alignment_net_fp32.onnx"
    quantized_path = f"{output_dir}/alignment_net.onnx"  # naming per Section 4.6 step 4

    export_to_onnx(model, dummy_input, float32_path)
    quantize_dynamic_int8(float32_path, quantized_path)
    # TODO(Layer 1): call verify_quantized_accuracy() from
    # models/common/export_utils.py with a real val-set metric function
    # before trusting `quantized_path` - fall back to float16 if it
    # fails the 5% check (Section 4.6 step 3).


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.checkpoint)
