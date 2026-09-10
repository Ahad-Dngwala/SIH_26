"""Export a trained road-signature "signature pack" to ONNX + quantized
int8 - Section 4.6. One export per corridor/region (Section 4.4).

Usage:
    python models/road_signature/export.py --checkpoint <path to .ckpt> \
        --corridor ahmedabad_sg_highway --n-segments 40
"""

import argparse

import torch

from models.common.export_utils import export_to_onnx, quantize_dynamic_int8
from models.road_signature.model import RoadSignatureNet


def main(checkpoint_path: str, corridor: str, n_segments: int, output_dir: str = "models/road_signature/exported"):
    model = RoadSignatureNet(n_segments=n_segments)
    # TODO(Layer 1): load weights from the Lightning checkpoint, strip
    # the `model.` key prefix train.py's LightningModule adds.
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    dummy_input = torch.randn(1, 6, 200)  # batch=1, per Section 4.6 step 1

    float32_path = f"{output_dir}/{corridor}_fp32.onnx"
    # naming per Section 4.6 step 4: road_signature_<region>.onnx
    quantized_path = f"{output_dir}/road_signature_{corridor}.onnx"

    export_to_onnx(model, dummy_input, float32_path)
    quantize_dynamic_int8(float32_path, quantized_path)
    # TODO(Layer 1): call verify_quantized_accuracy() with a real
    # val macro-F1 metric function (Section 4.7 - not raw accuracy)
    # before trusting quantized_path.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corridor", required=True)
    parser.add_argument("--n-segments", type=int, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.corridor, args.n_segments)
