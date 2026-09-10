"""Save a trained calibration adapter as a small per-vehicle-profile
weight file - MIP Section 4.5 ("a small adapter weight file (a few KB)
saved per vehicle profile").

Not an ONNX export like the other four models: this just saves the
adapter's A/B matrices. See module README's "open integration question"
note about how this actually composes with the ONNX-exported frozen
base model at Android inference time - not resolved here.

Usage:
    python models/calibration_adapter/export.py --vehicle-profile my_scooter \
        --adapter-a-state <path> --adapter-b-state <path>
"""

import argparse

import torch


def main(vehicle_profile: str, adapter_a_state: str, adapter_b_state: str, output_dir: str = "models/calibration_adapter/exported"):
    # TODO(Layer 1): these paths currently expect raw state_dicts saved
    # ad hoc from train.py - formalize once train.py actually calls
    # this function instead of leaving a TODO.
    state_a = torch.load(adapter_a_state, map_location="cpu")
    state_b = torch.load(adapter_b_state, map_location="cpu")

    output_path = f"{output_dir}/{vehicle_profile}.adapter"
    torch.save({"channel_a": state_a, "channel_b": state_b}, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle-profile", required=True)
    parser.add_argument("--adapter-a-state", required=True)
    parser.add_argument("--adapter-b-state", required=True)
    args = parser.parse_args()
    main(args.vehicle_profile, args.adapter_a_state, args.adapter_b_state)
