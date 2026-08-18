"""Export a trained CausalF0Net checkpoint to ONNX for plugin use.

The exported model expects (B, T_samples) float32 audio at 16 kHz and
returns (pitch_logits: (B, NUM_BINS, T_frames), voicing_logits: (B, T_frames)).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from python_prototype.training.model import CausalF0Net, CausalF0Config


def export(ckpt_path: Path, onnx_path: Path,
           sample_length: int = 16000) -> None:
    blob = torch.load(ckpt_path, map_location="cpu")
    cfg  = CausalF0Config(**blob["cfg"]) if isinstance(blob.get("cfg"), dict) else CausalF0Config()
    model = CausalF0Net(cfg)
    model.load_state_dict(blob["model_state"])
    model.eval()

    dummy = torch.zeros(1, sample_length, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["audio"],
        output_names=["pitch_logits", "voicing_logits"],
        opset_version=17,
        dynamic_axes={
            "audio":          {0: "batch", 1: "samples"},
            "pitch_logits":   {0: "batch", 2: "frames"},
            "voicing_logits": {0: "batch", 1: "frames"},
        },
    )
    print(f"Wrote ONNX model -> {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output",     required=True, type=Path)
    parser.add_argument("--sample-length", type=int, default=16000)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.sample_length)


if __name__ == "__main__":
    main()
