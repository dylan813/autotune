"""Train the causal F0 estimator on synthetic data."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from python_prototype.training.model import (
    CausalF0Net, CausalF0Config, NUM_BINS,
    hz_to_cents, cents_to_bin_target,
)
from python_prototype.training.synthetic_data import (
    SyntheticDataConfig, make_batch,
)


def train(
    out_dir: Path,
    steps: int = 5000,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: str = "auto",
    log_every: int = 50,
) -> None:
    device_t = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto" else torch.device(device)
    )

    cfg_model = CausalF0Config(sample_rate=16000, hop_size=256)
    cfg_data  = SyntheticDataConfig(sample_rate=cfg_model.sample_rate,
                                    hop_size=cfg_model.hop_size,
                                    duration_sec=1.5)

    model = CausalF0Net(cfg_model).to(device_t)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng   = np.random.default_rng(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    start = time.perf_counter()

    for step in range(1, steps + 1):
        audio, f0_frames, voiced_frames = make_batch(batch_size, cfg_data, rng,
                                                     device=str(device_t))

        pitch_logits, voicing_logits = model(audio)
        # pitch_logits: (B, NUM_BINS, T_frames). The frontend's strided conv
        # may produce a slightly different T than the dataset's T_frames due
        # to padding; align by trimming to the shorter.
        T_pred = pitch_logits.shape[-1]
        T_tgt  = f0_frames.shape[-1]
        T = min(T_pred, T_tgt)
        pitch_logits   = pitch_logits[..., :T]
        voicing_logits = voicing_logits[..., :T]
        f0_frames      = f0_frames[..., :T]
        voiced_frames  = voiced_frames[..., :T]

        # Pitch loss: soft cross-entropy on bin distribution. Only on voiced.
        voiced_mask = voiced_frames.bool()
        if voiced_mask.any():
            cents     = hz_to_cents(f0_frames.clamp_min(1e-3))    # (B, T)
            tgt_probs = cents_to_bin_target(cents)                # (B, T, NUM_BINS)
            log_p     = F.log_softmax(pitch_logits, dim=1).transpose(1, 2)  # (B, T, NUM_BINS)
            pitch_loss = -(tgt_probs * log_p).sum(dim=-1)  # (B, T)
            pitch_loss = pitch_loss[voiced_mask].mean()
        else:
            pitch_loss = torch.tensor(0.0, device=device_t)

        voicing_loss = F.binary_cross_entropy_with_logits(
            voicing_logits, voiced_frames.float(),
        )

        loss = pitch_loss + 0.2 * voicing_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % log_every == 0:
            elapsed = time.perf_counter() - start
            print(f"step {step:5d}/{steps}  loss={loss.item():.4f}  "
                  f"pitch={pitch_loss.item():.4f}  voicing={voicing_loss.item():.4f}  "
                  f"({elapsed:.1f}s)")
            if loss.item() < best_loss:
                best_loss = loss.item()
                ckpt_path = out_dir / "causal_f0_best.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "cfg": cfg_model.__dict__,
                    "step": step,
                    "loss": loss.item(),
                }, ckpt_path)

    final_ckpt = out_dir / "causal_f0_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg_model.__dict__,
        "step": steps,
        "loss": loss.item(),
    }, final_ckpt)
    print(f"Done. Final checkpoint -> {final_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("python_prototype/training/checkpoints"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    train(args.out_dir, steps=args.steps, batch_size=args.batch_size,
          lr=args.lr, device=args.device, log_every=args.log_every)


if __name__ == "__main__":
    main()
