"""Causal CNN F0 estimator.

Design constraints:
- **Causal**: every conv is dilated + masked so output[t] depends only on
  input[..., :t]. Used by streaming inference in the plugin.
- **Small**: ~few hundred-K params; aims to run on CPU in the JUCE plugin
  if GPU isn't available.
- **Two heads**: F0 (regression in cents over a fixed range) +
  voicing (binary classification).

Pitch is predicted by classifying into 360 cents-bins spanning roughly
50 - 1100 Hz (5 octaves, 5-cent resolution), the same scheme CREPE uses.
The final scalar F0 estimate is the bin-weighted mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pitch-bin helpers
# ---------------------------------------------------------------------------

NUM_BINS  = 360
CENTS_MIN = 1997.0          # ~50 Hz in cents from C0
CENTS_MAX = 1997.0 + (NUM_BINS - 1) * 20.0   # 20-cent spacing


def cents_for_bin() -> torch.Tensor:
    return CENTS_MIN + 20.0 * torch.arange(NUM_BINS, dtype=torch.float32)


def hz_to_cents(hz: torch.Tensor) -> torch.Tensor:
    """Cents above C0 (~16.35 Hz). Same convention CREPE uses."""
    return 1200.0 * torch.log2(hz.clamp_min(1e-3) / 16.351597831287414)


def cents_to_hz(cents: torch.Tensor) -> torch.Tensor:
    return 16.351597831287414 * torch.pow(2.0, cents / 1200.0)


def cents_to_bin_target(cents: torch.Tensor, sigma: float = 25.0) -> torch.Tensor:
    """Convert cents -> a soft target distribution over NUM_BINS using a
    Gaussian centered on the true bin (CREPE-style cents-bin target)."""
    bin_cents = cents_for_bin().to(cents.device)            # (NUM_BINS,)
    diff      = cents.unsqueeze(-1) - bin_cents             # (..., NUM_BINS)
    weight    = torch.exp(-0.5 * (diff / sigma) ** 2)
    weight    = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weight


def bin_probs_to_cents(probs: torch.Tensor) -> torch.Tensor:
    """Weighted mean over bin centers."""
    bin_cents = cents_for_bin().to(probs.device)
    return (probs * bin_cents).sum(dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1-D causal conv with optional dilation. Left-pads the input."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=0, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


@dataclass
class CausalF0Config:
    channels: tuple[int, ...] = (32, 64, 96, 128, 128)
    kernel_size: int          = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    hop_size: int             = 256        # samples between F0 outputs
    sample_rate: int          = 16000


class CausalF0Net(nn.Module):
    """Stack of dilated causal convs over a strided front-end.

    Input  : (B, T_samples) raw audio @ sample_rate.
    Output : (B, NUM_BINS, T_frames) bin logits, (B, T_frames) voicing logits.
             T_frames = T_samples // hop_size.
    """

    def __init__(self, cfg: CausalF0Config = CausalF0Config()):
        super().__init__()
        self.cfg = cfg

        # Front-end: a single strided conv that downsamples by `hop_size`.
        # Keeps the model small and replaces "extract frame, then conv".
        self.frontend = nn.Conv1d(1, cfg.channels[0],
                                  kernel_size=cfg.hop_size * 2,
                                  stride=cfg.hop_size,
                                  padding=cfg.hop_size // 2)

        layers: list[nn.Module] = []
        in_ch = cfg.channels[0]
        for out_ch, dil in zip(cfg.channels[1:], cfg.dilations):
            layers.append(CausalConv1d(in_ch, out_ch, cfg.kernel_size, dilation=dil))
            layers.append(nn.GELU())
            in_ch = out_ch
        self.trunk = nn.Sequential(*layers)

        self.pitch_head   = nn.Conv1d(in_ch, NUM_BINS, kernel_size=1)
        self.voicing_head = nn.Conv1d(in_ch, 1,        kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)                  # (B, 1, T)
        h = self.frontend(x)                    # (B, C0, T_frames)
        h = F.gelu(h)
        h = self.trunk(h)
        pitch_logits   = self.pitch_head(h)     # (B, NUM_BINS, T_frames)
        voicing_logits = self.voicing_head(h).squeeze(1)  # (B, T_frames)
        return pitch_logits, voicing_logits

    # Convenience inference helper.
    @torch.no_grad()
    def predict_hz(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pitch_logits, voicing_logits = self.forward(audio)
        probs = F.softmax(pitch_logits, dim=1).transpose(1, 2)  # (B, T_frames, NUM_BINS)
        cents = bin_probs_to_cents(probs)
        hz    = cents_to_hz(cents)
        voicing = torch.sigmoid(voicing_logits)
        return hz, voicing
