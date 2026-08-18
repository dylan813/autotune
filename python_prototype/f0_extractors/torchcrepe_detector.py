"""torchcrepe F0 detector.

CNN-based pitch estimator. Strong accuracy, GPU-friendly on the RTX 2060.
This is the default detector for the *high-quality* offline pipeline.

torchcrepe accepts arbitrary sample rates and returns one frame per hop.
We rely on the official `predict` helper which handles resampling and
returns both periodicity (used as voicing probability) and the median /
mean filters needed to smooth raw F0 estimates.
"""

from __future__ import annotations

import numpy as np

from . import F0Result


class TorchcrepeDetector:
    name = "torchcrepe"

    def __init__(
        self,
        model: str = "full",     # "full" or "tiny"; "tiny" is faster on RTX 2060
        device: str = "auto",    # "auto" | "cuda" | "cpu"
        fmin_hz: float = 50.0,
        fmax_hz: float = 1100.0,
        median_filter_ms: float = 30.0,   # 0 disables filtering
        voicing_threshold: float = 0.21,  # torchcrepe README default
        batch_size: int = 2048,
    ) -> None:
        self.model = model
        self.device = device
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz
        self.median_filter_ms = median_filter_ms
        self.voicing_threshold = voicing_threshold
        self.batch_size = batch_size

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # torchcrepe's internal CNN uses padding of 10 frames on each side at its
    # native 16 kHz / 10 ms hop. That means the model needs at least ~21 frames
    # of internal content, which corresponds to roughly 1024 samples at 44.1 kHz
    # after resampling. Inputs shorter than this cause a padding > dimension
    # error inside PyTorch. We pad up to this minimum and trim the output back.
    _MIN_SAMPLES = 1024

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_size: int,
    ) -> F0Result:
        import torch
        import torchcrepe

        device = self._resolve_device()

        # Guard against inputs that are too short for the model's receptive field.
        original_len = len(audio)
        if original_len < self._MIN_SAMPLES:
            audio = np.pad(audio, (0, self._MIN_SAMPLES - original_len))

        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        x = x.unsqueeze(0)  # (batch=1, samples)

        # torchcrepe.predict yields F0 in Hz and periodicity in [0, 1] for
        # every analysis hop. We pass hop_size in *samples at the input
        # sample rate* and torchcrepe handles internal resampling.
        f0, periodicity = torchcrepe.predict(
            x,
            sample_rate,
            hop_length=hop_size,
            fmin=self.fmin_hz,
            fmax=self.fmax_hz,
            model=self.model,
            batch_size=self.batch_size,
            device=device,
            return_periodicity=True,
        )

        # Optional median + mean filters: smooth pitch jumps and clean
        # periodicity. These are torchcrepe.filter helpers.
        if self.median_filter_ms > 0:
            win = max(1, int(round(self.median_filter_ms / 1000.0 *
                                   (sample_rate / hop_size))))
            f0          = torchcrepe.filter.median(f0,        win)
            periodicity = torchcrepe.filter.median(periodicity, win)

        # Mark frames below the voicing threshold as unvoiced. We expose
        # voicing as a continuous score so callers can re-threshold later.
        f0_np   = f0.detach().cpu().numpy()[0].astype(float)
        peri_np = periodicity.detach().cpu().numpy()[0].astype(float)

        # Trim any extra frames introduced by the minimum-length padding above.
        n_original_frames = int(np.ceil(original_len / hop_size))
        f0_np   = f0_np  [:n_original_frames]
        peri_np = peri_np[:n_original_frames]

        voiced_mask = peri_np >= self.voicing_threshold
        f0_clean = np.where(voiced_mask & np.isfinite(f0_np), f0_np, np.nan)

        times = np.arange(f0_clean.size, dtype=float) * (hop_size / sample_rate)
        return F0Result(times=times, f0_hz=f0_clean, voicing=peri_np)
