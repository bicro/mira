"""Sound effects for the Pong world model: synthesis, a tiny neural audio codec, and detection.

The audio pipeline mirrors how MIRA's sibling systems at Kyutai treat sound (the Mimi codec):
waveforms are compressed by a small neural autoencoder into a low-rate latent, and the world model
predicts in that latent space. Here:

- Sound effects are classic-Pong square waves synthesized from the env's event flags
  (paddle movement tick, wall bounce, paddle hit, score) at 8 kHz mono.
- :class:`AudioCodec` compresses each 100 ms chunk (800 samples, one latent-frame of time) into an
  8-channel latent, trained as a plain autoencoder in minutes.
- The world model consumes the audio latent through the **pointwise-sum trick**: the per-frame
  audio latent is broadcast over the 4x4 spatial grid and concatenated to the video latent
  channels, so the DiT's input projection pointwise-sums a learned audio embedding into every
  spatial token (a linear layer over concatenated channels is exactly the sum of two linear
  projections), and the flow-matching objective diffuses video and audio jointly. No transformer
  changes needed.
- ``AVBatch`` extends mira's ``VideoActionBatch`` with the aligned waveform.
- :func:`detect_events` recovers events from generated audio by matched filtering, for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mira.data.batch import VideoActionBatch

SAMPLE_RATE = 8000
SAMPLES_PER_FRAME = SAMPLE_RATE // 20  # 400 samples per 20 fps video frame
AUDIO_LATENT_DIM = 8
CHUNK = 2 * SAMPLES_PER_FRAME  # one latent frame of audio (100 ms, 800 samples)


def _square(freq: float, ms: float, amp: float) -> np.ndarray:
    """A decaying square-wave beep — the classic Pong sound."""
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.arange(n) / SAMPLE_RATE
    wave = amp * np.sign(np.sin(2 * np.pi * freq * t))
    return (wave * np.linspace(1.0, 0.0, n) ** 0.5).astype(np.float32)


# One template per event channel: [p1 moved, p2 moved, wall, paddle hit, score].
TEMPLATES = [
    _square(90, 15, 0.15),  # movement tick (quiet: it fires on ~half of all frames)
    _square(90, 15, 0.15),
    _square(226, 40, 0.6),  # wall bounce
    _square(459, 40, 0.7),  # paddle hit
    _square(490, 200, 0.8),  # score
]
# Distinct, collision-and-score-only templates used for matched-filter evaluation (movement ticks
# share a template and overlap too much to score cleanly).
EVAL_EVENTS = {"wall": 2, "hit": 3, "score": 4}


def synthesize_audio(events: np.ndarray) -> np.ndarray:
    """Mix event templates into per-episode waveforms.

    Args:
        events: (E, T, 5) uint8 event flags, audible at each frame.

    Returns:
        (E, T * SAMPLES_PER_FRAME) float32 waveform in [-1, 1].
    """
    n_episodes, n_frames, n_channels = events.shape
    total = n_frames * SAMPLES_PER_FRAME
    longest = max(len(t) for t in TEMPLATES)
    waves = np.zeros((n_episodes, total + longest), dtype=np.float32)
    for ch in range(n_channels):
        template = TEMPLATES[ch]
        eps, frames = np.nonzero(events[:, :, ch])
        for e, f in zip(eps, frames):
            start = f * SAMPLES_PER_FRAME
            waves[e, start : start + len(template)] += template
    return np.clip(waves[:, :total], -1.0, 1.0)


class AudioCodec(nn.Module):
    """Tiny convolutional autoencoder: 800-sample chunk <-> 8-dim latent (100x compression).

    Plays the role of Kyutai's Mimi codec at toy scale: the world model never sees waveforms,
    only these latents.
    """

    def __init__(self, latent_dim: int = AUDIO_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 8, stride=4, padding=2),  # 800 -> 200
            nn.GELU(),
            nn.Conv1d(16, 32, 8, stride=4, padding=2),  # 200 -> 50
            nn.GELU(),
            nn.Conv1d(32, 64, 10, stride=10),  # 50 -> 5
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 5, latent_dim),
        )
        self.dec_in = nn.Linear(latent_dim, 64 * 5)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 10, stride=10),  # 5 -> 50
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, 8, stride=4, padding=2),  # 50 -> 200
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, 8, stride=4, padding=2),  # 200 -> 800
            nn.Tanh(),
        )

    def encode(self, chunks: Tensor) -> Tensor:
        """(..., CHUNK) waveform -> (..., latent_dim)."""
        shape = chunks.shape[:-1]
        z = self.encoder(chunks.reshape(-1, 1, CHUNK))
        return z.reshape(*shape, self.latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        """(..., latent_dim) -> (..., CHUNK) waveform in [-1, 1]."""
        shape = z.shape[:-1]
        x = self.dec_in(z.reshape(-1, self.latent_dim)).reshape(-1, 64, 5)
        return self.decoder(x).reshape(*shape, CHUNK)

    def forward(self, chunks: Tensor) -> Tensor:
        return self.decode(self.encode(chunks))

    def save_checkpoint(self, path: str | Path, extra_info: dict | None = None) -> None:
        torch.save({"state_dict": self.state_dict(), "info": extra_info or {}}, path)

    @classmethod
    def load_from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> AudioCodec:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        codec = cls()
        codec.load_state_dict(checkpoint["state_dict"])
        codec.to(device)
        codec.info_from_checkpoint = checkpoint["info"]
        return codec


def train_audio_codec(waves: Tensor, steps: int, device: str = "cuda") -> AudioCodec:
    """Train the chunk autoencoder on (E, S) waveforms, oversampling event-bearing chunks."""
    codec = AudioCodec().to(device)
    opt = torch.optim.AdamW(codec.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    chunks = waves.reshape(-1, CHUNK)  # all aligned 100 ms chunks in the dataset
    energy = chunks.abs().amax(dim=1)
    loud = torch.nonzero(energy > 0.05).squeeze(1)
    batch_size = 512
    for step in range(1, steps + 1):
        # Half the batch uniform, half from event-bearing chunks (most chunks are near-silence).
        idx = torch.cat(
            [
                torch.randint(chunks.shape[0], (batch_size // 2,), device=device),
                loud[torch.randint(loud.shape[0], (batch_size // 2,), device=device)],
            ]
        )
        x = chunks[idx]
        recon = codec(x)
        loss = F.mse_loss(recon, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            print(f"[audio codec {step}/{steps}] mse={loss.item():.6f}", flush=True)
    return codec


@torch.no_grad()
def detect_events(wave: Tensor, threshold: float = 0.5) -> dict[str, Tensor]:
    """Matched-filter detection of collision/score events in a waveform (T_samples,).

    Returns, per event name, a bool tensor over frames marking detected onsets. Used to check
    whether the *generated* audio contains the right effect at the right time.
    """
    out = {}
    for name, ch in EVAL_EVENTS.items():
        template = torch.as_tensor(TEMPLATES[ch], device=wave.device)
        template = template / template.norm()
        score = F.conv1d(wave.view(1, 1, -1), template.view(1, 1, -1)).squeeze()
        # Local energy normalization -> normalized cross-correlation in [0, 1].
        energy = F.avg_pool1d(
            (wave**2).view(1, 1, -1), kernel_size=len(template), stride=1
        ).squeeze() * len(template)
        ncc = score / (energy.sqrt() + 1e-3)
        n_frames = wave.shape[0] // SAMPLES_PER_FRAME
        per_frame = torch.zeros(n_frames, dtype=torch.bool, device=wave.device)
        hits = torch.nonzero((ncc > threshold) & (score > 0.3)).squeeze(1)
        per_frame[(hits // SAMPLES_PER_FRAME).clamp(max=n_frames - 1)] = True
        # Onset suppression: a beep spans several frames of correlation; report only its first.
        sustained = torch.zeros_like(per_frame)
        span = max(1, len(TEMPLATES[ch]) // SAMPLES_PER_FRAME + 1)
        for shift in range(1, span + 1):
            sustained[shift:] |= per_frame[:-shift]
        out[name] = per_frame & ~sustained
    return out


@dataclass
class AVBatch(VideoActionBatch):
    """A ``VideoActionBatch`` plus the frame-aligned audio waveform (B, T, SAMPLES_PER_FRAME)."""

    audio: torch.Tensor = None  # type: ignore[assignment]

    def to(self, *args: Any, **kwargs: Any) -> AVBatch:
        return AVBatch(
            video=self.video.to(*args, **kwargs),
            actions=self.actions.to(*args, **kwargs),
            audio=self.audio.to(*args, **kwargs),
        )

    def pin_memory(self) -> AVBatch:
        return AVBatch(
            video=self.video.pin_memory(),
            actions=self.actions.pin_memory(),
            audio=self.audio.pin_memory(),
        )

    def clone(self) -> AVBatch:
        return AVBatch(video=self.video.clone(), actions=self.actions.clone(), audio=self.audio.clone())

    def slice_time(self, start: int | None, end: int | None, *, fps: int) -> AVBatch:
        return AVBatch(
            video=self.video[:, start:end],
            actions=self.actions.slice_time(start, end),
            audio=self.audio[:, start:end],
        )

    def cat_time(self, other: VideoActionBatch) -> AVBatch:
        assert isinstance(other, AVBatch)
        base = super().cat_time(other)
        return AVBatch(
            video=base.video,
            actions=base.actions,
            audio=torch.cat([self.audio, other.audio.to(self.audio.device)], dim=1),
        )
