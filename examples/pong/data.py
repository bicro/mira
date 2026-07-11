"""GPU-resident batch sampler over the toy Pong dataset.

The whole dataset (a few hundred MB of 16x16 frames) fits in VRAM, so batches are assembled by
pure tensor indexing with no dataloader. Batches come out as mira's :class:`VideoActionBatch`
(video uint8 + :class:`ActionTensors`), the same container the Rocket League loader produces; the
keyboard-only convention matches theirs (zero mouse deltas, NaN sensitivity).

Multiplayer batches follow the MultiWrapper loader invariant: the two views of an episode occupy
contiguous, player-id-ordered batch rows, so ``rearrange("(b p) ...")`` lines up with the tiling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionConfig, ActionTensors

from examples.pong.pong_env import KEY_NAMES

PONG_ACTIONS = ActionConfig(valid_keys=KEY_NAMES, source_fps=20, target_fps=20)


class PongData:
    """Loads one split of the generated dataset onto a device and samples random clip batches.

    With ``with_audio=True`` the event flags are synthesized into 8 kHz waveforms once at load
    time, and every sampled batch is an :class:`~examples.pong.pong_audio.AVBatch` carrying the
    frame-aligned audio.
    """

    def __init__(self, path: str | Path, device: str = "cuda", with_audio: bool = False):
        raw = np.load(path)
        # (E, P, T, 3, 16, 16) uint8; (E, P, T, 2) uint8; (E, T, 8) float32
        self.frames = torch.from_numpy(raw["frames"]).to(device)
        self.keys = torch.from_numpy(raw["keys"]).to(torch.int32).to(device)
        self.physics = torch.from_numpy(raw["physics"]).to(device)
        self.device = device
        self.n_episodes, self.n_players, self.n_frames = self.frames.shape[:3]

        self.audio = None
        if with_audio:
            from examples.pong.pong_audio import SAMPLES_PER_FRAME, synthesize_audio

            assert "events" in raw, "dataset has no event flags; re-run generate_data"
            waves = synthesize_audio(raw["events"])  # (E, T * SAMPLES_PER_FRAME)
            self.audio = (
                torch.from_numpy(waves).reshape(self.n_episodes, self.n_frames, SAMPLES_PER_FRAME).to(device)
            )
        self.events = torch.from_numpy(raw["events"]).to(device) if "events" in raw else None

    def _windows(self, batch_size: int, clip_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep = torch.randint(self.n_episodes, (batch_size,), device=self.device)
        t0 = torch.randint(self.n_frames - clip_len + 1, (batch_size,), device=self.device)
        t = t0[:, None] + torch.arange(clip_len, device=self.device)[None]
        return ep, t

    def _actions(self, key_presses: torch.Tensor) -> ActionTensors:
        b, t = key_presses.shape[:2]
        actions = ActionTensors(config=PONG_ACTIONS, batch_size=b)
        actions.key_presses = key_presses
        actions.mouse_movements = torch.zeros(b, t, 2, device=key_presses.device)
        actions.game_mouse_sensitivity = torch.full(
            (b,), float("nan"), device=key_presses.device
        )
        return actions

    def _wrap(self, video: torch.Tensor, keys: torch.Tensor, ep: torch.Tensor, t: torch.Tensor,
              rows_per_sample: int = 1) -> VideoActionBatch:
        """Build the batch container, attaching audio (shared across a sample's rows) if loaded."""
        if self.audio is None:
            return VideoActionBatch(video=video, actions=self._actions(keys))
        from examples.pong.pong_audio import AVBatch

        audio = self.audio[ep[:, None], t]  # (b, T, SAMPLES_PER_FRAME)
        if rows_per_sample > 1:  # duplicate the shared world audio onto each player's row
            audio = audio.repeat_interleave(rows_per_sample, dim=0)
        return AVBatch(video=video, actions=self._actions(keys), audio=audio)

    def sample_views(self, batch_size: int, clip_len: int) -> VideoActionBatch:
        """Single-player batch: random (episode, view, start) clips with the view's own actions."""
        ep, t = self._windows(batch_size, clip_len)
        view = torch.randint(self.n_players, (batch_size,), device=self.device)
        video = self.frames[ep[:, None], view[:, None], t]
        keys = self.keys[ep[:, None], view[:, None], t]
        return self._wrap(video, keys, ep, t)

    def sample_multiplayer(self, batch_size: int, clip_len: int) -> VideoActionBatch:
        """Multiplayer batch: both views of each sampled window, players contiguous in the batch.

        Rows are ordered [ep0/p0, ep0/p1, ep1/p0, ...] with a total of ``batch_size * n_players``
        rows, matching MultiWrapperWorldModel's grouping invariant.
        """
        ep, t = self._windows(batch_size, clip_len)
        video = self.frames[ep[:, None, None], torch.arange(self.n_players, device=self.device)[None, :, None], t[:, None, :]]
        keys = self.keys[ep[:, None, None], torch.arange(self.n_players, device=self.device)[None, :, None], t[:, None, :]]
        video = video.flatten(0, 1)  # (b*p, T, 3, H, W)
        keys = keys.flatten(0, 1)  # (b*p, T, 2)
        return self._wrap(video, keys, ep, t, rows_per_sample=self.n_players)

    def eval_clip(self, episode: int, t0: int, clip_len: int, multiplayer: bool) -> VideoActionBatch:
        """A deterministic clip for evaluation, in single- or multi-player layout."""
        sl = slice(t0, t0 + clip_len)
        if multiplayer:
            video = self.frames[episode, :, sl]
            keys = self.keys[episode, :, sl]
        else:
            video = self.frames[episode, :1, sl]
            keys = self.keys[episode, :1, sl]
        if self.audio is None:
            return VideoActionBatch(video=video.clone(), actions=self._actions(keys.clone()))
        from examples.pong.pong_audio import AVBatch

        audio = self.audio[episode, sl][None].repeat(video.shape[0], 1, 1)
        return AVBatch(video=video.clone(), actions=self._actions(keys.clone()), audio=audio.clone())
