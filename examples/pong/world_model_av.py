"""Audio-video Pong world model: joint diffusion over video + audio latents.

The **pointwise-sum trick**: each latent frame's 8-channel audio latent (from the tiny
:class:`~examples.pong.pong_audio.AudioCodec`) is broadcast over the 4x4 spatial grid and
concatenated to the video latent channels. Because mira's ``DiffusionTransformer`` embeds tokens
with a single linear projection, this is mathematically a learned audio embedding **pointwise
summed** into every spatial token of that frame — the same additive-fusion style the architecture
already uses for actions and flow time. The flow-matching loss, diffusion forcing, KV-cache
streaming, and multiplayer tiling then treat the extra channels like any other latent dimension:
video and audio are denoised jointly, and sounds stay causally consistent with what happens on
screen (a paddle hit generates its beep in the same latent step as the bounce).

Nothing in mira's transformer changes; only the world model's latent width (8 video + 8 audio
channels) and the encode/decode endpoints differ from the video-only model, so the trained
video-only multiplayer checkpoint warm-starts everything except the input/output projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mira.world_model.config import LatentWorldModelConfig
from mira.world_model.diffusion_transformer import DiffusionTransformer
from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModelConfig

from examples.pong.pong_audio import CHUNK, AudioCodec, AVBatch
from examples.pong.world_model import PongMultiWrapper, PongWorldModel


class PongAVWorldModel(PongWorldModel):
    """PongWorldModel with audio latent channels appended to every spatial token."""

    def __init__(self, config: LatentWorldModelConfig, audio_codec_checkpoint: str) -> None:
        super().__init__(config)

        self.audio_codec = AudioCodec.load_from_checkpoint(audio_codec_checkpoint, device="cpu")
        self.audio_codec.requires_grad_(False)
        self.audio_codec.eval()
        info = self.audio_codec.info_from_checkpoint
        self.audio_mean, self.audio_std = info["audio_latent_mean_std"]

        # Extend the latent width and rebuild the parts whose shapes depend on it (the DiT's
        # input/output projections and the beginning-of-sequence latent). Everything else keeps
        # its video-only shape and can be warm-started from a video-only checkpoint.
        self.video_latent_dim = self.latent_dim
        self.latent_dim = self.video_latent_dim + self.audio_codec.latent_dim
        self.world_model = DiffusionTransformer(
            config,
            latent_dim=self.latent_dim,
            temporal_downsampling=self.temporal_downsampling,
            spatial_downsampling=self.spatial_downsampling,
        )
        if config.use_clean_past:
            self.bos = nn.Parameter(
                0.02
                * torch.randn(
                    self.world_model.latent_height, self.world_model.latent_width, self.latent_dim
                )
            )

    def encode_video(self, batch) -> Tensor:
        """Video latents (parent) + normalized audio latents broadcast over the spatial grid."""
        z_video = super().encode_video(batch)  # (b, t_lat, h, w, video_dim), normalized
        assert isinstance(batch, AVBatch), "PongAVWorldModel needs AVBatch batches (with_audio=True)"
        b, t_lat, h, w, _ = z_video.shape

        chunks = batch.audio[:, : t_lat * self.temporal_downsampling].reshape(b, t_lat, CHUNK)
        with torch.no_grad():
            z_audio = self.audio_codec.encode(chunks)  # (b, t_lat, audio_dim)
        z_audio = (z_audio - self.audio_mean) / self.audio_std
        z_audio = z_audio[:, :, None, None, :].expand(b, t_lat, h, w, self.audio_codec.latent_dim)
        return torch.cat([z_video, z_audio], dim=-1)

    def decode_to_video(self, z: Tensor) -> Tensor:
        return super().decode_to_video(z[..., : self.video_latent_dim])

    def decode_audio(self, z: Tensor) -> Tensor:
        """(b, t_lat, h, w, c) joint latents -> (b, t_lat * td, SAMPLES_PER_FRAME) waveform.

        The audio channels are averaged over the whole spatial grid (they were broadcast at encode
        time, so at generation this is a small ensemble over the denoised copies).
        """
        z_audio = z[..., self.video_latent_dim :].float().mean(dim=(2, 3))
        z_audio = z_audio * self.audio_std + self.audio_mean
        wave = self.audio_codec.decode(z_audio)  # (b, t_lat, CHUNK)
        b, t_lat, _ = wave.shape
        return wave.reshape(b, t_lat * self.temporal_downsampling, -1).clamp(-1, 1)


class PongAVMultiWrapper(PongMultiWrapper):
    """MultiWrapper over the AV world model (same tiling; audio channels ride along)."""

    def __init__(self, config: MultiWrapperWorldModelConfig, audio_codec_checkpoint: str) -> None:
        nn.Module.__init__(self)  # replicate the parent __init__ with the AV inner model
        self.n_players = config.n_players

        original_height = config.wm_config.video.height
        wm_config = config.wm_config.model_copy(deep=True)
        wm_config.video.height *= config.n_players
        self.single_world_model = PongAVWorldModel(wm_config, audio_codec_checkpoint)
        self.single_world_model.codec.config.encoder.video.height = original_height

        action_dim = self.single_world_model.action_encoder.dim
        self.player_embedding = nn.Parameter(torch.randn(config.n_players, action_dim) * 0.02)
        self.player_action_projection = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, action_dim))

    def decode_audio(self, z: Tensor) -> Tensor:
        """Decode audio from (possibly per-player-split) latents, averaging over players."""
        wave = self.single_world_model.decode_audio(z)
        if wave.shape[0] > 1:  # per-player rows carry the same world audio; average them
            wave = wave.mean(dim=0, keepdim=True)
        return wave
