"""The toy representation-autoencoder codec for 16x16 Pong, mirroring MIRA's RAEv2 design.

Structure (paper Section 4.2, at 1/1000 scale):

- :class:`PongFeatureExtractor` — a tiny per-frame ViT playing the role of the **frozen DINOv3-L**.
  It is pretrained self-supervised on Pong frames (SimMIM-style masked reconstruction, see
  ``train_codec.py``), then frozen. Like the paper, the codec aggregates several intermediate
  blocks: ``mean(features at S) + features[last]``.
- :class:`PongEncoder` — the frozen extractor plus a strided-conv **bottleneck** (a Conv3d with
  kernel = stride = 2x2x2), compressing 2x2 in space and 2x in time: 16x16 pixels @ 20 fps ->
  a 4x4 latent grid with ``latent_dim`` channels @ 10 Hz. Mirrors ``mira.codec.RAEEncoder``.
- The decoder is **mira's own** :class:`~mira.codec.vit_decoder.ViTVideoDecoder` (causal
  space-time ViT), just configured tiny.
- :class:`PongCodec` — duck-types :class:`mira.codec.codec_model.VideoCodec` (same ``encode`` /
  ``decode`` / ``preprocess_batch`` surface, downsampling factors, and ``info_from_checkpoint``
  with the latent mean/std), so mira's ``LatentWorldModel`` and ``MultiWrapperWorldModel`` can use
  it unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch
from einops import rearrange
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from mira.codec.config import StridedConvBottleneckConfig, ViTDecoderConfig
from mira.codec.rae_encoder import RAEEncoderOutputs
from mira.codec.vit_decoder import SwiGLU, ViTVideoDecoder, spatial_rope
from mira.data.batch import VideoActionBatch
from mira.ml import ImageConfig
from mira.ml.attention import SelfAttention, SelfAttentionConfig
from mira.ml.init import init_weights


class PongExtractorConfig(BaseModel):
    """The tiny frozen feature extractor (the toy DINOv3)."""

    model_config = ConfigDict(extra="forbid")

    width: int = 64
    depth: int = 4
    num_heads: int = 4
    patch_size: int = 2
    # Blocks aggregated as mean(S) + last, mirroring the paper's multi-layer feature readout.
    aggregation_layers: list[int] = Field(default_factory=lambda: [1, 2, 3])
    rope_theta: float = 100.0


class PongEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latent_dim: int = 8
    video: ImageConfig
    extractor: PongExtractorConfig = Field(default_factory=PongExtractorConfig)
    bottleneck: StridedConvBottleneckConfig = Field(
        default_factory=lambda: StridedConvBottleneckConfig(stride=2, temporal_stride=2)
    )


class PongCodecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder: PongEncoderConfig
    decoder: ViTDecoderConfig


class PongFeatureExtractor(nn.Module):
    """Per-frame ViT feature extractor; pretrained with masked reconstruction, then frozen.

    Takes frames in ``[0, 1]`` (like mira's DinoModel), returns per-block feature maps.
    """

    def __init__(self, config: PongExtractorConfig):
        super().__init__()
        self.config = config
        w, p = config.width, config.patch_size
        self.head_dim = w // config.num_heads
        self.patch_embed = nn.Conv2d(3, w, kernel_size=p, stride=p)
        attn_config = SelfAttentionConfig(embed_dim=w, num_heads=config.num_heads, num_kv_heads=None)
        self.norms1 = nn.ModuleList(nn.LayerNorm(w) for _ in range(config.depth))
        self.attns = nn.ModuleList(SelfAttention(attn_config, causal=False) for _ in range(config.depth))
        self.norms2 = nn.ModuleList(nn.LayerNorm(w) for _ in range(config.depth))
        self.mlps = nn.ModuleList(SwiGLU(w, dim_multiplier=4, multiple_of=32) for _ in range(config.depth))

        # SimMIM-style pretraining pieces: a learned token for masked patches and a pixel head.
        self.mask_token = nn.Parameter(0.02 * torch.randn(1, 1, w))
        self.pixel_head = nn.Linear(w, 3 * p * p)

        self.apply(init_weights)

    def forward_tokens(self, frames01: Tensor, patch_mask: Tensor | None = None) -> list[Tensor]:
        """Run the ViT on frames ``(N, 3, H, W)`` in [0, 1]; returns per-block tokens (N, h*w, C).

        ``patch_mask`` (N, h*w) bool marks patches replaced by the learned mask token (pretraining).
        """
        x = self.patch_embed(frames01 - 0.5)
        h, w = x.shape[-2:]
        x = rearrange(x, "n c h w -> n (h w) c")
        if patch_mask is not None:
            x = torch.where(patch_mask[..., None], self.mask_token.to(x.dtype), x)
        rope = spatial_rope(h, w, self.head_dim, self.config.rope_theta, x.device)
        features = []
        for norm1, attn, norm2, mlp in zip(self.norms1, self.attns, self.norms2, self.mlps):
            x = x + attn(norm1(x), rotary_emb=rope)
            x = x + mlp(norm2(x))
            features.append(x)
        return features

    def forward(self, video01: Tensor) -> list[Tensor]:
        """Feature maps for video ``(B, T, 3, H, W)`` in [0, 1]: list of (B, T, C, h, w)."""
        b, t = video01.shape[:2]
        frames = rearrange(video01, "b t c h w -> (b t) c h w")
        grid = frames.shape[-1] // self.config.patch_size
        features = self.forward_tokens(frames)
        return [
            rearrange(f, "(b t) (h w) c -> b t c h w", b=b, t=t, h=grid, w=grid) for f in features
        ]

    def reconstruct_pixels(self, tokens: Tensor, image_hw: tuple[int, int]) -> Tensor:
        """Pixel head for pretraining: tokens (N, h*w, C) -> frames (N, 3, H, W) in [0, 1] offsets."""
        p = self.config.patch_size
        h, w = image_hw[0] // p, image_hw[1] // p
        x = self.pixel_head(tokens)
        return rearrange(x, "n (h w) (c p1 p2) -> n c (h p1) (w p2)", h=h, w=w, c=3, p1=p, p2=p)


class PongEncoder(nn.Module):
    """Frozen feature extractor -> layer aggregation -> strided-conv bottleneck -> latent.

    Mirrors :class:`mira.codec.RAEEncoder` with the toy extractor in DINOv3's role.
    """

    def __init__(self, config: PongEncoderConfig):
        super().__init__()
        self.config = config
        bn = config.bottleneck
        self.rae_projection = nn.Conv3d(
            config.extractor.width,
            config.latent_dim,
            kernel_size=(bn.temporal_stride, bn.stride, bn.stride),
            stride=(bn.temporal_stride, bn.stride, bn.stride),
            bias=True,
        )
        self.apply(init_weights)
        self.extractor = PongFeatureExtractor(config.extractor)

    def freeze_extractor(self) -> None:
        self.extractor.requires_grad_(False)
        self.extractor.eval()

    def get_downsampling_factors(self) -> tuple[int, int]:
        return (
            self.config.bottleneck.temporal_stride,
            self.config.extractor.patch_size * self.config.bottleneck.stride,
        )

    def forward(self, video: Tensor) -> RAEEncoderOutputs:
        # PongCodec normalizes to [-1, 1]; the extractor expects [0, 1] (like DinoModel).
        video = (video + 1) / 2
        with torch.no_grad():
            features = self.extractor(video)

        agg_layers = self.config.extractor.aggregation_layers
        agg = torch.stack([features[i] for i in agg_layers]).mean(dim=0) + features[agg_layers[-1]]

        x = rearrange(agg, "b t c h w -> b c t h w")
        z = self.rae_projection(x)
        z = rearrange(z, "b c t h w -> b t c h w")
        return RAEEncoderOutputs(z=z, dino_features=tuple(features))

    def aggregate(self, video01: Tensor) -> Tensor:
        """Aggregated features of video in [0, 1] (used by the feature-consistency loss)."""
        features = self.extractor(video01)
        agg_layers = self.config.extractor.aggregation_layers
        return torch.stack([features[i] for i in agg_layers]).mean(dim=0) + features[agg_layers[-1]]


class PongCodec(nn.Module):
    """Duck-typed :class:`mira.codec.codec_model.VideoCodec` over the toy encoder/decoder."""

    def __init__(self, config: PongCodecConfig):
        super().__init__()
        self.config = config
        self.encoder = PongEncoder(config.encoder)
        self.decoder = ViTVideoDecoder(config.decoder)

        self.temporal_downsampling, self.spatial_downsampling = self.encoder.get_downsampling_factors()
        assert config.decoder.patch_size_t == self.temporal_downsampling
        assert config.decoder.bottleneck.stride * config.decoder.patch_size == self.spatial_downsampling
        assert config.decoder.latent_dim == config.encoder.latent_dim
        self.latent_dim = config.encoder.latent_dim

        self.info_from_checkpoint: dict | None = None

    def preprocess_batch(self, batch: VideoActionBatch) -> None:
        """uint8 -> [0, 1] float. The toy frames are already at the codec resolution."""
        video = batch.video / 255.0
        target = (self.config.encoder.video.height, self.config.encoder.video.width)
        assert video.shape[-2:] == target, f"expected {target} frames, got {tuple(video.shape[-2:])}"
        batch.video = video

    def normalize_video(self, x: Tensor, trim_video: bool = True) -> Tensor:
        if trim_video:
            x = x[:, : self.config.encoder.video.timesteps]
        return (x - 0.5) / 0.5  # [-1, 1]

    def forward(self, batch: VideoActionBatch, trim_video: bool = True):
        from mira.codec.codec_model import VideoCodecOutputs  # local: avoids cycles at import

        self.preprocess_batch(batch)
        input_video, encoder_output = self.encode(batch.video, trim_video=trim_video)
        output_video = self.decode(encoder_output.z)
        return VideoCodecOutputs(
            input_video=input_video,
            output_video=output_video,
            z=encoder_output.z,
            dino_features=encoder_output.dino_features,
        )

    def encode(self, video: Tensor, trim_video: bool = True) -> tuple[Tensor, RAEEncoderOutputs]:
        input_video = self.normalize_video(video, trim_video=trim_video)
        return input_video, self.encoder(input_video)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def save_checkpoint(self, checkpoint_path: str | Path, extra_info: dict | None = None) -> None:
        """Single-file checkpoint: weights + config + info (e.g. ``latent_mean_std``)."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.model_dump(),
                "info": extra_info or {},
            },
            checkpoint_path,
        )

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> PongCodec:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        codec = cls(PongCodecConfig.model_validate(checkpoint["config"]))
        codec.load_state_dict(checkpoint["state_dict"])
        codec.to(device)
        codec.info_from_checkpoint = checkpoint["info"]
        return codec


def default_codec_config(clip_len: int = 32) -> PongCodecConfig:
    """The toy codec configuration: 16x16 @ 20 fps -> 4x4 x 8ch latents @ 10 Hz."""
    video = ImageConfig(height=16, width=16, channels=3, timesteps=clip_len, fps=20)
    return PongCodecConfig(
        encoder=PongEncoderConfig(video=video),
        decoder=ViTDecoderConfig(
            video=video,
            out_channels=3,
            latent_dim=8,
            patch_size=2,  # ViT grid 8x8 -> 2x2-pixel patches
            patch_size_t=2,  # each latent decodes to 2 frames (10 Hz -> 20 fps)
            is_causal=True,
            bottleneck=StridedConvBottleneckConfig(stride=2, temporal_stride=2),
            vit_width=128,
            vit_depth=6,
            vit_num_heads=4,
            mlp_dim_multiplier=4,
        ),
    )
