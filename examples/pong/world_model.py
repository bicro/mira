"""Pong world models: mira's LatentWorldModel / MultiWrapperWorldModel over the toy codec.

The only change relative to mira is which codec gets loaded — the diffusion transformer, action
encoder, diffusion-forcing loss, PSD self-distillation, streaming KV-cache rollout, and the
multiplayer tiling/warm-start logic are all reused unmodified.

- :class:`PongWorldModel` re-implements ``LatentWorldModel.__init__`` with
  :class:`~examples.pong.codec.PongCodec` in place of the DINOv3 ``VideoCodec``; every method is
  inherited.
- :class:`PongMultiWrapper` does the same for ``MultiWrapperWorldModel`` (its ``__init__`` builds
  the inner single-player model directly), inheriting the height-tiled two-view forward, the
  per-player action combination, and the single-player warm-start ``load_state_dict``.
"""

from __future__ import annotations

import torch.nn as nn

from mira.world_model.config import LatentWorldModelConfig
from mira.world_model.diffusion_transformer import DiffusionTransformer
from mira.world_model.latent_world_model import LatentWorldModel
from mira.world_model.layers.action_encoder import ActionEncoder
from mira.world_model.multi_wrapper_world_model import (
    MultiWrapperWorldModel,
    MultiWrapperWorldModelConfig,
)

from examples.pong.codec import PongCodec
from examples.pong.data import PONG_ACTIONS

import torch

from mira.ml import ImageConfig


class PongWorldModel(LatentWorldModel):
    """LatentWorldModel with the toy Pong codec. Mirrors the parent ``__init__`` line for line,
    swapping only the codec loading; all training/inference methods are inherited."""

    def __init__(self, config: LatentWorldModelConfig) -> None:
        nn.Module.__init__(self)  # skip LatentWorldModel.__init__; replicate it below
        self.config = config

        assert config.codec_checkpoint is not None, "PongWorldModel needs a codec checkpoint"
        self.codec = PongCodec.load_from_checkpoint(config.codec_checkpoint, device="cpu")
        self.codec.requires_grad_(False)

        self.codec.config.encoder.video.height = self.config.video.height
        self.codec.config.encoder.video.width = self.config.video.width
        self.codec.eval()

        if config.latent_mean_std is not None:
            self.latent_mean, self.latent_std = config.latent_mean_std
        elif self.codec.info_from_checkpoint and "latent_mean_std" in self.codec.info_from_checkpoint:
            self.latent_mean, self.latent_std = self.codec.info_from_checkpoint["latent_mean_std"]
        else:
            raise ValueError("Codec latent mean/std not found in config or checkpoint.")

        self.latent_dim = self.codec.latent_dim
        self.temporal_downsampling = self.codec.temporal_downsampling
        self.spatial_downsampling = self.codec.spatial_downsampling
        self.n_context_latents = config.n_context_frames // self.temporal_downsampling
        self.n_context_frames = config.n_context_frames
        assert config.n_context_frames < config.video.timesteps

        self.world_model = DiffusionTransformer(
            config,
            latent_dim=self.latent_dim,
            temporal_downsampling=self.temporal_downsampling,
            spatial_downsampling=self.spatial_downsampling,
        )

        self.action_encoder = ActionEncoder(
            num_key_presses=len(config.actions.valid_keys),
            dim=config.hidden_dim,
            temporal_downsampling=self.action_temporal_downsampling,
            dropout_prob=config.dropout_action_prob,
            learned_temporal_pool=config.learned_temporal_pool,
            dropout_action_per_player=config.dropout_action_per_player,
            key_field_names=config.actions.valid_keys,
            subset_drop_prob=config.action_subset_drop_prob,
        )
        self.bos = None
        if config.use_clean_past:
            self.bos = nn.Parameter(
                0.02
                * torch.randn(
                    self.world_model.latent_height, self.world_model.latent_width, self.latent_dim
                )
            )


class PongMultiWrapper(MultiWrapperWorldModel):
    """MultiWrapperWorldModel whose inner model is a :class:`PongWorldModel`.

    Mirrors the parent ``__init__`` (tiled-height config copy, per-player embedding + projection);
    the tiled forward, action combination, streaming inference, and single-player warm-start
    ``load_state_dict`` are inherited unchanged.
    """

    def __init__(self, config: MultiWrapperWorldModelConfig) -> None:
        nn.Module.__init__(self)  # skip MultiWrapperWorldModel.__init__; replicate it below
        self.n_players = config.n_players

        original_height = config.wm_config.video.height
        wm_config = config.wm_config.model_copy(deep=True)
        wm_config.video.height *= config.n_players
        self.single_world_model = PongWorldModel(wm_config)
        self.single_world_model.codec.config.encoder.video.height = original_height

        action_dim = self.single_world_model.action_encoder.dim
        self.player_embedding = nn.Parameter(torch.randn(config.n_players, action_dim) * 0.02)
        self.player_action_projection = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, action_dim))

    def streaming_inference_step(  # noqa: D417 -- extends the parent with drop_players
        self,
        z,
        actions_history,
        streaming_kv_cache=None,
        config=None,
        drop_players: torch.Tensor | None = None,
    ):
        """Parent step plus ``drop_players``: a ``(b * n_players,)`` bool mask of players whose
        actions are replaced by the learned absent token, so the world model itself drives them
        (the paper's inference-time use of per-player action dropout). Mirrors the parent body,
        threading the mask into the action encoder."""
        from mira.world_model.config import WorldModelInferenceConfig

        if config is None:
            config = WorldModelInferenceConfig()
        swm = self.single_world_model
        z_t = torch.cat([z[:, 1:], torch.randn_like(z[:, :1])], dim=1)

        off = swm.action_temporal_downsampling - 1
        n_action_steps = (z.shape[1] - 1) * swm.action_temporal_downsampling
        a_flat = swm.action_encoder(
            actions_history.slice_time(-n_action_steps - off, -off if off else None).to(self.device),
            drop_mask=drop_players,
        )
        current_a = self._combine_player_actions(a_flat)

        return swm.denoise_streaming(
            z_t,
            current_a,
            streaming_kv_caches=streaming_kv_cache,
            n_diffusion_steps=config.n_diffusion_steps,
            noise_level=config.noise_level,
            schedule_type=config.schedule_type,
        )


def pong_wm_config(
    codec_checkpoint: str,
    multiplayer: bool,
    clip_len: int = 64,
    psd_loss_prob: float = 0.0,
) -> LatentWorldModelConfig:
    """The toy world-model configuration, mirroring the released MIRA config at 1/500 scale.

    Same knobs as ``configs/model/latent_world_model.yaml``: flow matching with diffusion forcing,
    clean-past conditioning, AdaLN on attention + MLP, attention gating, learned temporal action
    pooling (2 actions pooled per 10 Hz latent), RoPE. Sizes are scaled down: hidden 256 / 8 layers
    / 8 heads (GQA 8:2) instead of 4096 / 16 / 32 (GQA 32:8).
    """
    return LatentWorldModelConfig(
        actions=PONG_ACTIONS,
        video=ImageConfig(height=16, width=16, channels=3, timesteps=clip_len, fps=20),
        codec_checkpoint=codec_checkpoint,
        causal=True,
        hidden_dim=256,
        n_head=8,
        n_kv_head=2,
        n_layers=8,
        time_attention_every=2,
        patch_size=1,
        n_register_tokens=0,
        attention_gating=True,
        ada_attn_ln=True,
        use_clean_past=True,
        learned_temporal_pool=True,
        n_context_frames=38,  # inference: 19 context latents + 1 generated, like the paper's T=20
        dropout_action_prob=0.1,
        # Per-player action dropout only makes sense in the multiplayer stage; subset-drop is a
        # Rocket-League-specific key split, so a dropped Pong player always drops all their keys.
        dropout_action_per_player=multiplayer,
        action_subset_drop_prob=0.0,
        psd_loss_prob=psd_loss_prob,
    )
