# Pong world model — MIRA's pipeline at 1/1000 scale

A toy reproduction of the full MIRA recipe (multiplayer, real-time, latent-diffusion world model)
on 16×16 two-player Pong, small enough to train end-to-end on one consumer GPU in ~3 hours.
Wherever possible it **reuses mira's actual code** — the diffusion transformer, action encoder,
diffusion-forcing loss, PSD self-distillation, streaming KV-cache rollout, ViT codec decoder, and
the entire multiplayer wrapper run unmodified; only the environment, the tiny frozen feature
extractor, and the training scripts are new.

## Paper → toy mapping

| MIRA (paper) | This toy |
|---|---|
| Rocket League 2v2, 4 players, 9-key keyboard | Pong, 2 players, 2-key vocabulary (Up/Down; none = stay) |
| 10,000 h of Nexto bot self-play + action-noise injection | ~5.5 h (780k frames) of scripted ball-tracking bots with reaction delay + action noise |
| 4 per-player camera views of a shared match | 2 mirrored first-person views (own paddle always left/blue) |
| 3 arenas | 3 background tints |
| In-game clock/score HUD coherence | Score-pip row (top row of the frame) |
| Privileged physics state logged for eval | Ball pos/vel + paddle positions + score logged per frame |
| Frozen **DINOv3-L** feature extractor | Tiny 0.2M ViT pretrained with masked reconstruction (SimMIM-style), then frozen |
| RAE codec: multi-layer feature aggregation → linear 2×2×2 bottleneck → causal space-time ViT decoder (`mean(S)+last`, 288×512 → /32, 10 Hz, 32 ch) | Same, tiny: 16×16@20fps → 4×4×8ch latents @10 Hz (`mira.codec.ViTVideoDecoder`, reused) |
| Codec loss: L1 + LPIPS + DINO feature-consistency, VQ-GAN adaptive weighting, no GAN/KL | L1 + extractor feature-consistency, same `calculate_adaptive_weight` (reused), no GAN/KL |
| 5B DiT: flow matching + diffusion forcing, factorized space-time attention, AdaLN(actions+τ), GQA, RoPE, clean-past conditioning | Same class (`mira.world_model.DiffusionTransformer`, reused) at 10M: hidden 256, 8 layers, GQA 8:2 |
| Multiplayer: views tiled along height, per-player action embeddings combined, per-player action dropout, SP→MP warm start | Same class (`MultiWrapperWorldModel`, reused) with `n_players=2` |
| Single-player pretrain 30k → multiplayer 100k steps | `sp` 20k → `mp` 60k steps |
| PSD-M few-step self-distillation | Same code path, `psd` stage 10k steps |
| Streaming: rolling 20-latent window, KV-cache, few-step sampling, decode 2 frames/latent | Same methods (`denoise_streaming` / `streaming_inference_step`, reused) |
| 20 fps on one B200 (~70 ms per latent step) | 20 fps on one RTX 5080 (~73 ms per latent step, `torch.compile`) |
| Live demo at mira-wm.com | `python -m examples.pong.play` (pygame) |
| Game-state probing + Action Recoverability Ratio | `eval_probe.py`: ridge-probe R² + toy ARR |

## Quickstart

```bash
# 1. Data (~30 s): 400 episodes of bot-vs-bot play, two views + actions + physics state
python -m examples.pong.generate_data

# 2. Codec (~30 min): SSL-pretrain the tiny feature extractor, then train bottleneck + decoder
python -m examples.pong.train_codec

# 3. World model (~2.5 h): single-player -> multiplayer warm start -> PSD distillation
python -m examples.pong.train_wm --stage all

# 4. Play inside the model (W/S vs Up/Down; keys 1/2 hand a player to the model itself)
python -m examples.pong.play

# 5. Probes: physics readout R^2 and action recoverability
python -m examples.pong.eval_probe
```

Requires the base repo deps plus `pygame` (demo) and `triton-windows` on Windows (for
`torch.compile`; pass `--no-compile` to skip).

## Results (RTX 5080, ~4.5 h total)

| Metric | Value |
|---|---|
| Codec reconstruction (held-out) | 36.4 dB PSNR |
| Single-player rollout (20k steps) | ~20 dB (plateaus — opponent unpredictable without their actions) |
| Multiplayer rollout (60k steps, warm-started) | **35.6 dB** — saturates the codec ceiling |
| Toy ARR (rollout obeys commanded actions) | **0.998** (real-video baseline = 1.0) |
| Physics probe R² from codec latents | paddles 0.98 / ball position 0.77 / ball velocity 0.26–0.40 |
| Few-step sampling (6-clip avg) | mp: 26.6 dB @ 2 steps; psd: flat 24.7–25.6 dB across 1–10 steps |
| Streaming speed (2-step, compiled, incl. decode) | **11 ms per latent step ≈ 177 fps equivalent** |

Two paper findings reproduce in miniature: the multiplayer model beats the single-player one once
both action streams condition the prediction (Section 6.6), and the diffusion-forcing model stays
usable at very few flow-matching steps even before distillation (Figure 11).

Note: the PSD stage needs `activation_checkpointing="psd-only"` and batch 16 to fit in 16 GB — the
PSD student pass doubles live activations and silently thrashes into Windows shared memory
otherwise.

## Audio extension: joint video + sound-effect generation

The world model can also generate **sound effects** (paddle-movement ticks, wall/paddle collision
beeps, the score jingle) jointly with the video, via the **pointwise-sum trick**:

1. The env logs sound events per frame; classic-Pong square waves are synthesized at 8 kHz.
2. A tiny neural audio codec (`pong_audio.AudioCodec`, the Mimi analog) compresses each 100 ms
   chunk — one latent frame of time — into 8 channels.
3. The audio latent is broadcast over the 4x4 spatial grid and concatenated to the video latent
   channels. Since the DiT embeds tokens with one linear projection, this is exactly a learned
   audio embedding **pointwise-summed into every spatial token** — the same additive fusion the
   architecture uses for actions and flow time. Flow matching + diffusion forcing then denoise
   video and audio jointly; mira's transformer is unchanged.
4. Warm-started from the video-only `wm_mp.pt` (only the input/output projections and bos are
   width-changed), so it trains in 25k steps.

```bash
python -m examples.pong.train_av    # audio codec (~2 min) + AV world model (~70 min)
python -m examples.pong.play_av     # play with generated sound
```

Audio eval: matched-filter detection of collision/score beeps in the *generated* waveform, scored
against the ground-truth events of the same rollout window (recall within +-2 frames + false
alarms/s) — the audio analog of the toy ARR.

## Files

- `pong_env.py` — vectorized 16×16 Pong + scripted bots (data collection analog of Section 3)
- `generate_data.py` / `data.py` — dataset generation and GPU-resident batch sampling
- `codec.py` / `train_codec.py` — the representation-autoencoder codec (Section 4.2)
- `world_model.py` / `train_wm.py` — world model + 3-stage training (Sections 4.3–4.5)
- `play.py` — real-time interactive demo (Section 5)
- `eval_probe.py` — physics probe + toy ARR (Section 6.2)
