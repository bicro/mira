"""Evaluation probes for the Pong world model, mirroring the paper's Section 6.2 at toy scale.

1. **Physics probe** — a ridge-regression readout from the frozen codec latents to the privileged
   game state (ball position/velocity, paddle positions). High R^2 means the codec's latent space
   encodes physically meaningful quantities, the paper's game-state probing.
2. **Toy Action Recoverability Ratio (ARR)** — roll the world model out under commanded actions and
   recover each player's action from the *generated* video (did their paddle actually move the way
   they pressed?). Reported relative to the same recovery run on real video, so 1.0 means the
   rollout obeys actions as faithfully as the real game renders them.

Usage:
    python -m examples.pong.eval_probe [--checkpoint examples/pong/runs/wm/wm_mp.pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mira.world_model.config import WorldModelInferenceConfig

from examples.pong.codec import PongCodec
from examples.pong.data import PongData
from examples.pong.pong_env import OWN_COLOR, PADDLE_Y_MAX, PADDLE_Y_MIN
from examples.pong.train_wm import CLIP_LEN, build_model

DEVICE = "cuda"
PROBE_TARGETS = ["ball_y", "ball_x", "vel_y", "vel_x", "paddle1_y", "paddle2_y"]


@torch.no_grad()
def physics_probe(codec: PongCodec, train: PongData, test: PongData) -> dict[str, float]:
    """Closed-form ridge regression from codec latents to game state; R^2 per variable."""

    def collect(data: PongData, n_clips: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for i in range(n_clips):
            episode = i % data.n_episodes
            t0 = (i * 97) % (data.n_frames - CLIP_LEN)
            batch = data.eval_clip(episode, t0, CLIP_LEN, multiplayer=False)
            codec.preprocess_batch(batch)
            _, enc = codec.encode(batch.video, trim_video=False)
            z = enc.z[0].flatten(1)  # (t_lat, 128)
            # Each latent covers 2 frames; probe the state at its second frame.
            state = data.physics[episode, t0 + 1 : t0 + CLIP_LEN : 2, :6]
            xs.append(z.float())
            ys.append(state)
        return torch.cat(xs), torch.cat(ys)

    x_train, y_train = collect(train, 600)
    x_test, y_test = collect(test, 60)
    x_train = torch.cat([x_train, torch.ones_like(x_train[:, :1])], dim=1)
    x_test = torch.cat([x_test, torch.ones_like(x_test[:, :1])], dim=1)

    ridge = 1e-3 * torch.eye(x_train.shape[1], device=DEVICE)
    w = torch.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ y_train)
    pred = x_test @ w
    ss_res = ((pred - y_test) ** 2).sum(0)
    ss_tot = ((y_test - y_test.mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).cpu().numpy()
    return {name: round(float(v), 3) for name, v in zip(PROBE_TARGETS, r2)}


def recover_paddle_y(video01: torch.Tensor) -> torch.Tensor:
    """Recover the own-paddle top row from frames (T, 3, 16, 16) in [0,1] by template-scoring
    column 0 (the own paddle is always the left column of a player's view)."""
    col = video01[:, :, :, 0]  # (T, 3, 16)
    own = torch.tensor(OWN_COLOR, device=video01.device).float()[None, :, None] / 255
    blueness = -(col - own).abs().sum(1)  # (T, 16) higher = more own-colored
    scores = torch.stack(
        [blueness[:, y : y + 4].sum(-1) for y in range(PADDLE_Y_MIN, PADDLE_Y_MAX + 1)], dim=-1
    )
    return scores.argmax(-1) + PADDLE_Y_MIN  # (T,)


@torch.no_grad()
def action_recoverability(model, test: PongData, n_clips: int = 10) -> dict[str, float]:
    """Toy ARR: does each player's paddle in the *generated* video move as commanded?"""
    swm = model.single_world_model
    n_ctx = swm.n_context_frames

    def accuracy(video01: torch.Tensor, keys: torch.Tensor) -> tuple[int, int]:
        """video01 (P, T, 3, 16, 16), keys (P, T, 2). Counts (matches, total) over moves."""
        matches = total = 0
        for p in range(2):
            paddle = recover_paddle_y(video01[p])
            observed = (paddle[1:] - paddle[:-1]).clamp(-1, 1)
            commanded = (keys[p, :-1, 1] - keys[p, :-1, 0]).long()  # Down - Up in {-1, 0, 1}
            # Wall clamp: pushing into a wall legitimately produces no movement; skip those.
            clamped = ((paddle[:-1] == PADDLE_Y_MIN) & (commanded == -1)) | (
                (paddle[:-1] == PADDLE_Y_MAX) & (commanded == 1)
            )
            valid = ~clamped
            matches += int((observed == commanded)[valid].sum())
            total += int(valid.sum())
        return matches, total

    gen_m = gen_t = real_m = real_t = 0
    for i in range(n_clips):
        episode = i % test.n_episodes
        t0 = (i * 131) % (test.n_frames - CLIP_LEN)
        batch = test.eval_clip(episode, t0, CLIP_LEN, multiplayer=True)
        keys = batch.actions.key_presses.clone()
        real01 = batch.video.clone().float() / 255.0

        outputs = model.inference(batch, WorldModelInferenceConfig(n_diffusion_steps=10), progress_bar=False)
        pred = outputs.output_video  # (1, T, 3, 2*16, 16) tiled
        pred01 = torch.stack([pred[0, :, :, :16], pred[0, :, :, 16:]]).float().clamp(0, 1)

        m, t = accuracy(pred01[:, n_ctx:], keys[:, n_ctx:])
        gen_m, gen_t = gen_m + m, gen_t + t
        m, t = accuracy(real01[:, n_ctx:], keys[:, n_ctx:])
        real_m, real_t = real_m + m, real_t + t

    acc_gen, acc_real = gen_m / max(1, gen_t), real_m / max(1, real_t)
    return {
        "action_accuracy_rollout": round(acc_gen, 3),
        "action_accuracy_real": round(acc_real, 3),
        "ARR": round(acc_gen / max(1e-6, acc_real), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="world-model checkpoint; defaults to wm_av.pt, else wm_mp.pt")
    parser.add_argument("--audio-codec", type=str, default="examples/pong/runs/av/audio_codec.pt")
    parser.add_argument("--codec", type=str, default="examples/pong/runs/codec/codec.pt")
    parser.add_argument("--data", type=Path, default=Path("examples/pong/runs/data"))
    args = parser.parse_args()

    if args.checkpoint is None:
        for candidate in ("examples/pong/runs/av/wm_av.pt", "examples/pong/runs/wm/wm_mp.pt"):
            if Path(candidate).exists():
                args.checkpoint = Path(candidate)
                break
        assert args.checkpoint is not None, "no world-model checkpoint found"

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    stage = checkpoint.get("stage")
    # The audio-video model diffuses audio channels too, so it needs audio-carrying batches.
    with_audio = stage == "av"

    train = PongData(args.data / "train.npz", device=DEVICE, with_audio=with_audio)
    test = PongData(args.data / "test.npz", device=DEVICE, with_audio=with_audio)

    codec = PongCodec.load_from_checkpoint(args.codec, device=DEVICE).eval()
    print("physics probe R^2 (codec latents -> game state):")
    print("  ", physics_probe(codec, train, test))

    if stage == "av":
        from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModelConfig

        from examples.pong.world_model import pong_wm_config
        from examples.pong.world_model_av import PongAVMultiWrapper

        config = pong_wm_config(args.codec, multiplayer=True, clip_len=CLIP_LEN)
        model = PongAVMultiWrapper(
            MultiWrapperWorldModelConfig(n_players=2, wm_config=config), args.audio_codec
        ).to(DEVICE)
    else:
        model = build_model("psd" if stage == "psd" else "mp", args.codec)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"action recoverability ({args.checkpoint.name}, stage {stage}):")
    print("  ", action_recoverability(model, test))


if __name__ == "__main__":
    main()
