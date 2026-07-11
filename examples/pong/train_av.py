"""Train the audio-video Pong world model.

Two stages, both cheap because everything possible is reused:

1. **Audio codec** (~2 min): the tiny chunk autoencoder over the synthesized sound effects, plus
   its latent mean/std (the audio analog of the video codec's latent statistics).
2. **AV world model** (~70 min): the multiplayer world model with 8 audio channels appended to
   every latent token (see ``world_model_av.py``), **warm-started from the trained video-only
   ``wm_mp.pt``** — only the DiT's input/output projections and the bos latent start fresh, so it
   converges in a fraction of the video model's schedule.

Eval combines the video rollout PSNR with an audio check: matched-filter detection of
collision/score effects in the *generated* waveform, scored against the ground-truth events of the
same window (recall within +-2 frames, plus false alarms per second).

Usage:
    python -m examples.pong.train_av [--steps 25000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mira.training.ema import ModelEMA
from mira.world_model.config import WorldModelInferenceConfig
from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModelConfig

from examples.pong.data import PongData
from examples.pong.pong_audio import EVAL_EVENTS, detect_events, train_audio_codec
from examples.pong.train_wm import CLIP_LEN, quick_eval, save_checkpoint
from examples.pong.world_model import pong_wm_config
from examples.pong.world_model_av import PongAVMultiWrapper

DEVICE = "cuda"


def ensure_audio_codec(data: PongData, out_dir: Path, steps: int = 3000) -> Path:
    path = out_dir / "audio_codec.pt"
    if path.exists():
        print("reusing trained audio codec")
        return path
    assert data.audio is not None
    codec = train_audio_codec(data.audio.flatten(1, 2), steps=steps, device=DEVICE)
    with torch.no_grad():
        sample = data.audio.flatten(1, 2)[:64].reshape(-1, 2, 400).reshape(-1, 800)
        z = codec.encode(sample)
    stats = [z.mean().item(), z.std().item() + 1e-6]
    print(f"audio latent mean/std: {stats[0]:.4f} / {stats[1]:.4f}")
    codec.save_checkpoint(path, extra_info={"audio_latent_mean_std": stats})
    return path


@torch.no_grad()
def audio_event_eval(model, test: PongData, n_clips: int = 4) -> dict[str, float]:
    """Do collision/score sounds appear in the generated audio when they should?"""
    model.eval()
    swm = model.single_world_model
    n_ctx = swm.n_context_frames
    recalls = {name: [0, 0] for name in EVAL_EVENTS}  # hits, total
    false_alarms, gen_seconds = 0, 0.0

    for i in range(n_clips):
        episode = i % test.n_episodes
        t0 = (i * 211) % (test.n_frames - CLIP_LEN)
        batch = test.eval_clip(episode, t0, CLIP_LEN, multiplayer=True)
        outputs = model.inference(batch, WorldModelInferenceConfig(n_diffusion_steps=10), progress_bar=False)
        wave = model.decode_audio(outputs.z_t)[0].flatten()  # generated audio, whole window
        detections = detect_events(wave)
        assert test.events is not None
        gt = test.events[episode, t0 : t0 + CLIP_LEN]

        for name, ch in EVAL_EVENTS.items():
            det = detections[name]
            gt_frames = torch.nonzero(gt[:, ch]).squeeze(1)
            gt_frames = gt_frames[gt_frames >= n_ctx]  # score only the generated region
            for f in gt_frames.tolist():
                lo, hi = max(0, f - 2), min(det.shape[0], f + 3)
                recalls[name][0] += int(det[lo:hi].any())
                recalls[name][1] += 1
            # False alarms: detections in the generated region with no GT event within 2 frames.
            gt_any = torch.zeros(CLIP_LEN, dtype=torch.bool, device=det.device)
            for f in torch.nonzero(gt[:, ch]).squeeze(1).tolist():
                gt_any[max(0, f - 2) : f + 3] = True
            det_frames = torch.nonzero(det).squeeze(1)
            det_frames = det_frames[det_frames >= n_ctx]
            false_alarms += int((~gt_any[det_frames]).sum())
        gen_seconds += (CLIP_LEN - n_ctx) / 20

    out = {
        f"recall_{name}": round(h / t, 3) if t else float("nan") for name, (h, t) in recalls.items()
    }
    out["false_alarms_per_s"] = round(false_alarms / gen_seconds, 2)
    model.train()
    return out


@torch.no_grad()
def av_consistency_eval(model, test: PongData, n_clips: int = 10) -> dict[str, float]:
    """Audio-video consistency *within the model's own imagination*.

    Track the ball in the generated video, detect its wall/paddle bounces there, and check that
    the generated audio contains the right beep within +-2 frames. This is the fair test for
    rare/divergent events: the rollout may lawfully differ from the ground truth, but its own
    sounds should match its own pictures.
    """
    model.eval()
    swm = model.single_world_model
    n_ctx = swm.n_context_frames
    hits = {"wall": [0, 0], "hit": [0, 0]}

    for i in range(n_clips):
        episode = i % test.n_episodes
        t0 = (i * 271) % (test.n_frames - CLIP_LEN)
        batch = test.eval_clip(episode, t0, CLIP_LEN, multiplayer=True)
        outputs = model.inference(batch, WorldModelInferenceConfig(n_diffusion_steps=10), progress_bar=False)
        video = outputs.output_video[0, :, :, :16]  # player-1 view (T, 3, 16, 16)
        wave = model.decode_audio(outputs.z_t)[0].flatten()
        detections = detect_events(wave)

        # Ball = pixel closest to white, excluding the HUD row and the paddle columns.
        whiteness = video[:, :, 1:, 1:15].float().sum(1)  # (T, 15, 14)
        flat = whiteness.flatten(1).argmax(1)
        ball_y = (flat // 14 + 1).float()
        ball_x = (flat % 14 + 1).float()
        dy, dx = ball_y.diff(), ball_x.diff()

        for t in range(max(n_ctx, 2), CLIP_LEN - 2):
            # Wall bounce in the generated video: vertical direction flip near the top/bottom.
            if dy[t - 1] * dy[t] < 0 and (ball_y[t] <= 3 or ball_y[t] >= 13):
                hits["wall"][0] += int(detections["wall"][t - 1 : t + 3].any())
                hits["wall"][1] += 1
            # Paddle hit: horizontal direction flip near either paddle column.
            if dx[t - 1] * dx[t] < 0 and (ball_x[t] <= 3 or ball_x[t] >= 12):
                hits["hit"][0] += int(detections["hit"][t - 1 : t + 3].any())
                hits["hit"][1] += 1

    model.train()
    return {
        f"av_consistency_{k}": round(h / t, 3) if t else float("nan") for k, (h, t) in hits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", type=Path, default=Path("examples/pong/runs/data"))
    parser.add_argument("--codec", type=str, default="examples/pong/runs/codec/codec.pt")
    parser.add_argument("--init-from", type=Path, default=Path("examples/pong/runs/wm/wm_mp.pt"))
    parser.add_argument("--out", type=Path, default=Path("examples/pong/runs/av"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    data = PongData(args.data / "train.npz", device=DEVICE, with_audio=True)
    test = PongData(args.data / "test.npz", device=DEVICE, with_audio=True)

    audio_codec_path = ensure_audio_codec(data, args.out)

    config = pong_wm_config(args.codec, multiplayer=True, clip_len=CLIP_LEN)
    model = PongAVMultiWrapper(
        MultiWrapperWorldModelConfig(n_players=2, wm_config=config), str(audio_codec_path)
    ).to(DEVICE)

    # Warm start from the video-only multiplayer model: keep every parameter whose shape still
    # matches; the input/output projections and bos (whose widths grew by the audio channels)
    # start fresh, as do the frozen audio-codec weights (absent from the checkpoint).
    checkpoint = torch.load(args.init_from, map_location=DEVICE)
    model_sd = model.state_dict()
    filtered = {
        k: v for k, v in checkpoint["state_dict"].items()
        if k in model_sd and model_sd[k].shape == v.shape
    }
    fresh = sorted(set(checkpoint["state_dict"]) - set(filtered))
    model.load_state_dict(filtered, strict=False)
    print(f"warm start from {args.init_from} ({checkpoint['stage']}@{checkpoint['step']}); "
          f"{len(fresh)} params fresh (width changed): {fresh}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"AV model: {sum(p.numel() for p in trainable) / 1e6:.1f}M trainable params")
    opt = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000))
    ema = ModelEMA(model, decay=0.999)
    ckpt_path = args.out / "wm_av.pt"
    log_path = args.out / "av_log.jsonl"

    model.train()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = data.sample_multiplayer(args.batch_size, CLIP_LEN)
        with torch.autocast("cuda", torch.bfloat16):
            losses = model(batch)
        opt.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        ema.step()

        if step % 250 == 0 or step == 1:
            msg = {
                "step": step,
                "loss": round(losses["loss_diffusion"].item(), 5),
                "sec": round(time.perf_counter() - t0),
            }
            if step % 2500 == 0:
                with ema.average_parameters():
                    msg.update(quick_eval(model, test, True, args.out, f"av_{step}"))
                    msg.update(audio_event_eval(model, test))
            print(json.dumps(msg), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(msg) + "\n")
        if step % 10000 == 0:
            save_checkpoint(model, ema, "av", step, ckpt_path)

    save_checkpoint(model, ema, "av", args.steps, ckpt_path)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
