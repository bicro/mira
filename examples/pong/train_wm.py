"""Train the Pong world model with MIRA's three-stage recipe.

Stage ``sp``  — single-player pretraining: one view per sample, that player's own 2-key action
                stream (the paper pretrains its multiplayer model from exactly such a model).
Stage ``mp``  — multiplayer training: both views tiled along the height by mira's
                ``MultiWrapperWorldModel`` (via :class:`PongMultiWrapper`), per-player action
                streams combined through the shared action encoder + player embeddings,
                **warm-started from the sp checkpoint** using the wrapper's built-in remap.
Stage ``psd`` — few-step self-distillation finetune: continues the mp model with the PSD-M loss
                enabled so 1-2 denoising steps suffice at inference (paper Section 4.3).

All stages share the released MIRA optimizer settings (AdamW 1e-4, betas (0.9, 0.99), wd 0.1,
1k-step warmup then constant, grad clip 1.0, weight EMA) and train with flow matching + diffusion
forcing on 64-frame (3.2 s) clips.

Usage:
    python -m examples.pong.train_wm --stage sp   [--steps 20000]
    python -m examples.pong.train_wm --stage mp   [--steps 60000]
    python -m examples.pong.train_wm --stage psd  [--steps 10000]
    python -m examples.pong.train_wm --stage all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mira.training.ema import ModelEMA
from mira.world_model.config import LatentWorldModelConfig, WorldModelInferenceConfig
from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModelConfig

from examples.pong.data import PongData
from examples.pong.world_model import PongMultiWrapper, PongWorldModel, pong_wm_config

DEVICE = "cuda"
CLIP_LEN = 64


def build_model(stage: str, codec_checkpoint: str, psd: bool = False):
    config = pong_wm_config(
        codec_checkpoint,
        multiplayer=(stage != "sp"),
        clip_len=CLIP_LEN,
        psd_loss_prob=0.5 if psd else 0.0,
    )
    if psd:
        # The PSD student pass doubles live activations (a second grad-carrying full-sequence
        # forward on top of the diagonal pass), which overflows 16 GB at full batch and thrashes
        # into shared memory. Checkpoint only that pass, exactly the knob mira exposes for this.
        config.activation_checkpointing = "psd-only"
    if stage == "sp":
        return PongWorldModel(config).to(DEVICE)
    return PongMultiWrapper(MultiWrapperWorldModelConfig(n_players=2, wm_config=config)).to(DEVICE)


def save_checkpoint(model, ema: ModelEMA, stage: str, step: int, path: Path) -> None:
    with ema.average_parameters():  # checkpoint the EMA weights, like the release models
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "stage": stage, "step": step}, path)


@torch.no_grad()
def quick_eval(model, test: PongData, multiplayer: bool, out_dir: Path, tag: str) -> dict:
    """Teacher-action rollout on a held-out clip; pixel PSNR of the generated (non-context) frames.

    Pong is near-deterministic given both action streams, so unlike the paper's stochastic setting
    a reference-based pixel metric against the real continuation is meaningful.
    """
    model.eval()
    batch = test.eval_clip(episode=0, t0=200, clip_len=CLIP_LEN, multiplayer=multiplayer)
    gt = batch.video.clone()
    outputs = model.inference(batch, WorldModelInferenceConfig(n_diffusion_steps=10), progress_bar=False)
    pred = outputs.output_video  # (b, t, c, [p*]h, w) in [0, 1]
    swm = model.single_world_model if multiplayer else model
    n_ctx = swm.n_context_frames
    if multiplayer:  # tile the ground-truth views to match the wrapper's output layout
        gt = gt.reshape(1, 2, *gt.shape[1:]).permute(0, 2, 3, 1, 4, 5).flatten(3, 4)
    gt01 = gt[:, : pred.shape[1]].float() / 255.0
    mse = F.mse_loss(pred[:, n_ctx:].float().clamp(0, 1), gt01[:, n_ctx:]).item()
    psnr = -10 * float(np.log10(mse + 1e-12))

    strip = torch.cat([gt01[0, ::4], pred[0, ::4].float().clamp(0, 1)], dim=-2)
    strip = (strip.permute(2, 0, 3, 1).flatten(1, 2) * 255).byte().cpu().numpy()
    import PIL.Image

    PIL.Image.fromarray(np.kron(strip, np.ones((4, 4, 1))).astype(np.uint8)).save(
        out_dir / f"rollout_{tag}.png"
    )
    model.train()
    return {"rollout_psnr": round(psnr, 2)}


def train_stage(
    stage: str,
    steps: int,
    data: PongData,
    test: PongData,
    codec_checkpoint: str,
    out_dir: Path,
    batch_size: int,
    init_from: Path | None = None,
) -> Path:
    multiplayer = stage != "sp"
    model = build_model(stage, codec_checkpoint, psd=(stage == "psd"))
    if init_from is not None:
        checkpoint = torch.load(init_from, map_location=DEVICE)
        # strict=False: the psd stage adds the step-size embedding; the mp stage's wrapper remaps
        # and exempts single-player-only params via MultiWrapperWorldModel.load_state_dict.
        result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        missing = [k for k in result.missing_keys]
        print(f"warm start from {init_from} ({checkpoint['stage']}@{checkpoint['step']}), "
              f"{len(missing)} params fresh: {missing[:6]}{'...' if len(missing) > 6 else ''}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"stage {stage}: {n_params / 1e6:.1f}M trainable params, batch {batch_size}, {steps} steps")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000))
    ema = ModelEMA(model, decay=0.999)
    log_path = out_dir / f"{stage}_log.jsonl"
    ckpt_path = out_dir / f"wm_{stage}.pt"

    model.train()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        batch = (
            data.sample_multiplayer(batch_size, CLIP_LEN)
            if multiplayer
            else data.sample_views(batch_size, CLIP_LEN)
        )
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
                "stage": stage,
                "step": step,
                "loss": round(losses["loss_diffusion"].item(), 5),
                "sec": round(time.perf_counter() - t0),
            }
            if "loss_psd" in losses:
                msg["loss_psd"] = round(losses["loss_psd"].item(), 5)
            if step % 2500 == 0:
                with ema.average_parameters():
                    msg.update(quick_eval(model, test, multiplayer, out_dir, f"{stage}_{step}"))
            print(json.dumps(msg), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(msg) + "\n")
        if step % 10000 == 0:
            save_checkpoint(model, ema, stage, step, ckpt_path)

    save_checkpoint(model, ema, stage, steps, ckpt_path)
    print(f"saved {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sp", "mp", "psd", "all"], default="all")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sp-steps", type=int, default=20000)
    parser.add_argument("--mp-steps", type=int, default=60000)
    parser.add_argument("--psd-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data", type=Path, default=Path("examples/pong/runs/data"))
    parser.add_argument("--codec", type=str, default="examples/pong/runs/codec/codec.pt")
    parser.add_argument("--out", type=Path, default=Path("examples/pong/runs/wm"))
    parser.add_argument("--init-from", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    data = PongData(args.data / "train.npz", device=DEVICE)
    test = PongData(args.data / "test.npz", device=DEVICE)

    stage_defaults = {  # (steps, batch in samples; a multiplayer sample is 2 player-views)
        "sp": (args.sp_steps, 64),
        "mp": (args.mp_steps, 32),
        "psd": (args.psd_steps, 32),
    }
    stages = ["sp", "mp", "psd"] if args.stage == "all" else [args.stage]
    default_init = {"sp": None, "mp": args.out / "wm_sp.pt", "psd": args.out / "wm_mp.pt"}

    init = args.init_from
    for stage in stages:
        steps, batch = stage_defaults[stage]
        if args.steps is not None and args.stage != "all":
            steps = args.steps
        if args.batch_size is not None:
            batch = args.batch_size
        init_from = init if init is not None else default_init[stage]
        ckpt = train_stage(stage, steps, data, test, args.codec, args.out, batch, init_from)
        init = ckpt


if __name__ == "__main__":
    main()
