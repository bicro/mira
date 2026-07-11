"""Train the toy Pong codec, mirroring the paper's two-step representation-autoencoder recipe.

Stage ``ssl``  — pretrain the tiny feature extractor self-supervised on Pong frames (SimMIM-style
                 masked patch reconstruction). This plays the role of DINOv3's pretraining; the
                 extractor is frozen afterwards.
Stage ``codec`` — train the linear(ish) bottleneck + causal ViT decoder to reconstruct video from
                 the frozen features, with an L1 anchor plus a feature-consistency loss balanced by
                 the same VQ-GAN-style adaptive gradient-norm rule the paper uses (their Eq. for
                 lambda; ``mira.codec.loss.calculate_adaptive_weight``). No GAN, no KL, matching
                 the release codec. Ends by estimating the scalar latent mean/std that the world
                 model uses to normalize tokens (saved in the checkpoint like theirs).

Usage:
    python -m examples.pong.train_codec [--ssl-steps 3000] [--codec-steps 20000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mira.codec.loss import calculate_adaptive_weight

from examples.pong.codec import PongCodec, PongFeatureExtractor, default_codec_config
from examples.pong.data import PongData

DEVICE = "cuda"


def train_extractor(data: PongData, steps: int, out_dir: Path) -> PongFeatureExtractor:
    """SimMIM-style self-supervised pretraining of the toy feature extractor."""
    config = default_codec_config().encoder.extractor
    model = PongFeatureExtractor(config).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    pool = data.frames.flatten(0, 2)  # (E*P*T, 3, 16, 16) uint8, on GPU
    n_patches = (16 // config.patch_size) ** 2
    batch_size, mask_ratio = 1024, 0.6

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        frames01 = pool[torch.randint(pool.shape[0], (batch_size,), device=DEVICE)] / 255.0
        patch_mask = torch.rand(batch_size, n_patches, device=DEVICE) < mask_ratio
        with torch.autocast("cuda", torch.bfloat16):
            tokens = model.forward_tokens(frames01, patch_mask=patch_mask)[-1]
            recon = model.reconstruct_pixels(tokens, (16, 16))
            target = frames01 - 0.5
            pixel_mask = patch_mask.reshape(batch_size, 1, 8, 8).repeat_interleave(2, -1).repeat_interleave(2, -2)
            loss = (F.l1_loss(recon, target, reduction="none") * pixel_mask).sum() / (
                pixel_mask.sum() * 3 + 1e-8
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            print(f"[ssl {step}/{steps}] loss={loss.item():.4f} ({time.perf_counter() - t0:.0f}s)")

    torch.save(model.state_dict(), out_dir / "extractor.pt")
    return model


def train_codec(data: PongData, extractor_state: dict, steps: int, out_dir: Path, clip_len: int = 32):
    codec = PongCodec(default_codec_config(clip_len)).to(DEVICE)
    codec.encoder.extractor.load_state_dict(extractor_state)
    codec.encoder.freeze_extractor()

    trainable = [p for p in codec.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_lambda(step: int) -> float:  # 1k warmup then cosine decay to ~1e-6, like Table 9
        if step < 1000:
            return step / 1000
        p = (step - 1000) / max(1, steps - 1000)
        return max(1e-6 / 2e-4, 0.5 * (1 + np.cos(np.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    last_layer = codec.decoder.patch_unembed.proj.weight
    batch_size, frame_frac = 16, 0.25
    log_path = out_dir / "codec_log.jsonl"

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        batch = data.sample_views(batch_size, clip_len)
        with torch.autocast("cuda", torch.bfloat16):
            out = codec(batch, trim_video=False)
            loss_mae = F.l1_loss(out.output_video.float(), out.input_video.float())

            # Feature-consistency on a random 25% of frames: re-encode the reconstruction with the
            # frozen extractor and match the aggregated features of the original.
            t_total = out.output_video.shape[1]
            t_idx = torch.randperm(t_total, device=DEVICE)[: max(1, round(t_total * frame_frac))]
            recon01 = (out.output_video[:, t_idx].float() + 1) / 2
            with torch.no_grad():
                target_feat = codec.encoder.aggregate((out.input_video[:, t_idx].float() + 1) / 2)
            loss_feat = F.mse_loss(codec.encoder.aggregate(recon01), target_feat)

        weight = calculate_adaptive_weight(loss_mae, loss_feat, last_layer)
        loss = loss_mae + weight * loss_feat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % 500 == 0 or step == 1:
            msg = {
                "step": step,
                "loss_mae": round(loss_mae.item(), 5),
                "loss_feat": round(loss_feat.item(), 5),
                "feat_auto_w": round(weight.item(), 4),
                "sec": round(time.perf_counter() - t0),
            }
            print(f"[codec {step}/{steps}] {msg}")
            with open(log_path, "a") as f:
                f.write(json.dumps(msg) + "\n")

    # Latent statistics for the world model's token normalization (scalar mean/std, like Table 11).
    codec.eval()
    values = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for _ in range(64):
            batch = data.sample_views(batch_size, clip_len)
            codec.preprocess_batch(batch)
            _, enc = codec.encode(batch.video, trim_video=False)
            values.append(enc.z.float().flatten())
    z = torch.cat(values)
    latent_mean_std = [z.mean().item(), z.std().item()]
    print(f"latent mean/std: {latent_mean_std[0]:.4f} / {latent_mean_std[1]:.4f}")

    codec.save_checkpoint(out_dir / "codec.pt", extra_info={"latent_mean_std": latent_mean_std})
    return codec


@torch.no_grad()
def evaluate(codec: PongCodec, test: PongData, out_dir: Path, clip_len: int = 32) -> None:
    """Reconstruction metrics on held-out episodes plus a side-by-side filmstrip."""
    codec.eval()
    batch = test.eval_clip(episode=0, t0=100, clip_len=clip_len, multiplayer=True)
    out = codec(batch, trim_video=False)
    mae = F.l1_loss(out.output_video.float(), out.input_video.float()).item()
    mse = F.mse_loss(
        out.output_video.float() * 0.5 + 0.5, out.input_video.float() * 0.5 + 0.5
    ).item()
    psnr = -10 * np.log10(mse + 1e-12)
    print(f"test reconstruction: L1={mae:.4f} PSNR={psnr:.1f} dB")

    gt = (out.input_video[0, ::4].float() * 0.5 + 0.5).clamp(0, 1)
    pred = (out.output_video[0, ::4].float() * 0.5 + 0.5).clamp(0, 1)
    strip = torch.cat([gt, pred], dim=-2)  # stack GT over reconstruction
    strip = (strip.permute(2, 0, 3, 1).flatten(1, 2) * 255).byte().cpu().numpy()
    strip = np.kron(strip, np.ones((6, 6, 1))).astype(np.uint8)
    import PIL.Image

    PIL.Image.fromarray(strip).save(out_dir / "codec_recon.png")
    print(f"saved {out_dir / 'codec_recon.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-steps", type=int, default=3000)
    parser.add_argument("--codec-steps", type=int, default=20000)
    parser.add_argument("--data", type=Path, default=Path("examples/pong/runs/data"))
    parser.add_argument("--out", type=Path, default=Path("examples/pong/runs/codec"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    data = PongData(args.data / "train.npz", device=DEVICE)
    print(f"train data: {data.n_episodes} episodes x {data.n_players} views x {data.n_frames} frames")

    extractor_path = args.out / "extractor.pt"
    if extractor_path.exists():
        print("reusing pretrained extractor")
        extractor_state = torch.load(extractor_path, map_location=DEVICE)
    else:
        extractor_state = train_extractor(data, args.ssl_steps, args.out).state_dict()

    codec = train_codec(data, extractor_state, args.codec_steps, args.out)
    evaluate(codec, PongData(args.data / "test.npz", device=DEVICE), args.out)


if __name__ == "__main__":
    main()
