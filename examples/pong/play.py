"""Play Pong *inside the world model*, in real time — the toy version of mira-wm.com.

Both views are generated frame by frame by the diffusion world model from the two players'
key presses; the real environment is never stepped. Mirrors the paper's serving design at toy
scale: a session primed from a short real clip, a rolling 20-latent context window, a streaming
KV-cache, few-step denoising, and a decode of the newest latents only.

Controls:
    W / S       player 1 up / down
    Up / Down   player 2 up / down
    1 / 2       toggle whether the *model* drives that player (per-player action dropout)
    R           re-prime from a fresh real clip
    Esc         quit

Usage:
    python -m examples.pong.play [--checkpoint examples/pong/runs/wm/wm_psd.pt] [--steps 2]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from mira.world_model.config import WorldModelInferenceConfig

from examples.pong.data import PONG_ACTIONS, PongData
from examples.pong.train_wm import CLIP_LEN, build_model
from mira.world_model.actions_config import ActionTensors

DEVICE = "cuda"
SCALE = 24  # 16x16 -> 384x384 per view
HISTORY = 128  # actions kept in the ring buffer (must exceed the 39-step context slice)


def make_actions(keys: torch.Tensor) -> ActionTensors:
    """Wrap per-player key tensors (2, T, 2) int32 into the encoder's container."""
    actions = ActionTensors(config=PONG_ACTIONS, batch_size=keys.shape[0])
    actions.key_presses = keys
    actions.mouse_movements = torch.zeros(keys.shape[0], keys.shape[1], 2, device=keys.device)
    actions.game_mouse_sensitivity = torch.full((keys.shape[0],), float("nan"), device=keys.device)
    return actions


class WorldModelSession:
    """A streaming rollout: rolling latent window + KV cache + action ring buffer."""

    def __init__(self, model, test: PongData, config: WorldModelInferenceConfig):
        self.model = model
        self.test = test
        self.config = config
        self.swm = model.single_world_model
        self.window = self.swm.n_context_latents + 1
        self.prime()

    @torch.no_grad()
    def prime(self) -> None:
        """Start from a short clip of real gameplay, like the paper's prefill."""
        episode = int(torch.randint(self.test.n_episodes, ()).item())
        t0 = int(torch.randint(self.test.n_frames - CLIP_LEN, ()).item())
        batch = self.test.eval_clip(episode, t0, 2 * self.window, multiplayer=True)
        z = self.model.init_streaming_inference(batch.slice_time(0, 2 * self.window, fps=20))
        self.z = z[:, -self.window :]
        self.keys = batch.actions.key_presses[:, -HISTORY:].clone()
        self.kv = None

    @torch.no_grad()
    def step(self, live_keys: np.ndarray, drop_players: torch.Tensor | None) -> np.ndarray:
        """Advance one latent step (= 2 video frames). ``live_keys`` is (2 sub-frames, 2 players, 2)."""
        new = torch.as_tensor(live_keys, dtype=torch.int32, device=self.keys.device)
        new = rearrange(new, "t p k -> p t k")
        self.keys = torch.cat([self.keys, new], dim=1)[:, -HISTORY:]

        self.z, self.kv = self.model.streaming_inference_step(
            self.z, make_actions(self.keys), self.kv, self.config, drop_players=drop_players
        )
        # Decode only the newest latents (the causal decoder needs a little context).
        z_split = rearrange(self.z[:, -3:], "b t (p h) w c -> (b p) t h w c", p=2)
        frames = self.model.decode_to_video(z_split)[:, -2:]  # (2 views, 2 frames, 3, 16, 16)
        frames = rearrange(frames.float().clamp(0, 1) * 255, "p t c h w -> t h (p w) c")
        return frames.byte().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--codec", type=str, default="examples/pong/runs/codec/codec.pt")
    parser.add_argument("--data", type=Path, default=Path("examples/pong/runs/data"))
    parser.add_argument("--steps", type=int, default=None, help="denoising steps (default 2)")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None:
        for name in ("wm_psd.pt", "wm_mp.pt"):
            candidate = Path("examples/pong/runs/wm") / name
            if candidate.exists():
                args.checkpoint = candidate
                break
        assert args.checkpoint is not None, "no world-model checkpoint found; run train_wm first"
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    is_psd = checkpoint.get("stage") == "psd"
    # Both the distilled and the plain diffusion-forcing model sample well at 2 steps (see README).
    n_steps = args.steps if args.steps is not None else 2
    print(f"checkpoint {args.checkpoint} (stage {checkpoint.get('stage')}), {n_steps} denoising steps")

    torch.set_float32_matmul_precision("high")
    model = build_model("psd" if is_psd else "mp", args.codec)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if not args.no_compile:
        print("compiling model (first steps take ~1-2 min)...")
        swm = model.single_world_model
        swm.world_model.forward = torch.compile(swm.world_model.forward)
        swm.codec.decoder.forward = torch.compile(swm.codec.decoder.forward)

    test = PongData(args.data / "test.npz", device=DEVICE)
    session = WorldModelSession(model, test, WorldModelInferenceConfig(n_diffusion_steps=n_steps))

    # Warm up (and torch.compile) BEFORE opening the window, so it never sits unresponsive.
    print("warming up the model (compiles on first steps, ~1-2 min)...", flush=True)
    t0 = time.perf_counter()
    for i in range(4):
        session.step(np.zeros((2, 2, 2), dtype=np.int32), torch.tensor([True, True], device=DEVICE))
        print(f"  warmup step {i + 1}/4 ({time.perf_counter() - t0:.0f}s)", flush=True)
    session.prime()  # fresh context after warmup
    print("ready — opening window", flush=True)

    import pygame

    pygame.init()
    screen = pygame.display.set_mode((2 * 16 * SCALE, 16 * SCALE + 48))
    pygame.display.set_caption("Pong world model — P1: W/S   P2: Up/Down   1/2: toggle bot   R: re-prime")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    model_drives = [False, True]  # start with the model driving player 2
    frame_queue: list[np.ndarray] = []
    fps_ema = 0.0
    running = True
    while running:
        # Refill: one latent step produces two frames. Poll the keyboard once per produced frame.
        if len(frame_queue) < 2:
            live = np.zeros((2, 2, 2), dtype=np.int32)  # (sub-frame, player, [Up, Down])
            for sub in range(2):
                pygame.event.pump()
                pressed = pygame.key.get_pressed()
                live[sub, 0] = [pressed[pygame.K_w], pressed[pygame.K_s]]
                live[sub, 1] = [pressed[pygame.K_UP], pressed[pygame.K_DOWN]]
            drop = torch.tensor(model_drives, device=DEVICE) if any(model_drives) else None
            t0 = time.perf_counter()
            frames = session.step(live, drop)
            dt = time.perf_counter() - t0
            fps_ema = 0.9 * fps_ema + 0.1 * (2 / dt) if fps_ema else 2 / dt
            frame_queue.extend(frames)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    model_drives[0] = not model_drives[0]
                elif event.key == pygame.K_2:
                    model_drives[1] = not model_drives[1]
                elif event.key == pygame.K_r:
                    session.prime()
                    frame_queue.clear()

        if frame_queue:
            frame = frame_queue.pop(0)  # (16, 32, 3)
            surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            surface = pygame.transform.scale(surface, (2 * 16 * SCALE, 16 * SCALE))
            screen.blit(surface, (0, 0))
            who = [("model" if m else "human") for m in model_drives]
            hud = f"gen {fps_ema:5.1f} fps | P1 (left view): {who[0]} | P2 (right view): {who[1]}"
            screen.fill((20, 20, 20), rect=(0, 16 * SCALE, 2 * 16 * SCALE, 48))
            screen.blit(font.render(hud, True, (230, 230, 230)), (12, 16 * SCALE + 14))
            pygame.display.flip()

        clock.tick(20)  # display cadence: 20 fps

    pygame.quit()


if __name__ == "__main__":
    main()
