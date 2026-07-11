"""Play Pong inside the audio-video world model — with generated sound effects.

Same real-time loop as ``play.py``, but the model jointly generates video *and* audio: every
latent step yields two frames plus 100 ms of waveform, decoded from the same denoised latent, so
the beeps you hear (paddle ticks, wall/paddle bounces, the score jingle) are the model's own
predictions of what this moment should sound like — not triggered by any game logic.

Controls are identical to play.py: W/S and Up/Down, 1/2 toggle model-driven players, R re-primes.

Usage:
    python -m examples.pong.play_av [--checkpoint examples/pong/runs/av/wm_av.pt]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from mira.world_model.config import WorldModelInferenceConfig
from mira.world_model.multi_wrapper_world_model import MultiWrapperWorldModelConfig

from examples.pong.data import PongData
from examples.pong.play import SCALE, WorldModelSession
from examples.pong.pong_audio import SAMPLE_RATE
from examples.pong.train_wm import CLIP_LEN
from examples.pong.world_model import pong_wm_config
from examples.pong.world_model_av import PongAVMultiWrapper

DEVICE = "cuda"


class AVSession(WorldModelSession):
    """Streaming rollout that also decodes the newest latent's audio chunk."""

    @torch.no_grad()
    def step(self, live_keys: np.ndarray, drop_players: torch.Tensor | None):
        new = torch.as_tensor(live_keys, dtype=torch.int32, device=self.keys.device)
        new = rearrange(new, "t p k -> p t k")
        self.keys = torch.cat([self.keys, new], dim=1)[:, -128:]

        from examples.pong.play import make_actions

        self.z, self.kv = self.model.streaming_inference_step(
            self.z, make_actions(self.keys), self.kv, self.config, drop_players=drop_players
        )
        z_new = self.z[:, -3:]
        z_split = rearrange(z_new, "b t (p h) w c -> (b p) t h w c", p=2)
        frames = self.model.decode_to_video(z_split)[:, -2:]
        frames = rearrange(frames.float().clamp(0, 1) * 255, "p t c h w -> t h (p w) c")
        wave = self.model.decode_audio(self.z[:, -1:])[0].flatten()  # (2 * SAMPLES_PER_FRAME,)
        audio = (wave.cpu().numpy() * 0.8 * 32767).astype(np.int16)
        return frames.byte().cpu().numpy(), audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("examples/pong/runs/av/wm_av.pt"))
    parser.add_argument("--audio-codec", type=str, default="examples/pong/runs/av/audio_codec.pt")
    parser.add_argument("--codec", type=str, default="examples/pong/runs/codec/codec.pt")
    parser.add_argument("--data", type=Path, default=Path("examples/pong/runs/data"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    print(f"checkpoint {args.checkpoint} (stage {checkpoint.get('stage')}), {args.steps} denoising steps")

    torch.set_float32_matmul_precision("high")
    config = pong_wm_config(args.codec, multiplayer=True, clip_len=CLIP_LEN)
    model = PongAVMultiWrapper(
        MultiWrapperWorldModelConfig(n_players=2, wm_config=config), args.audio_codec
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if not args.no_compile:
        swm = model.single_world_model
        swm.world_model.forward = torch.compile(swm.world_model.forward)
        swm.codec.decoder.forward = torch.compile(swm.codec.decoder.forward)

    test = PongData(args.data / "test.npz", device=DEVICE, with_audio=True)
    session = AVSession(model, test, WorldModelInferenceConfig(n_diffusion_steps=args.steps))

    print("warming up the model (compiles on first steps)...", flush=True)
    t0 = time.perf_counter()
    for i in range(4):
        session.step(np.zeros((2, 2, 2), dtype=np.int32), torch.tensor([True, True], device=DEVICE))
        print(f"  warmup step {i + 1}/4 ({time.perf_counter() - t0:.0f}s)", flush=True)
    session.prime()
    print("ready — opening window", flush=True)

    import pygame

    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 256)
    pygame.init()
    channel = pygame.mixer.Channel(0)
    screen = pygame.display.set_mode((2 * 16 * SCALE, 16 * SCALE + 48))
    pygame.display.set_caption("Pong AV world model — P1: W/S  P2: Up/Down  1/2: toggle bot  R: re-prime")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    model_drives = [False, True]
    frame_queue: list[np.ndarray] = []
    fps_ema = 0.0
    running = True
    while running:
        if len(frame_queue) < 2:
            live = np.zeros((2, 2, 2), dtype=np.int32)
            for sub in range(2):
                pygame.event.pump()
                pressed = pygame.key.get_pressed()
                live[sub, 0] = [pressed[pygame.K_w], pressed[pygame.K_s]]
                live[sub, 1] = [pressed[pygame.K_UP], pressed[pygame.K_DOWN]]
            drop = torch.tensor(model_drives, device=DEVICE) if any(model_drives) else None
            t0 = time.perf_counter()
            frames, audio = session.step(live, drop)
            dt = time.perf_counter() - t0
            fps_ema = 0.9 * fps_ema + 0.1 * (2 / dt) if fps_ema else 2 / dt
            frame_queue.extend(frames)
            # Queue the generated 100 ms of audio; the channel plays chunks back to back.
            sound = pygame.mixer.Sound(buffer=audio.tobytes())
            if channel.get_busy():
                channel.queue(sound)
            else:
                channel.play(sound)

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
            frame = frame_queue.pop(0)
            surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            surface = pygame.transform.scale(surface, (2 * 16 * SCALE, 16 * SCALE))
            screen.blit(surface, (0, 0))
            who = [("model" if m else "human") for m in model_drives]
            hud = f"gen {fps_ema:5.1f} fps + audio | P1: {who[0]} | P2: {who[1]}"
            screen.fill((20, 20, 20), rect=(0, 16 * SCALE, 2 * 16 * SCALE, 48))
            screen.blit(font.render(hud, True, (230, 230, 230)), (12, 16 * SCALE + 14))
            pygame.display.flip()

        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()
