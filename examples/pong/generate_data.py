"""Generate the toy "Rocket Science" dataset: bot-vs-bot Pong with per-player views and actions.

The output mirrors the paper's per-player recordings: for every episode, two synchronized
first-person video streams, each player's key actions, and the privileged physics state (used only
for evaluation probes). Usage:

    python examples/pong/generate_data.py [--episodes 400] [--frames 1000] [--out examples/pong/runs/data]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from examples.pong.pong_env import rollout_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--frames", type=int, default=1000, help="frames per episode (20 fps)")
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--envs-per-round", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("examples/pong/runs/data"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    chunks = []
    done = 0
    t0 = time.perf_counter()
    while done < args.episodes:
        n = min(args.envs_per_round, args.episodes - done)
        chunks.append(rollout_episodes(n_envs=n, n_frames=args.frames, seed=args.seed + done))
        done += n
        print(f"{done}/{args.episodes} episodes ({time.perf_counter() - t0:.1f}s)")

    data = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    n_train = args.episodes - args.test_episodes
    for split, sl in (("train", slice(0, n_train)), ("test", slice(n_train, None))):
        path = args.out / f"{split}.npz"
        np.savez_compressed(path, **{k: v[sl] for k, v in data.items()})
        n_ep = data["frames"][sl].shape[0]
        print(f"{path}: {n_ep} episodes, {n_ep * args.frames:,} frames/view, "
              f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
