# Rocket League 4-player dataset

Time-aligned **video, keyboard actions, game events, and per-frame game state** for all four players
of a 2v2 Rocket League match.

## Loading

```python
from mira.data import RocketScienceDataset

ds = RocketScienceDataset.from_hub("kyutai/rocket-science", split="test")
clip = ds.load_match(ds.match_ids()[0], clip_len=16, target_fps=10)[0]
```

`pip install mira[decode]` (provide your own torch + a compatible FFmpeg), or use the
repo's pixi environment.

## What's in a sample

One sample is a short (~4 s, 80 frames at 20 fps) window of a match with all four players'
synchronised views. A **clip** is a fixed-length, fps-downsampled slice read from one. For each
clip, per perspective:

- **`frames`** — `(P, T, C, H, W)` uint8 video (1280×720), one decoded view per perspective.
- **`actions`** — `(P, T, 9)` int32 multi-hot keyboard over a fixed key set (`W A S D Q E Space
  LShiftKey LControlKey`), OR-ed over each downsample window so it stays frame-aligned.
- **`events`** — discrete game events (`KickoffStarted/Ended`, `GoalScored`, `Demolition`,
  `GoalReplayStarted/Ended`, …) on a single match-wide master clock, mapped onto each perspective's
  frames via its `recording_offset_sec`. `exclude_replays=True` skips clips overlapping goal replays.
- **`physics`** — per-frame game state, one list of `FrameState` dicts per perspective, frame-aligned
  with the video: the **ball** (location, velocity, rotation, angular velocity), the four **cars**
  (location, velocity, boost, on-ground / supersonic flags, …), and the **game** info (score,
  overtime, clock). See `mira.data.state` for the typed field layout.

Perspectives are ordered by `player_id`; `clip.teams` gives each one's team (0 / 1).

## Exploring

`mira.data.viz` renders a clip as a synchronised 4-view grid with a keyboard overlay plus a
top-down arena radar and HUD; `examples/explore.py` is a guided, interactive tour (`pixi run
explore`).
