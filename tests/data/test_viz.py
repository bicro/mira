"""Tests for viz rendering. Pure numpy/matplotlib/pillow tests run anywhere; the mp4 tests need
ffmpeg (run via `pixi run test`)."""

import shutil

import numpy as np
import pytest
import torch

from mira.data import viz


def _with_frames(clip, h=90, w=160):
    """Give a fixture clip realistically-sized frames (the conftest default is tiny for video)."""
    p, t = len(clip.player_ids), clip.actions.shape[1]
    clip.frames = torch.randint(0, 256, (p, t, 3, h, w), dtype=torch.uint8)
    return clip


_needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg (run via pixi)")


# --- on-frame overlay + grid composition (no ffmpeg) --------------------------------------------


def test_decorate_frame_keeps_shape():
    img = np.random.randint(0, 256, (120, 200, 3), dtype=np.uint8)
    out = viz.decorate_frame(img, [1, 0, 1, 0, 0, 0, 1, 0, 0], label="P480", team=1)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_compose_grid_frame_dims():
    tiles = [np.zeros((30, 40, 3), np.uint8) for _ in range(4)]
    grid = viz._compose_grid_frame(tiles, [0, 1, 0, 1], cols=2, gutter=8, border=3)
    # rows=cols=2; bordered tile 36x46; H=2*36+3*8=96, W=2*46+3*8=116
    assert grid.shape == (96, 116, 3)


def test_compose_grid_frame_paints_team_borders():
    tiles = [np.zeros((20, 20, 3), np.uint8), np.zeros((20, 20, 3), np.uint8)]
    grid = viz._compose_grid_frame(tiles, [0, 1], cols=2, gutter=4, border=2)
    assert tuple(grid[4 + 0, 4 + 0]) == viz._TEAM_COLORS[0]  # first tile border = team 0
    assert tuple(grid[4 + 0, 4 + 2 * 2 + 20 + 4]) == viz._TEAM_COLORS[1]  # second tile border = team 1


def test_keystroke_timeline_axes(make_clip):
    fig = viz.keystroke_timeline(make_clip(p=4), perspective=0)
    assert len(fig.axes) == 1


# --- minimap + HUD (numpy/pillow, no ffmpeg) ----------------------------------------------------


def test_minimap_frame_shape_and_dtype(make_frame):
    out = viz.minimap_frame(make_frame(10), size=200)
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
    assert out.shape[0] > out.shape[1]  # taller than wide (arena is longer along y)


def test_radar_panel_frame_shape(make_frame):
    panel = viz.radar_panel_frame(make_frame(10), panel_h=400, radar_size=240)
    assert panel.shape[0] == 400  # exact target height
    assert panel.ndim == 3 and panel.shape[2] == 3 and panel.dtype == np.uint8


def test_radar_panel_frame_shape_with_badge(make_frame):
    from mira.data.physics import Badge

    badge = Badge("REPLAY", "REPLAY · ⏸ physics frozen", True)  # a frozen badge must not change dims
    panel = viz.radar_panel_frame(make_frame(10), panel_h=400, radar_size=240, badge=badge)
    assert panel.shape[0] == 400 and panel.dtype == np.uint8


def test_radar_panel_radar_fits_height(make_frame):
    # A short panel forces the radar to scale down; it must still produce a valid panel of panel_h.
    panel = viz.radar_panel_frame(make_frame(10), panel_h=120, radar_size=400)
    assert panel.shape[0] == 120 and panel.dtype == np.uint8


def test_game_state_md_shows_local_tag_and_dash_without_clock(make_clip):
    md = viz.game_state_md(make_clip(tr=0.0), frame=0)  # tr=0 -> this perspective has no clock
    assert "you" in md  # local car tagged
    assert "—" in md  # clock shown as a dash when this perspective doesn't carry it


def test_demolition_attacker_in_readout(make_clip):
    clip = make_clip(t=2)
    assert clip.physics is not None
    for ti in range(2):  # car 2 demolished by player 10
        clip.physics[0][ti]["cars"][2]["attacker_player_id"] = 10
    md = viz.game_state_md(clip, frame=0)
    assert "demo by P10" in md


# --- mp4 encoding (needs ffmpeg) ----------------------------------------------------------------


def test_clip_grid_video_requires_frames(make_clip):
    clip = make_clip()
    clip.frames = None
    with pytest.raises(ValueError):
        viz.clip_grid_video(clip)


@_needs_ffmpeg
def test_clip_grid_video_all_perspectives(make_clip):
    mp4 = viz.clip_grid_video(_with_frames(make_clip(p=4, t=6)))
    assert isinstance(mp4, bytes) and b"ftyp" in mp4[:64]


@_needs_ffmpeg
def test_clip_grid_video_single_perspective(make_clip):
    mp4 = viz.clip_grid_video(_with_frames(make_clip(p=1, t=6)), keyboard=False)
    assert isinstance(mp4, bytes) and b"ftyp" in mp4[:64]


@_needs_ffmpeg
def test_clip_grid_with_radar_video_is_one_mp4(make_clip):
    mp4 = viz.clip_grid_with_radar_video(_with_frames(make_clip(p=4, t=4)), radar_size=220)
    assert isinstance(mp4, bytes) and b"ftyp" in mp4[:64]
