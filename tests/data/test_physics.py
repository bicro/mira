"""Tests for the numpy-only physics helpers: consistency checks, frozen detection, anchor windows,
and per-step badges."""

import numpy as np

from mira.data import physics


def test_consistency_all_pass_on_good_clip(make_clip):
    checks = physics.consistency_checks(make_clip())
    assert all(c.ok for c in checks)
    labels = {c.label for c in checks}
    assert "is_local ↔ perspective player_id" in labels
    assert "ball moves over clip" in labels


def test_consistency_flags_frozen_ball(make_clip):
    checks = physics.consistency_checks(make_clip(moving=False))
    moved = next(c for c in checks if c.label == "ball moves over clip")
    assert not moved.ok


def test_consistency_flags_length_mismatch(make_clip):
    clip = make_clip(t=6)
    assert clip.physics is not None
    clip.physics[1] = clip.physics[1][:-1]  # drop a frame on one perspective
    checks = physics.consistency_checks(clip)
    lens = next(c for c in checks if c.label.startswith("frames == actions"))
    assert not lens.ok


def test_consistency_never_raises_without_physics(make_clip):
    clip = make_clip()
    clip.physics = None
    checks = physics.consistency_checks(clip)
    assert any(not c.ok and c.label == "physics present" for c in checks)


def test_perspective_has_clock(make_clip):
    advancing, flat = make_clip(tr=120.0).physics, make_clip(tr=0.0).physics
    assert advancing is not None and flat is not None
    assert physics.perspective_has_clock(advancing[0])  # clock advances on this perspective
    assert not physics.perspective_has_clock(flat[0])  # constant 0.0 -> clock is on another


def test_min_lag_residual_finds_shift():
    a = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    assert physics._min_lag_residual(a, a + 0.0, 1) == 0.0  # identical -> 0 at lag 0
    shifted = np.array([[1.0, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])  # a advanced by 1 step
    assert physics._min_lag_residual(a, shifted, 1) < 1e-9  # collapses to ~0 at some lag in [-1,1]


def test_local_car_is_unique_and_matches(make_frame):
    assert physics.local_car(make_frame(20))["player_id"] == 20


def test_anchor_window_lifecycle(anchor_events):
    ev = anchor_events
    assert physics.anchor_window_at_master_sec(ev, 2.0) == physics._WIN_KICKOFF
    assert physics.anchor_window_at_master_sec(ev, 20.0) == physics._WIN_PLAY
    assert physics.anchor_window_at_master_sec(ev, 42.0) == physics._WIN_GOAL_PAUSE
    assert physics.anchor_window_at_master_sec(ev, 48.0) == physics._WIN_REPLAY
    assert physics.anchor_window_at_master_sec(ev, 60.0) == physics._WIN_PLAY


def test_step_badges_no_live_plus_frozen_overlap_across_goal(make_clip_with_anchors):
    # A goal-straddling clip: cars/ball MOVE up to the goal, then FREEZE during the pause. src_fps =
    # target_fps*stride = 20. Steps at g = 700,740,780,820,860 -> 35,37,39,41,43 s. Goal @40s, so the
    # last two steps fall in the goal-pause window. Ball y moves until the goal then is constant.
    ball_ys = [0.0, 200.0, 400.0, 600.0, 600.0]
    clip = make_clip_with_anchors(ball_ys, [700, 740, 780, 820, 860], 5)
    badges = physics.step_badges(clip, 0)
    # Motion is authoritative: moving steps are LIVE (never frozen); frozen steps are PAUSE.
    assert [b.code for b in badges] == ["LIVE", "LIVE", "LIVE", "PAUSE", "PAUSE"]
    for b in badges:
        assert not (b.code == "LIVE" and b.frozen)  # never both
    assert badges[0].frozen is False and badges[-1].frozen is True


def test_step_badges_kickoff_moving_has_no_frozen_note(make_clip_with_anchors):
    # In the kickoff window (0-4 s) with cars driving to the ball -> KICKOFF, never a frozen note.
    clip = make_clip_with_anchors([0.0, 100.0, 200.0, 300.0], [0, 20, 40, 60], 4)
    badges = physics.step_badges(clip, 0)
    assert all(b.code == "KICKOFF" and not b.frozen for b in badges)


def test_step_badges_frozen_without_window_is_neutral_not_live(make_clip):
    # Frozen but no anchors -> neutral FROZEN badge (never green LIVE), with a frozen note.
    clip = make_clip(t=4, moving=False)
    badges = physics.step_badges(clip, 0)
    assert all(b.code == "FROZEN" and b.frozen for b in badges)


def test_step_badges_falls_back_to_live_when_moving_without_anchors(make_clip):
    clip = make_clip(t=4, moving=True)
    badges = physics.step_badges(clip, 0)
    assert all(b.code == "LIVE" and not b.frozen for b in badges)


def test_frozen_steps_detects_freeze(make_clip):
    frozen_phys = make_clip(t=5, moving=False).physics  # ball/cars constant -> frozen
    moving_phys = make_clip(t=5, moving=True).physics  # ball moves 200uu/step -> not frozen
    assert frozen_phys is not None and moving_phys is not None
    assert all(physics.frozen_steps(frozen_phys[0]))
    assert not any(physics.frozen_steps(moving_phys[0]))
