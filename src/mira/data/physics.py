"""Pure helpers for the per-frame physics / game state attached to a clip.

``clip.physics`` is ``list[list[FrameState]]`` — one list per selected perspective, each a per-frame
state frame-aligned 1:1 with the clip's frames / actions along the T axis (see
:mod:`mira.data.state` for the field layout).

These functions are numpy-only so they are easy to unit-test, separate from any rendering layer that
consumes them.

Coordinate system (standard Rocket League "Standard" arena): x in ~[-4096, 4096] (side walls), y in
~[-5120, 5120] (the goal-to-goal long axis), z in ~[0, 2044] (floor to ceiling). Distances are in
"uu" (Unreal units).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .events import parse_anchors
from .state import CarState, FrameState, Vec3

if TYPE_CHECKING:
    from .dataset import MatchClip

# Rocket League "Standard" arena half-extents in Unreal units (uu). Used to frame the minimap; the
# observed data slightly exceeds these at the corners/goal mouths, so the minimap pads a little.
FIELD_X = 4096.0  # half-width  (side walls)
FIELD_Y = 5120.0  # half-length (back walls / goal line)
FIELD_Z = 2044.0  # ceiling

N_CARS = 4  # the dataset is 2v2, so every physics frame carries exactly 4 cars


def _xyz(d: Vec3) -> tuple[float, float, float]:
    """Extract an (x, y, z) triple from a ``{"x":..,"y":..,"z":..}`` location/velocity dict."""
    return float(d["x"]), float(d["y"]), float(d["z"])


def perspective_has_clock(persp_physics: list[FrameState]) -> bool:
    """The match clock (``game.time_remaining``) is logged only on the local (``is_local``)
    perspective; on the other perspectives it is identically 0.0. Returns whether THIS perspective
    carries the clock (its ``time_remaining`` advances over the clip)."""
    vals = {round(float(fr["game"]["time_remaining"]), 3) for fr in persp_physics}
    return vals != {0.0}


def ball_track(persp_physics: list[FrameState]) -> np.ndarray:
    """(T, 3) array of ball world positions over one perspective's clip."""
    return np.array([_xyz(fr["ball"]["location"]) for fr in persp_physics], dtype=np.float64)


def _min_lag_residual(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Smallest mean per-frame distance between two (T,3) tracks over integer lags in [-max_lag,
    max_lag]. Compares two (T,3) tracks that share a world but sit on offset recording clocks.

    Returns ``inf`` when no lag yields an overlap to compare (e.g. an empty track), so a degenerate
    clip reads as "could not establish agreement" rather than a misleading perfect 0.0."""
    best = float("inf")
    for k in range(-max_lag, max_lag + 1):
        x, y = (a[k:], b[: len(b) - k]) if k >= 0 else (a[:k], b[-k:])
        if len(x) == 0 or len(x) != len(y):
            continue
        best = min(best, float(np.linalg.norm(x - y, axis=1).mean()))
    return best


def ball_moves(persp_physics: list[FrameState], min_uu: float = 1.0) -> bool:
    """True if the ball position is not frozen across the clip (total path length > ``min_uu``)."""
    track = ball_track(persp_physics)
    if len(track) < 2:
        return False
    return float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum()) > min_uu


def local_car(frame: FrameState) -> CarState:
    """The ``is_local`` car in one physics frame (raises if not exactly one)."""
    locals_ = [c for c in frame["cars"] if c.get("is_local")]
    if len(locals_) != 1:
        raise ValueError(f"expected exactly one is_local car, found {len(locals_)}")
    return locals_[0]


# --- per-step badge (motion-authoritative, anchor-explained) ------------------------------------
#
# During post-goal celebrations and replays the live physics freezes (ball/cars/clock constant)
# while the broadcast video keeps rolling. Actual motion is authoritative for the frozen state; the
# event anchors supply the reason. A frame is never simultaneously "moving/LIVE" and "frozen".
#
# Anchor windows on the master clock (per goal):
#   KickoffStarted -> KickoffEnded     : KICKOFF window  (cars actively drive — NOT frozen)
#   GoalScored     -> next KickoffStarted : GOAL-PAUSE window (celebration; physics frozen)
#   GoalReplayStarted -> GoalReplayEnded  : REPLAY window     (physics frozen)
# Anchors are matched even when the governing event lies OUTSIDE the clip window.

# Badge codes.
LIVE = "LIVE"
KICKOFF = "KICKOFF"
PAUSE = "PAUSE"  # goal pause (post-goal celebration)
REPLAY = "REPLAY"
FROZEN = "FROZEN"  # frozen with no matching anchor window (boundary/offset slack)

# Anchor-window codes (what the anchors say is happening, independent of motion).
_WIN_KICKOFF = "kickoff"
_WIN_GOAL_PAUSE = "goal_pause"
_WIN_REPLAY = "replay"
_WIN_PLAY = "play"


@dataclass
class Badge:
    """One step's display badge: a `code`, a short `label`, and whether to show a frozen note."""

    code: str
    label: str
    frozen: bool


def anchor_window_at_master_sec(events, master_sec: float) -> str:
    """Which anchor window a master-clock time falls in: kickoff / goal_pause / replay / play.

    `events` is the full anchor list (pass the match-wide anchors, not the in-window subset)."""
    win = _WIN_PLAY
    for e in sorted(events, key=lambda x: x.master_sec):
        if e.master_sec > master_sec:
            break
        name = e.event_name
        if name == "KickoffStarted":
            win = _WIN_KICKOFF
        elif name == "KickoffEnded":
            win = _WIN_PLAY
        elif name == "GoalScored":
            win = _WIN_GOAL_PAUSE
        elif name == "GoalReplayStarted":
            win = _WIN_REPLAY
        elif name == "GoalReplayEnded":
            win = _WIN_PLAY  # play resumes at the following KickoffStarted (usually same instant)
    return win


def frozen_steps(persp_physics: list[FrameState], eps_uu: float = 2.0) -> list[bool]:
    """Per-step flag: True where live physics is frozen (ball + all cars unmoved vs a neighbour).

    Motion is the source of truth for whether the radar is moving. Each step is compared to its
    successor; the last step is compared to its predecessor (so a lone trailing step is judged too)."""
    n = len(persp_physics)
    if n < 2:
        return [False] * n

    def _still(a: FrameState, b: FrameState) -> bool:
        moved = np.linalg.norm(np.array(_xyz(a["ball"]["location"])) - _xyz(b["ball"]["location"]))
        for ca, cb in zip(a["cars"], b["cars"]):
            moved += np.linalg.norm(np.array(_xyz(ca["location"])) - _xyz(cb["location"]))
        return bool(moved < eps_uu)

    out = [_still(persp_physics[i], persp_physics[i + 1]) for i in range(n - 1)]
    out.append(_still(persp_physics[-1], persp_physics[-2]))
    return out


def _windows_for_clip(clip: "MatchClip", perspective: int) -> list[str] | None:
    """Per-step anchor window for a clip, or None if anchors/global indices are unavailable."""
    anchors = clip.metadata[perspective].get("anchors") if clip.metadata else None
    g_idx = clip.global_frame_indices
    if not anchors or g_idx is None:
        return None
    events = parse_anchors(anchors)
    offset = clip.recording_offsets[perspective]
    return [anchor_window_at_master_sec(events, offset + g / clip.src_fps) for g in g_idx]


def step_badges(clip: "MatchClip", perspective: int = 0) -> list[Badge]:
    """Per-step display badge for a clip (length T). MOTION decides frozen vs moving; ANCHORS supply
    the reason. The badge can never be both "moving/LIVE" and "frozen":

      moving + kickoff window -> KICKOFF (no frozen note; cars are driving to the ball)
      moving otherwise        -> LIVE
      frozen + replay window  -> REPLAY      (frozen note)
      frozen + goal-pause win  -> PAUSE      (frozen note)
      frozen + no window match -> FROZEN     (neutral frozen note; never green LIVE)
    """
    t = int(clip.actions.shape[1])
    frozen = frozen_steps(clip.physics[perspective]) if clip.physics is not None else [False] * t
    windows = _windows_for_clip(clip, perspective)

    badges: list[Badge] = []
    for i in range(t):
        is_frozen = frozen[i] if i < len(frozen) else False
        win = windows[i] if windows is not None else None
        if not is_frozen:
            if win == _WIN_KICKOFF:
                badges.append(Badge(KICKOFF, "KICKOFF", False))
            else:
                badges.append(Badge(LIVE, "● LIVE", False))
        else:
            if win == _WIN_REPLAY:
                badges.append(Badge(REPLAY, "REPLAY · ⏸ physics frozen", True))
            elif win == _WIN_GOAL_PAUSE:
                badges.append(Badge(PAUSE, "GOAL PAUSE · ⏸ physics frozen", True))
            else:
                badges.append(Badge(FROZEN, "⏸ physics frozen", True))
    return badges


@dataclass
class Check:
    """One consistency assertion: a human label, whether it passed, and a short detail string."""

    label: str
    ok: bool
    detail: str = ""


def consistency_checks(clip: "MatchClip") -> list[Check]:
    """Frame-alignment / sanity checks across a clip's frames, actions and physics.

    Each :class:`Check` is independent and never raises, so the explorer can render a full PASS/FAIL
    table even when one mapping is broken. The match clock's per-perspective availability is reported
    as a non-fatal informational check.
    """
    checks: list[Check] = []
    p = len(clip.player_ids)
    t = int(clip.actions.shape[1])

    checks.append(Check("≥1 perspective", p >= 1, f"P={p}"))

    if clip.physics is None:
        checks.append(Check("physics present", False, "clip.physics is None"))
        return checks
    checks.append(Check("physics present", True, f"{p} perspectives"))

    # len(frames) == len(actions) == len(physics), per perspective.
    n_frames = None if clip.frames is None else int(clip.frames.shape[1])
    lens_ok = all(len(pp) == t for pp in clip.physics)
    if n_frames is not None:
        lens_ok = lens_ok and n_frames == t
    detail = f"T_actions={t}" + (f", T_frames={n_frames}" if n_frames is not None else "")
    detail += f", T_physics={[len(pp) for pp in clip.physics]}"
    checks.append(Check("frames == actions == physics (per perspective)", lens_ok, detail))

    # All perspectives observe the SAME world, but each is recorded on its own clock (see
    # `recording_offset_sec`), so chunk-local frame f is a slightly different master-time instant
    # per perspective — the ball tracks agree only up to a ~1-frame shift. Reports the best
    # cross-perspective residual after allowing a +/-1 frame lag; a small residual means the
    # perspectives share one world (just sub-frame-misaligned in time).
    if p > 1:
        b0 = ball_track(clip.physics[0])
        residuals = [_min_lag_residual(b0, ball_track(clip.physics[pi]), 1) for pi in range(1, p)]
        worst = max(residuals) if residuals else 0.0
        offs = [round(o, 3) for o in clip.recording_offsets]
        checks.append(
            Check(
                "perspectives share one world (ball ≈, ±1-frame lag)",
                worst < 30.0,  # uu; well under a car length, vs ~50 uu at zero lag
                f"worst residual {worst:.1f} uu (offsets {offs})",
            )
        )

    # is_local picks exactly one car, and it matches the perspective's player_id.
    local_ok, local_detail = True, ""
    for pi, pp in enumerate(clip.physics):
        try:
            lc = local_car(pp[0])
        except ValueError as exc:
            local_ok, local_detail = False, str(exc)
            break
        if lc["player_id"] != clip.player_ids[pi]:
            local_ok = False
            local_detail = f"persp {pi}: is_local player {lc['player_id']} != {clip.player_ids[pi]}"
            break
    checks.append(Check("is_local ↔ perspective player_id", local_ok, local_detail))

    # Physics actually evolves (ball not frozen) — catches a stuck/duplicated state stream.
    moves = ball_moves(clip.physics[0])
    track = ball_track(clip.physics[0])
    path = 0.0 if len(track) < 2 else float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())
    checks.append(Check("ball moves over clip", moves, f"path={path:.0f} uu"))

    # N_CARS per frame (2v2), boost in [0, 1].
    n_cars = {len(fr["cars"]) for pp in clip.physics for fr in pp}
    checks.append(
        Check(f"{N_CARS} cars every frame (2v2)", n_cars == {N_CARS}, f"car counts={sorted(n_cars)}")
    )
    boosts = [c.get("boost_amount", 0.0) for pp in clip.physics for fr in pp for c in fr["cars"]]
    bmin, bmax = (min(boosts), max(boosts)) if boosts else (0.0, 0.0)
    checks.append(Check("boost in [0, 1]", 0.0 <= bmin and bmax <= 1.0, f"[{bmin:.2f}, {bmax:.2f}]"))

    # The match clock is logged only on the local perspective; informational, never fails the suite.
    has_clock = perspective_has_clock(clip.physics[0])
    checks.append(
        Check(
            "match clock present (logged on the local perspective)",
            True,  # informational
            "clock advances" if has_clock else "≡ 0.0 — clock is carried by another (local) perspective",
        )
    )
    return checks
