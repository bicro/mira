"""Rendering helpers for inspecting Rocket League clips — frames, key presses, and events.

The primary renderer is :func:`clip_grid_video`: it tiles every perspective of a clip into a single,
frame-synchronised MP4 with a live on-frame keyboard overlay and team-coloured framing. All
perspectives share one video, so playback is in lockstep.

Pure functions; frames/actions may be torch tensors or numpy. Needs numpy + matplotlib + pillow
(the ``viz`` extra). Video encoding shells out to ``ffmpeg`` (provided by the ``pixi`` env).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

from .actions import DEFAULT_RL_KEYS

if TYPE_CHECKING:
    from PIL import ImageDraw, ImageFont

    from .state import FrameState

# --- palette ------------------------------------------------------------------------------------

_BG = (13, 14, 18)  # canvas behind tiles
_TEAM_COLORS = {0: (56, 135, 255), 1: (255, 146, 41)}  # Rocket League blue / orange
_ACCENT = (60, 222, 152)  # lit key
_KEY_IDLE = (255, 255, 255, 30)
_KEY_IDLE_EDGE = (255, 255, 255, 70)

# 9-key cluster, laid out QWE / ASD / Shift-Space-Ctrl.
_KB_POS = {
    "Q": (0, 0), "W": (0, 1), "E": (0, 2),
    "A": (1, 0), "S": (1, 1), "D": (1, 2),
    "LShiftKey": (2, 0), "Space": (2, 1), "LControlKey": (2, 2),
}  # fmt: skip
_KB_LABEL = {
    "Q": "Q", "W": "W", "E": "E", "A": "A", "S": "S", "D": "D",
    "LShiftKey": "SHIFT", "Space": "SPACE", "LControlKey": "CTRL",
}  # fmt: skip


# --- small utilities ----------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


@functools.lru_cache(maxsize=64)
def _font(size: int, bold: bool = True) -> "ImageFont.FreeTypeFont":
    """A crisp TrueType font (DejaVu Sans, bundled with matplotlib), cached by size/weight."""
    import matplotlib.font_manager as fm
    from PIL import ImageFont

    path = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def _fit_font(draw: "ImageDraw.ImageDraw", text: str, max_w: float, max_h: float) -> "ImageFont.FreeTypeFont":
    """Largest bold font for which `text` fits within (max_w, max_h)."""
    for size in range(int(max_h), 6, -1):
        if draw.textlength(text, font=_font(size)) <= max_w:
            return _font(size)
    return _font(7)


def _resize_even(img: np.ndarray, max_w: int) -> np.ndarray:
    """Downscale (preserving aspect) to <= max_w and force even dims (libx264 needs even W,H)."""
    from PIL import Image

    h, w = img.shape[:2]
    nw, nh = (max_w, round(h * max_w / w)) if w > max_w else (w, h)
    nw, nh = nw - nw % 2, nh - nh % 2
    if (nw, nh) != (w, h):
        img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((nw, nh)))
    return img


# --- on-frame overlay ---------------------------------------------------------------------------


def _draw_key(draw: "ImageDraw.ImageDraw", box: tuple[int, int, int, int], label: str, lit: bool) -> None:
    x0, y0, x1, y1 = box
    radius = max(3, (y1 - y0) // 5)
    if lit:
        draw.rounded_rectangle(box, radius, fill=(*_ACCENT, 240), outline=(255, 255, 255, 150), width=2)
        text_color = (8, 16, 12, 255)
    else:
        draw.rounded_rectangle(box, radius, fill=_KEY_IDLE, outline=_KEY_IDLE_EDGE, width=1)
        text_color = (228, 230, 236, 210)
    font = _fit_font(draw, label, (x1 - x0) * 0.82, (y1 - y0) * 0.62)
    tw = draw.textlength(label, font=font)
    ascent, descent = font.getmetrics()
    draw.text(
        (x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - ascent - descent) / 2), label, font=font, fill=text_color
    )


def _draw_keyboard(draw: "ImageDraw.ImageDraw", x: int, y: int, unit: int, pressed: set[str]) -> None:
    """Draw the 3×3 key cluster with its top-left at (x, y); lit keys filled with the accent colour."""
    gap = max(2, unit // 7)
    pad = gap + 1
    extent = 3 * unit + 2 * gap
    draw.rounded_rectangle(
        [x - pad, y - pad, x + extent + pad, y + extent + pad], radius=unit // 3, fill=(12, 14, 20, 150)
    )
    for key, (r, c) in _KB_POS.items():
        bx, by = x + c * (unit + gap), y + r * (unit + gap)
        _draw_key(draw, (bx, by, bx + unit, by + unit), _KB_LABEL[key], key in pressed)


def _draw_pill(
    draw: "ImageDraw.ImageDraw", x: int, y: int, text: str, color: tuple[int, int, int], h: int
) -> None:
    font = _font(int(h * 0.6))
    tw = draw.textlength(text, font=font)
    padx = int(h * 0.42)
    draw.rounded_rectangle([x, y, x + tw + 2 * padx, y + h], radius=h // 2, fill=(*color, 240))
    ascent, descent = font.getmetrics()
    draw.text((x + padx, y + (h - ascent - descent) / 2), text, font=font, fill=(255, 255, 255, 255))


def decorate_frame(
    img_hwc: np.ndarray, act_vec, *, label: str | None = None, team: int = 0, keys=DEFAULT_RL_KEYS
) -> np.ndarray:
    """Composite a live keyboard widget (lit = pressed) and an optional label pill onto one RGB frame."""
    from PIL import Image, ImageDraw

    base = Image.fromarray(np.ascontiguousarray(img_hwc)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size

    pressed = {k for k, v in zip(keys, list(act_vec)) if v}
    unit = max(14, int(0.072 * w))
    margin = max(6, int(0.022 * w))
    kb_extent = 3 * unit + 2 * max(2, unit // 7)
    _draw_keyboard(draw, margin, h - kb_extent - margin, unit, pressed)

    if label:
        _draw_pill(draw, margin, margin, label, _TEAM_COLORS.get(team, (90, 90, 90)), int(0.07 * w))

    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


# --- grid composition + encoding ----------------------------------------------------------------


def _player_display_numbers(teams: list[int]) -> list[int]:
    """Per-perspective 1-based display number in team-grouped order (blue team first), so tiles and
    the HUD both read P1..P4 with the blue players first."""
    order = sorted(range(len(teams)), key=lambda i: teams[i])
    nums = [0] * len(teams)
    for rank, pi in enumerate(order):
        nums[pi] = rank + 1
    return nums


def _compose_grid_frame(
    tiles: list[np.ndarray], teams: list[int], cols: int, gutter: int, border: int
) -> np.ndarray:
    """Lay tiles (same HxW) on a dark canvas with gutters and team-coloured borders. Tiles are
    grouped by team so blue (team 0) fills the top row(s) and orange (team 1) the bottom."""
    order = sorted(range(len(tiles)), key=lambda i: teams[i])  # stable: blue (0) before orange (1)
    tiles = [tiles[i] for i in order]
    teams = [teams[i] for i in order]
    h, w = tiles[0].shape[:2]  # all perspectives share one camera resolution, so tiles[0] sizes them all
    rows = -(-len(tiles) // cols)
    bh, bw = h + 2 * border, w + 2 * border
    canvas = np.full((rows * bh + (rows + 1) * gutter, cols * bw + (cols + 1) * gutter, 3), _BG, np.uint8)
    for i, (tile, team) in enumerate(zip(tiles, teams)):
        r, c = divmod(i, cols)
        y, x = gutter + r * (bh + gutter), gutter + c * (bw + gutter)
        canvas[y : y + bh, x : x + bw] = _TEAM_COLORS.get(team, (90, 90, 90))
        canvas[y + border : y + border + h, x + border : x + border + w] = tile
    return canvas


def _encode_mp4(video_thwc: np.ndarray, fps: int) -> bytes:
    """Encode (T,H,W,3) uint8 RGB to MP4 bytes via the system ffmpeg (rawvideo pipe -> libx264)."""
    import os
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "The `ffmpeg` binary was not found on PATH, but video encoding requires it. "
            "Install ffmpeg (e.g. via your system package manager or conda) and ensure it is on PATH."
        )

    v = np.ascontiguousarray(video_thwc, dtype=np.uint8)
    _, h, w, _ = v.shape
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
        ]  # fmt: skip
        proc = subprocess.run(cmd, input=v.tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[-1500:]}")
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.remove(path)


def clip_grid_video(
    clip,
    *,
    fps: int | None = None,
    keyboard: bool = True,
    cols: int = 2,
    max_tile_w: int = 460,
    gutter: int = 8,
    border: int = 3,
) -> bytes:
    """Render a clip's perspectives as one frame-synced grid MP4 (bytes), ready to embed.

    Tiles are grouped by team (blue on top, orange on the bottom), framed in their team colour, and
    (by default) carry a live keyboard overlay. Single-perspective clips render as a single framed tile.
    """
    if clip.frames is None:
        raise ValueError("clip has no frames; load it with decode=True")
    frames = np.transpose(_to_numpy(clip.frames), (0, 1, 3, 4, 2))  # P,T,H,W,C
    acts = _to_numpy(clip.actions)  # P,T,K
    p, t = frames.shape[0], frames.shape[1]
    fps = int(fps or clip.target_fps)
    cols = min(cols, p)
    pnums = _player_display_numbers(list(clip.teams))

    decorated = np.empty((p,) + (t,) + _resize_even(frames[0, 0], max_tile_w).shape, np.uint8)
    for pi in range(p):
        label = f"P{pnums[pi]}"
        for ti in range(t):
            tile = _resize_even(frames[pi, ti], max_tile_w)
            decorated[pi, ti] = (
                decorate_frame(tile, acts[pi, ti], label=label, team=clip.teams[pi]) if keyboard else tile
            )

    video = np.stack(
        [_compose_grid_frame(list(decorated[:, ti]), clip.teams, cols, gutter, border) for ti in range(t)]
    )
    return _encode_mp4(video, fps)


# --- physics minimap (top-down arena radar) -----------------------------------------------------
#
# The physics view is a top-down radar: ball + 4 cars
# as dots on the arena floor, team-coloured, with a short velocity arrow and a boost ring. Rendered
# per physics frame and animated in lockstep with the clip video. World
# coords -> image: x spans the side walls (left/right), y the goal-to-goal axis (we draw +y up so
# blue's goal sits at the bottom), z is dropped (top-down). See mira.data.physics for the ranges.

_MINIMAP_BG = (18, 20, 26)
_FIELD_FILL = (28, 32, 42)
_FIELD_LINE = (70, 78, 96)
_BALL_COLOR = (236, 238, 244)


def _world_to_img(x: float, y: float, w: int, h: int, pad: int) -> tuple[float, float]:
    """Map an arena (x, y) in uu to pixel (col, row). y is drawn upward; x rightward."""
    from .physics import FIELD_X, FIELD_Y

    fx = (x + FIELD_X) / (2 * FIELD_X)  # 0..1 left->right
    fy = (y + FIELD_Y) / (2 * FIELD_Y)  # 0..1 bottom->top
    col = pad + fx * (w - 2 * pad)
    row = pad + (1 - fy) * (h - 2 * pad)  # invert: +y is up on screen
    return col, row


def _draw_field(draw: "ImageDraw.ImageDraw", w: int, h: int, pad: int) -> None:
    """Pitch outline, halfway line, centre circle, and the two goal mouths."""
    from .physics import FIELD_X, FIELD_Y

    x0, y0 = _world_to_img(-FIELD_X, -FIELD_Y, w, h, pad)  # bottom-left corner (screen)
    x1, y1 = _world_to_img(FIELD_X, FIELD_Y, w, h, pad)  # top-right corner (screen)
    draw.rectangle([x0, y1, x1, y0], fill=_FIELD_FILL, outline=_FIELD_LINE, width=2)
    # halfway line
    ml, mr = _world_to_img(-FIELD_X, 0, w, h, pad), _world_to_img(FIELD_X, 0, w, h, pad)
    draw.line([ml, mr], fill=_FIELD_LINE, width=1)
    # centre circle
    cx, cy = _world_to_img(0, 0, w, h, pad)
    r = 0.10 * (x1 - x0)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=_FIELD_LINE, width=1)
    # goal mouths (goal is ~1786 uu wide, centred on x=0, at y = +/- 5120)
    gw = 893.0
    for team, gy in ((0, -5120.0), (1, 5120.0)):
        gl, _ = _world_to_img(-gw, gy, w, h, pad)
        gr, gyr = _world_to_img(gw, gy, w, h, pad)
        draw.line([(gl, gyr), (gr, gyr)], fill=(*_TEAM_COLORS[team], 255), width=4)


def _draw_dot(
    draw: "ImageDraw.ImageDraw", cx: float, cy: float, r: float, fill, outline=None, width: int = 1
) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)


def minimap_frame(physics_frame: "FrameState", *, size: int = 420, vel_arrows: bool = True) -> np.ndarray:
    """Render one physics frame as a top-down arena radar (RGB uint8, HxWx3).

    Ball is white; cars are their team colour; the local car gets a white ring. Optional short
    velocity arrows hint at heading/speed. Pure given the frame dict (numpy/pillow only)."""
    from PIL import Image, ImageDraw

    from .physics import FIELD_X, FIELD_Y, local_car

    w = size
    h = round(size * (2 * FIELD_Y) / (2 * FIELD_X))  # keep arena aspect (taller than wide)
    pad = max(8, size // 24)
    img = Image.new("RGB", (w, h), _MINIMAP_BG)
    draw = ImageDraw.Draw(img, "RGBA")
    _draw_field(draw, w, h, pad)

    vel_scale = (w - 2 * pad) / (2 * FIELD_X) * 0.45  # uu/s -> px

    def _arrow(px: float, py: float, vx: float, vy: float, color) -> None:
        ex, ey = px + vx * vel_scale, py - vy * vel_scale  # +y up on screen
        draw.line([(px, py), (ex, ey)], fill=(*color, 220), width=2)

    # ball — dot radius grows with height (z), plus a tiny vertical z-bar that gauges aerial height.
    from .physics import FIELD_Z

    bx, by, bz = (physics_frame["ball"]["location"][k] for k in ("x", "y", "z"))
    pbx, pby = _world_to_img(bx, by, w, h, pad)
    if vel_arrows:
        bv = physics_frame["ball"]["velocity"]
        _arrow(pbx, pby, bv["x"], bv["y"], _BALL_COLOR)
    z_frac = max(0.0, min(1.0, bz / FIELD_Z))
    base_r = max(4, size // 70)
    _draw_dot(draw, pbx, pby, base_r * (1 + z_frac), _BALL_COLOR, outline=(0, 0, 0, 180), width=1)
    # z-bar: a short gauge just right of the ball; fill height tracks z.
    zbx = pbx + base_r * 2.4
    zb_h = max(10, size // 12)
    zb_top, zb_bot = pby - zb_h / 2, pby + zb_h / 2
    draw.line([(zbx, zb_top), (zbx, zb_bot)], fill=(*_FIELD_LINE, 200), width=2)
    draw.line([(zbx, zb_bot), (zbx, zb_bot - z_frac * zb_h)], fill=(*_BALL_COLOR, 230), width=3)

    local_id = local_car(physics_frame)["player_id"]
    for car in physics_frame["cars"]:
        loc = car["location"]
        px, py = _world_to_img(loc["x"], loc["y"], w, h, pad)
        color = _TEAM_COLORS.get(car["team"], (140, 140, 140))
        if car.get("attacker_player_id", -1) != -1:  # -> demolished
            color = (90, 90, 96)
        if vel_arrows:
            cv = car["velocity"]
            _arrow(px, py, cv["x"], cv["y"], color)
        r = max(5, size // 56)
        is_local = car["player_id"] == local_id
        outline = (255, 255, 255, 255) if is_local else (0, 0, 0, 160)
        _draw_dot(draw, px, py, r, (*color, 255), outline=outline, width=3 if is_local else 1)
        if car.get("is_supersonic"):  # speed flames -> thin bright ring
            _draw_dot(draw, px, py, r + 3, None, outline=(255, 255, 255, 200), width=1)
    return np.asarray(img)


def minimap_video(clip, *, perspective: int = 0, fps: int | None = None, size: int = 420) -> bytes:
    """Animate the arena radar across a clip, one frame per physics step (MP4 bytes).

    Plays in lockstep with the clip's grid video. Uses one perspective's physics — all perspectives
    share the world state."""
    if clip.physics is None:
        raise ValueError("clip has no physics; load a dataset that carries physics")
    frames = [minimap_frame(fr, size=size) for fr in clip.physics[perspective]]
    fps = int(fps or clip.target_fps)
    return _encode_mp4(np.stack([_resize_even(f, size) for f in frames]), fps)


# --- live HUD + combined grid|radar video -------------------------------------------------------
#
# A compact HUD (score, clock, per-car boost bars + flags) baked into the radar panel so it
# animates frame-by-frame, and a single composited [grid | radar+HUD] video so the gameplay tiles
# and the physics view are locked together in one mp4.

_HUD_BG = (12, 14, 20)
_HUD_TEXT = (228, 230, 236)
_HUD_MUTED = (150, 156, 170)
_BOOST_COLOR = (255, 209, 102)  # amber boost fill
_BOOST_TRACK = (44, 48, 60)


def _draw_boost_bar(draw: "ImageDraw.ImageDraw", box: tuple[int, int, int, int], frac: float, color) -> None:
    """A rounded 0..1 boost gauge filling left-to-right (`frac` is the boost_amount, already 0..1)."""
    x0, y0, x1, y1 = box
    r = (y1 - y0) // 2
    draw.rounded_rectangle(box, r, fill=_BOOST_TRACK)
    fw = max(0.0, min(1.0, frac)) * (x1 - x0)
    if fw >= 2 * r:  # only draw a fill wide enough to round cleanly
        draw.rounded_rectangle([x0, y0, x0 + fw, y1], r, fill=(*color, 255))


def _draw_hud(
    draw: "ImageDraw.ImageDraw",
    x: int,
    y: int,
    w: int,
    h: int,
    frame: "FrameState",
    has_clock: bool,
    player_labels: dict[int, str] | None = None,
) -> None:
    """Bake a live game-state HUD into the rectangle (x, y, w, h): a score/clock header line, then
    one row per car with a team dot, label, a boost bar, and flag glyphs. With ``player_labels``
    (player_id -> "P1".."P4") the rows are ordered and labelled by it; otherwise rows use the raw
    player id and mark the local car with ``<``."""
    from .physics import local_car

    g = frame["game"]
    cars = frame["cars"]
    if player_labels is not None:
        cars = sorted(cars, key=lambda c: player_labels.get(c["player_id"], f"P{c['player_id']}"))
    local_id = local_car(frame)["player_id"]
    known_ids = {c["player_id"] for c in cars}

    line_h = max(14, h // (len(cars) + 2))
    pad = max(4, line_h // 4)

    # Header: score + clock (+ overtime).
    tr = g.get("time_remaining", 0.0)
    clock = f"{int(tr) // 60}:{int(tr) % 60:02d}" if has_clock else "—"
    head = f"BLUE {g.get('score_blue', 0)} - {g.get('score_orange', 0)} ORANGE    {clock}"
    if g.get("is_overtime"):
        head += "  OT"
    hf = _font(max(7, int(line_h * 0.62)))
    draw.text((x, y), head, font=hf, fill=_HUD_TEXT)

    row_y = y + line_h + pad
    label_w = int(w * 0.30)
    bar_x0 = x + label_w
    bar_x1 = x + int(w * 0.74)
    flag_x = bar_x1 + pad * 2
    cf = _font(max(7, int(line_h * 0.55)))
    for c in cars:
        cy = row_y + line_h // 2
        team_color = _TEAM_COLORS.get(c["team"], (140, 140, 140))
        dot_r = max(3, line_h // 6)
        draw.ellipse([x, cy - dot_r, x + 2 * dot_r, cy + dot_r], fill=(*team_color, 255))
        if player_labels is not None:
            plabel = player_labels.get(c["player_id"], f"P{c['player_id']}")
        else:
            plabel = f"P{c['player_id']}" + ("<" if c["player_id"] == local_id else "")
        draw.text((x + 3 * dot_r, row_y + line_h * 0.12), plabel, font=cf, fill=_HUD_TEXT)
        bh = max(6, int(line_h * 0.42))
        _draw_boost_bar(
            draw, (bar_x0, cy - bh // 2, bar_x1, cy + bh // 2), c.get("boost_amount", 0.0), _BOOST_COLOR
        )
        flags = ""
        if not c.get("is_on_ground", True):
            flags += "air "
        if c.get("is_supersonic"):
            flags += "SS "
        atk = c.get("attacker_player_id", -1)
        if atk != -1:  # -> demolished
            if player_labels is not None and atk in player_labels:
                flags += f"DEMO by {player_labels[atk]}"
            elif atk in known_ids:
                flags += f"DEMO by P{atk}"
            else:
                flags += "DEMO"
        draw.text((flag_x, row_y + line_h * 0.12), flags.strip(), font=cf, fill=_HUD_MUTED)
        row_y += line_h


# Badge colours + pill text by code: LIVE green, KICKOFF amber, the frozen reasons
# orange/violet/grey. The pill text is ASCII (the baked DejaVu font has no ⏸ glyph; the pause is
# drawn as an icon in the frozen note below the pill).
_BADGE_COLOR = {
    "LIVE": (60, 222, 152),
    "KICKOFF": (255, 209, 102),
    "PAUSE": (255, 146, 41),
    "REPLAY": (180, 150, 255),
    "FROZEN": (150, 156, 170),
}
_BADGE_PILL = {
    "LIVE": "● LIVE",
    "KICKOFF": "KICKOFF",
    "PAUSE": "GOAL PAUSE",
    "REPLAY": "REPLAY",
    "FROZEN": "FROZEN",
}


def _draw_pause_icon(draw: "ImageDraw.ImageDraw", x: float, y: float, h: float, color) -> float:
    """Two small bars (a pause glyph) at (x, y); returns the x just past the icon."""
    bw = max(2, h / 5)
    gap = bw
    draw.rectangle([x, y, x + bw, y + h], fill=(*color, 255))
    draw.rectangle([x + bw + gap, y, x + 2 * bw + gap, y + h], fill=(*color, 255))
    return x + 2 * bw + gap


def _draw_badge(draw: "ImageDraw.ImageDraw", x: int, y: int, w: int, badge) -> None:
    """Render one step's :class:`physics.Badge` top-left of the radar: a coloured pill with the
    phase name, and (when frozen) a pause-icon + 'physics frozen' note on the line below."""
    color = _BADGE_COLOR.get(badge.code, _BADGE_COLOR["LIVE"])
    name = _BADGE_PILL.get(badge.code, "● LIVE")  # ASCII-safe pill text
    h = max(16, w // 16)
    f = _font(max(8, int(h * 0.58)))
    tw = draw.textlength(name, font=f)
    padx = int(h * 0.4)
    draw.rounded_rectangle([x, y, x + tw + 2 * padx, y + h], radius=h // 3, fill=(10, 12, 18, 220))
    asc, desc = f.getmetrics()
    draw.text((x + padx, y + (h - asc - desc) / 2), name, font=f, fill=(*color, 255))

    if badge.frozen:
        nf = _font(max(7, int(h * 0.42)))
        iy = y + h + 3
        ih = max(7, int(h * 0.42))
        tx = _draw_pause_icon(draw, x + 1, iy, ih, color) + max(3, ih // 2)
        draw.text((tx, iy - 1), "physics frozen", font=nf, fill=(*color, 220))


def radar_panel_frame(
    physics_frame: "FrameState",
    *,
    panel_h: int,
    has_clock: bool = True,
    radar_size: int = 360,
    badge=None,
    player_labels: dict[int, str] | None = None,
) -> np.ndarray:
    """Compose one fixed-height panel: arena radar (with a motion-authoritative phase badge) on top,
    live HUD below (RGB uint8, panel_h tall).

    Sized to a target height so it can sit flush beside a gameplay grid in a combined frame. `badge`
    is a :class:`physics.Badge` (defaults to a plain ``● LIVE`` badge when omitted)."""
    from PIL import Image, ImageDraw

    from .physics import Badge

    if badge is None:
        badge = Badge("LIVE", "● LIVE", False)
    radar = minimap_frame(physics_frame, size=radar_size)
    rh, rw = radar.shape[:2]
    max_radar_h = panel_h - panel_h // 4  # reserve the lower quarter for the HUD
    if rh > max_radar_h:  # scale the radar down (preserve aspect) so the HUD always fits
        new_w = max(2, round(rw * max_radar_h / rh))
        radar = np.asarray(Image.fromarray(np.ascontiguousarray(radar)).resize((new_w, max_radar_h)))
        rh, rw = radar.shape[:2]

    panel = Image.new("RGB", (rw, panel_h), _HUD_BG)
    panel.paste(Image.fromarray(np.ascontiguousarray(radar)), (0, 0))
    draw = ImageDraw.Draw(panel, "RGBA")
    _draw_badge(draw, max(4, rw // 40), max(4, rh // 40), rw, badge)
    hud_y = rh + max(6, panel_h // 60)
    _draw_hud(
        draw,
        max(4, rw // 40),
        hud_y,
        rw - max(8, rw // 20),
        panel_h - hud_y - 4,
        physics_frame,
        has_clock,
        player_labels,
    )
    return np.asarray(panel)


def clip_grid_with_radar_video(
    clip,
    *,
    perspective: int = 0,
    fps: int | None = None,
    keyboard: bool = True,
    cols: int = 2,
    max_tile_w: int = 420,
    gutter: int = 8,
    border: int = 3,
    radar_size: int = 380,
) -> bytes:
    """One locked MP4 per clip: each frame is [ 4-POV gameplay grid | arena radar + live HUD ].

    The gameplay tiles, radar, and HUD share a single video and stay in sync."""
    if clip.frames is None:
        raise ValueError("clip has no frames; load it with decode=True")
    if clip.physics is None:
        raise ValueError("clip has no physics; load a dataset that carries physics")
    from .physics import perspective_has_clock, step_badges

    frames = np.transpose(_to_numpy(clip.frames), (0, 1, 3, 4, 2))  # P,T,H,W,C
    acts = _to_numpy(clip.actions)
    p, t = frames.shape[0], frames.shape[1]
    fps = int(fps or clip.target_fps)
    cols = min(cols, p)
    has_clock = perspective_has_clock(clip.physics[perspective])
    badges = step_badges(clip, perspective)
    pnums = _player_display_numbers(list(clip.teams))
    player_labels = {clip.player_ids[pi]: f"P{pnums[pi]}" for pi in range(p)}

    decorated = np.empty((p,) + (t,) + _resize_even(frames[0, 0], max_tile_w).shape, np.uint8)
    for pi in range(p):
        label = f"P{pnums[pi]}"
        for ti in range(t):
            tile = _resize_even(frames[pi, ti], max_tile_w)
            decorated[pi, ti] = (
                decorate_frame(tile, acts[pi, ti], label=label, team=clip.teams[pi]) if keyboard else tile
            )

    out = []
    for ti in range(t):
        grid = _compose_grid_frame(list(decorated[:, ti]), clip.teams, cols, gutter, border)
        gh = grid.shape[0]
        panel = radar_panel_frame(
            clip.physics[perspective][ti],
            panel_h=gh,
            has_clock=has_clock,
            radar_size=radar_size,
            badge=badges[ti],
            player_labels=player_labels,
        )
        canvas = np.full((gh, grid.shape[1] + gutter + panel.shape[1], 3), _BG, np.uint8)
        canvas[:, : grid.shape[1]] = grid
        canvas[: panel.shape[0], grid.shape[1] + gutter :] = panel
        out.append(_resize_even(canvas, canvas.shape[1]))  # ensure even W/H for libx264
    return _encode_mp4(np.stack(out), fps)


def game_state_md(clip, *, frame: int = 0, perspective: int = 0) -> str:
    """A compact markdown readout of one physics frame: game phase, score, clock (shown on the
    perspective that carries it), overtime, ball height, and per-car boost / flags / attacker."""
    from .physics import local_car, perspective_has_clock, step_badges

    fr = clip.physics[perspective][frame]
    g = fr["game"]
    has_clock = perspective_has_clock(clip.physics[perspective])
    tr = g.get("time_remaining", 0.0)
    clock = f"{int(tr) // 60}:{int(tr) % 60:02d}" if has_clock else "—"
    ot = "  · 🔴 **OVERTIME**" if g.get("is_overtime") else ""
    local_id = local_car(fr)["player_id"]
    known_ids = {c["player_id"] for c in fr["cars"]}
    badge = step_badges(clip, perspective)[frame]
    note = "  — _physics frozen_" if badge.frozen else ""
    bz = fr["ball"]["location"]["z"]
    rows = [
        f"**Phase** `{badge.code}`{note}",
        "",
        f"**Score** 🔵 {g.get('score_blue', 0)} — {g.get('score_orange', 0)} 🟠   ·   "
        f"**Clock** {clock}{ot}   ·   **Ball z** {bz:.0f} uu",
        "",
        "| car | team | boost | flags |",
        "|---|---|---|---|",
    ]
    for c in fr["cars"]:
        tag = "🔵" if c["team"] == 0 else "🟠"
        me = " ⬅︎you" if c["player_id"] == local_id else ""
        flags = []
        if not c.get("is_on_ground", True):
            flags.append("air")
        if c.get("is_supersonic"):
            flags.append("supersonic")
        atk = c.get("attacker_player_id", -1)
        if atk != -1:  # -> demolished
            flags.append(f"💥 demo by P{atk}" if atk in known_ids else "💥 demo")
        rows.append(
            f"| P{c['player_id']}{me} | {tag} | {round(100 * c.get('boost_amount', 0.0))}% | {', '.join(flags) or '—'} |"
        )
    return "\n".join(rows)


# --- static matplotlib views (no ffmpeg) --------------------------------------------------------


def keystroke_timeline(clip, perspective: int = 0, keys: tuple[str, ...] = DEFAULT_RL_KEYS):
    """A colour-coded timeline of key-press intervals for one perspective, with event markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    act = _to_numpy(clip.actions[perspective])  # (T, K)
    t, k = act.shape
    fig, ax = plt.subplots(figsize=(11, 0.32 * k + 1.1))
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("tab10")
    for ki in range(k):
        runs, start = [], None
        for i, on in enumerate(act[:, ki] > 0):
            if on and start is None:
                start = i
            elif not on and start is not None:
                runs.append((start, i - start))
                start = None
        if start is not None:
            runs.append((start, t - start))
        ax.broken_barh(runs, (ki - 0.38, 0.76), facecolors=cmap(ki % 10), edgecolor="none")

    ax.set_yticks(range(k))
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_ylim(-0.6, k - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, t)
    ax.set_xlabel(f"clip step  ({clip.target_fps} fps, {t} steps)")
    ax.set_title(f"key presses — player {clip.player_ids[perspective]} (team {clip.teams[perspective]})",
                 fontsize=10, loc="left")  # fmt: skip
    ax.grid(axis="x", color="0.9", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Events as dashed verticals, named in a top-right legend (no label/title collision).
    fps_src = clip.src_fps  # exact source fps, so markers line up with the loader's frame mapping
    f0 = clip.frame_indices[0]
    seen: set[str] = set()
    for e in clip.events:
        x = (e.frame_index(fps_src, clip.recording_offsets[perspective]) - f0) / clip.stride
        if 0 <= x < t:
            ax.axvline(x, color="0.2", lw=1.1, ls="--", label=None if e.event_name in seen else e.event_name)
            seen.add(e.event_name)
    if seen:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9, handlelength=1.4)
    fig.tight_layout()
    return fig
