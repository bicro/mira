"""A 16x16 two-player Pong environment, the toy analog of MIRA's Rocket League setup.

Mirrors the paper's data-collection design at toy scale:

- **Two players** on the left/right edges, each with a 2-key action vocabulary (Up/Down; neither
  pressed = stay), the analog of the 9-key Rocket League keyboard.
- **Per-player first-person views**: each player sees the shared world mirrored so their own paddle
  (blue) is always on the left and the opponent (orange) on the right, the analog of the four
  per-player camera perspectives. Cross-view coherence (the ball at mirrored positions) is the toy
  version of the paper's mutual-view consistency.
- **HUD**: score pips on the top row (own score left, opponent right), the analog of the in-game
  clock/score whose persistence MIRA's rollouts are probed for.
- **Arenas**: three background tints, the analog of the three Rocket League maps.
- **Scripted bots with action noise** stand in for the Nexto policy plus the paper's
  noise-injection procedure.
- **Privileged physics state** (ball position/velocity, paddle positions, scores) is logged next to
  every frame for evaluation probes only, never consumed by the world model.

Everything is vectorized over a batch of environments with numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

H = W = 16  # frame size in pixels
PLAY_TOP, PLAY_BOT = 1.0, 15.0  # ball y range (row 0 is the HUD)
PADDLE_H = 4
PADDLE_Y_MIN, PADDLE_Y_MAX = 1, 15 - PADDLE_H + 1  # top row of the paddle
BALL_X_MIN, BALL_X_MAX = 1.0, 14.0  # paddle contact planes (paddles sit at x=0 and x=15)
WIN_SCORE = 5  # first to 5 resets both scores (a "match")

# Role colours (per-view): own paddle is always blue, the opponent orange.
OWN_COLOR = np.array([90, 170, 255], dtype=np.uint8)
OPP_COLOR = np.array([255, 140, 60], dtype=np.uint8)
BALL_COLOR = np.array([255, 255, 255], dtype=np.uint8)
ARENAS = np.array(  # three background tints = the three maps
    [[6, 8, 16], [14, 6, 14], [6, 14, 9]], dtype=np.uint8
)

# Action ids and their multi-hot key encoding [Up, Down].
STAY, UP, DOWN = 0, 1, 2
ACTION_TO_KEYS = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.uint8)
KEY_NAMES = ["Up", "Down"]


@dataclass
class PongState:
    """Vectorized state of ``n`` parallel games. All arrays have leading dim ``n``."""

    n: int
    rng: np.random.Generator
    ball_y: np.ndarray = field(init=False)
    ball_x: np.ndarray = field(init=False)
    vel_y: np.ndarray = field(init=False)
    vel_x: np.ndarray = field(init=False)
    paddle_y: np.ndarray = field(init=False)  # (n, 2) top row of each paddle
    score: np.ndarray = field(init=False)  # (n, 2)
    arena: np.ndarray = field(init=False)  # (n,) arena id in [0, 3)

    def __post_init__(self) -> None:
        self.paddle_y = np.full((self.n, 2), 7, dtype=np.int64)
        self.score = np.zeros((self.n, 2), dtype=np.int64)
        self.arena = self.rng.integers(0, len(ARENAS), size=self.n)
        self.ball_y = np.empty(self.n)
        self.ball_x = np.empty(self.n)
        self.vel_y = np.empty(self.n)
        self.vel_x = np.empty(self.n)
        self._serve(np.ones(self.n, dtype=bool))

    def _serve(self, mask: np.ndarray) -> None:
        """Reset the ball at the centre with a random velocity for the masked games."""
        k = int(mask.sum())
        if k == 0:
            return
        self.ball_y[mask] = self.rng.uniform(5.0, 11.0, size=k)
        self.ball_x[mask] = 7.5
        speed = self.rng.choice([0.6, 0.75, 0.9], size=k)
        self.vel_x[mask] = speed * self.rng.choice([-1.0, 1.0], size=k)
        self.vel_y[mask] = self.rng.uniform(0.2, 0.9, size=k) * self.rng.choice([-1.0, 1.0], size=k)

    def physics(self) -> np.ndarray:
        """Privileged game state (n, 8): ball y/x/vy/vx, paddle ys, scores. Evaluation-only."""
        return np.stack(
            [
                self.ball_y,
                self.ball_x,
                self.vel_y,
                self.vel_x,
                self.paddle_y[:, 0].astype(float),
                self.paddle_y[:, 1].astype(float),
                self.score[:, 0].astype(float),
                self.score[:, 1].astype(float),
            ],
            axis=1,
        ).astype(np.float32)


def step(state: PongState, actions: np.ndarray) -> np.ndarray:
    """Advance every game by one frame. ``actions`` is (n, 2) with values in {STAY, UP, DOWN}.

    Convention (matches the dataset): action ``a_t`` is the control applied at frame ``t`` that
    produces frame ``t+1``.

    Returns per-game sound-event flags (n, 5) uint8 for the transition into the next frame:
    [player-1 moved, player-2 moved, wall bounce, paddle hit, score].
    """
    old_paddle_y = state.paddle_y
    move = np.zeros_like(actions, dtype=np.int64)
    move[actions == UP] = -1
    move[actions == DOWN] = 1
    state.paddle_y = np.clip(state.paddle_y + move, PADDLE_Y_MIN, PADDLE_Y_MAX)
    events = np.zeros((state.n, 5), dtype=np.uint8)
    events[:, :2] = state.paddle_y != old_paddle_y  # actual movement (clamped pushes don't tick)

    state.ball_y += state.vel_y
    state.ball_x += state.vel_x

    # Wall bounce (top/bottom of the play area).
    over = state.ball_y > PLAY_BOT
    state.ball_y[over] = 2 * PLAY_BOT - state.ball_y[over]
    under = state.ball_y < PLAY_TOP
    state.ball_y[under] = 2 * PLAY_TOP - state.ball_y[under]
    state.vel_y[over | under] *= -1
    events[:, 2] = over | under

    # Paddle bounce / scoring at each side's contact plane.
    for side, plane in ((0, BALL_X_MIN), (1, BALL_X_MAX)):
        approaching = state.vel_x < 0 if side == 0 else state.vel_x > 0
        crossed = approaching & ((state.ball_x < plane) if side == 0 else (state.ball_x > plane))
        if not crossed.any():
            continue
        centre = state.paddle_y[:, side] + (PADDLE_H - 1) / 2
        hit = crossed & (np.abs(state.ball_y - centre) <= PADDLE_H / 2 + 0.25)
        # Bounce: reflect off the plane, speed up slightly, add "english" from the hit offset.
        state.ball_x[hit] = 2 * plane - state.ball_x[hit]
        state.vel_x[hit] *= -1.05
        state.vel_x[hit] = np.clip(state.vel_x[hit], -1.1, 1.1)
        state.vel_y[hit] += 0.3 * (state.ball_y[hit] - centre[hit]) / (PADDLE_H / 2)
        state.vel_y[hit] = np.clip(state.vel_y[hit], -1.1, 1.1)
        # Keep a minimum vertical speed so play never degenerates to a horizontal loop.
        slow = hit & (np.abs(state.vel_y) < 0.15)
        state.vel_y[slow] = 0.15 * np.where(state.vel_y[slow] >= 0, 1.0, -1.0)

        events[:, 3] |= hit.astype(np.uint8)

        # Miss: the other player scores; first to WIN_SCORE resets the match.
        goal_line = 0.0 if side == 0 else 15.0
        missed = crossed & ~hit & ((state.ball_x < goal_line) if side == 0 else (state.ball_x > goal_line))
        if missed.any():
            state.score[missed, 1 - side] += 1
            won = missed & (state.score[:, 1 - side] >= WIN_SCORE)
            state.score[won] = 0
            state._serve(missed)
            events[:, 4] |= missed.astype(np.uint8)

    return events


def bot_side_actions(
    paddle_y_side: np.ndarray,
    ball_y_delayed: np.ndarray,
    vel_x_delayed: np.ndarray,
    side: int,
    noise_prob_side: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scripted ball-tracking policy for one player (the toy Nexto).

    The player tracks the (delayed) ball y when the ball approaches their side, otherwise
    recentres. ``noise_prob_side`` (n,) is the probability of replacing the action with a uniformly
    random one — the analog of the paper's action-noise injection.
    """
    n = paddle_y_side.shape[0]
    approaching = vel_x_delayed < 0 if side == 0 else vel_x_delayed > 0
    target = np.where(approaching, ball_y_delayed, 8.0)
    centre = paddle_y_side + (PADDLE_H - 1) / 2
    act = np.full(n, STAY, dtype=np.int64)
    act[centre > target + 0.75] = UP
    act[centre < target - 0.75] = DOWN
    noisy = rng.random(n) < noise_prob_side
    act[noisy] = rng.integers(0, 3, size=int(noisy.sum()))
    return act


def render_views(state: PongState) -> np.ndarray:
    """Render both players' first-person views, (n, 2, 3, 16, 16) uint8.

    Player 0's view is the world as-is; player 1's view is mirrored horizontally so that each
    player sees their own (blue) paddle on the left column and their opponent (orange) on the
    right, with their own score pips growing from the top-left.
    """
    n = state.n
    frames = np.empty((n, 2, 3, H, W), dtype=np.uint8)
    frames[:] = ARENAS[state.arena][:, None, :, None, None]

    idx = np.arange(n)
    rows = np.arange(PADDLE_H)
    ball_r = np.clip(np.rint(state.ball_y), 1, 15).astype(np.int64)
    ball_c = np.clip(np.rint(state.ball_x), 0, 15).astype(np.int64)

    for view in (0, 1):
        own, opp = view, 1 - view
        # Own paddle on the left column, opponent on the right (mirrored for player 1).
        own_rows = state.paddle_y[:, own, None] + rows
        opp_rows = state.paddle_y[:, opp, None] + rows
        frames[idx[:, None], view, :, own_rows, 0] = OWN_COLOR
        frames[idx[:, None], view, :, opp_rows, W - 1] = OPP_COLOR
        # Ball (mirrored x for player 1).
        bc = ball_c if view == 0 else (W - 1) - ball_c
        frames[idx, view, :, ball_r, bc] = BALL_COLOR
        # HUD row 0: own score pips from the left, opponent's from the right.
        for s in range(WIN_SCORE):
            frames[idx, view, :, 0, s] = np.where(
                (state.score[:, own] > s)[:, None], OWN_COLOR, frames[idx, view, :, 0, s]
            )
            frames[idx, view, :, 0, W - 1 - s] = np.where(
                (state.score[:, opp] > s)[:, None], OPP_COLOR, frames[idx, view, :, 0, W - 1 - s]
            )
    return frames


def rollout_episodes(
    n_envs: int,
    n_frames: int,
    seed: int,
    max_delay: int = 3,
) -> dict[str, np.ndarray]:
    """Run ``n_envs`` bot-vs-bot games for ``n_frames`` frames and record everything.

    Returns per-episode arrays:
        frames  (n, 2, T, 3, 16, 16) uint8 — both players' views
        keys    (n, 2, T, 2) uint8         — per-player multi-hot [Up, Down] at each frame
        physics (n, T, 8) float32          — privileged state (evaluation-only)
        events  (n, T, 5) uint8            — sound events audible AT each frame
                                             [p1 moved, p2 moved, wall, paddle hit, score]
    """
    rng = np.random.default_rng(seed)
    state = PongState(n_envs, rng)
    # Per-player reaction delay (frames) and noise probability, fixed per episode.
    delay = rng.integers(1, max_delay + 1, size=(n_envs, 2))
    noise_on = rng.random((n_envs, 2)) < 0.5  # half the players are noise-injected, like the paper
    noise_prob = np.where(noise_on, rng.choice([0.05, 0.1, 0.2], size=(n_envs, 2)), 0.0)

    frames = np.empty((n_envs, 2, n_frames, 3, H, W), dtype=np.uint8)
    keys = np.empty((n_envs, 2, n_frames, 2), dtype=np.uint8)
    physics = np.empty((n_envs, n_frames, 8), dtype=np.float32)
    events = np.zeros((n_envs, n_frames, 5), dtype=np.uint8)

    # Ring buffers of recent ball state for the per-player reaction delay.
    hist_y = np.tile(state.ball_y[:, None], (1, max_delay + 1))
    hist_vx = np.tile(state.vel_x[:, None], (1, max_delay + 1))
    idx = np.arange(n_envs)

    for t in range(n_frames):
        views = render_views(state)
        frames[:, :, t] = views
        physics[:, t] = state.physics()

        actions = np.empty((n_envs, 2), dtype=np.int64)
        for side in (0, 1):
            d = delay[:, side]
            actions[:, side] = bot_side_actions(
                state.paddle_y[:, side], hist_y[idx, -d], hist_vx[idx, -d], side, noise_prob[:, side], rng
            )

        keys[:, 0, t] = ACTION_TO_KEYS[actions[:, 0]]
        keys[:, 1, t] = ACTION_TO_KEYS[actions[:, 1]]

        step_events = step(state, actions)
        # The transition t -> t+1 becomes audible at frame t+1 (when the bounce is visible).
        if t + 1 < n_frames:
            events[:, t + 1] = step_events
        hist_y = np.concatenate([hist_y[:, 1:], state.ball_y[:, None]], axis=1)
        hist_vx = np.concatenate([hist_vx[:, 1:], state.vel_x[:, None]], axis=1)

    return {"frames": frames, "keys": keys, "physics": physics, "events": events}
