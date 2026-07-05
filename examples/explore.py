"""A guided tour of the 4-player Rocket League dataset.

Run it as an app, or open it editable to change the match/clip and re-render live:

    pixi run explore     # read-only app view
    pixi run edit        # editable (headless on :2718; SSH-tunnel to view)

Each section explains one facet of the data — what a sample is, the four co-temporal perspectives,
frame-aligned keyboard actions, master-clock events, and per-frame game state — and shows it on a
clip you pick.
"""

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import json

    import marimo as mo

    from mira.data import RocketLeagueDataset, physics, viz

    return RocketLeagueDataset, base64, json, mo, physics, viz


@app.cell
def _(mo):
    mo.md(
        """
        # 🚗 The 4-player Rocket League dataset

        Time-aligned **video · keyboard actions · game events · per-frame game state** for all four
        players of a 2v2 match. This notebook walks through what a sample contains; pick a match and
        clip in the controls below and every section re-renders on it.
        """
    )
    return


@app.cell
def _(mo):
    repo_in = mo.ui.text("kyutai/rocket-science", label="HuggingFace dataset repo")
    split_in = mo.ui.text("test", label="Split")
    local_in = mo.ui.text("", full_width=True, label="…or a local dataset directory (overrides the repo)")
    mo.accordion(
        {
            "⚙️ Data source": mo.vstack(
                [
                    mo.md("Load a split from the Hub, or point at a local dataset directory."),
                    repo_in,
                    split_in,
                    local_in,
                ]
            )
        }
    )
    return local_in, repo_in, split_in


@app.cell
def _(RocketLeagueDataset, local_in, mo, repo_in, split_in):
    try:
        if local_in.value.strip():
            ds = RocketLeagueDataset.from_local(local_in.value.strip())
        else:
            ds = RocketLeagueDataset.from_hub(repo_in.value.strip(), split=split_in.value.strip() or None)
        _err = None
    except Exception as exc:  # surface the load error in the UI rather than crashing the notebook
        ds, _err = None, str(exc)
    mo.stop(
        ds is None,
        mo.callout(mo.md(f"Couldn't load the dataset.\n\n```\n{_err}\n```"), kind="danger"),
    )
    return (ds,)


@app.cell
def _(ds, mo):
    _cf = ds.index.entries[0].chunk_frames
    _desc = f"{_cf[0]} frames each" if len(set(_cf)) == 1 else f"{min(_cf)}–{max(_cf)} frames"
    mo.md(
        f"""
        ## 1 · What a sample is

        The dataset is served as short (~4 s, 80 frames at 20 fps) windows of a match, each bundling
        all four players' synchronised views. A **clip** is a fixed-length, fps-downsampled slice you
        read out of one — pick its length and frame rate in the controls below.

        Loaded **{len(ds.match_ids())}** match(es); the first has **{len(_cf)}** such windows of {_desc}.
        """
    )
    return


@app.cell
def _(ds, mo):
    match_dd = mo.ui.dropdown(ds.match_ids(), value=ds.match_ids()[0], label="Match", searchable=True)
    persp = mo.ui.dropdown(
        ["all", "player1", "player2", "player3", "player4", "random"], value="all", label="Perspective"
    )
    fps = mo.ui.dropdown(["20", "10", "5"], value="10", label="Target fps")
    # Default to a full ~4 s window (40 frames @ 10 fps) rather than a short 1 s loop.
    clip_len = mo.ui.slider(4, 80, value=40, step=4, label="Clip length", show_value=True)
    mo.vstack([match_dd, mo.hstack([persp, fps, clip_len], justify="start", gap=1.5)])
    return clip_len, fps, match_dd, persp


@app.cell
def _(clip_len, ds, fps, match_dd, mo):
    # Enumerate clips without decoding (cheap) to size the selector; an over-long clip raises, which
    # we surface as a hint rather than a crash.
    try:
        _n = len(ds.load_match(match_dd.value, clip_len=clip_len.value, target_fps=int(fps.value), decode=False))
        clip_idx = mo.ui.number(0, max(0, _n - 1), value=0, label=f"Clip (0–{max(0, _n - 1)})")
        _out = clip_idx
    except ValueError as exc:
        clip_idx = None
        _out = mo.callout(str(exc), kind="warn")
    _out
    return (clip_idx,)


@app.cell
def _(clip_idx, clip_len, ds, fps, match_dd, mo, persp):
    mo.stop(clip_idx is None, mo.md("_Reduce the clip length so it fits._"))
    clip = ds.load_match(
        match_dd.value,
        clip_len=clip_len.value,
        target_fps=int(fps.value),
        perspective=persp.value,
        clip_ids=[clip_idx.value],
    )[0]
    mo.md(
        f"**Clip {clip.clip_id}** · {clip.actions.shape[1]} steps @ {clip.target_fps} fps · "
        f"players {clip.player_ids} (teams {clip.teams})"
    )
    return (clip,)


@app.cell
def _(base64, mo):
    def video_html(mp4: bytes):
        _b = base64.b64encode(mp4).decode()
        return mo.Html(
            f'<video controls loop muted playsinline style="max-width:100%;border-radius:10px" '
            f'src="data:video/mp4;base64,{_b}"></video>'
        )

    return (video_html,)


@app.cell
def _(clip, mo, video_html, viz):
    mo.vstack(
        [
            mo.md(
                """
                ## 2 · Four co-temporal perspectives

                Every player records the same match from their own camera. Here all selected
                perspectives are tiled into one **frame-synchronised** video, grouped by team —
                🔵 blue on top, 🟠 orange on the bottom — each framed in its team colour with a
                live keyboard overlay.
                """
            ),
            video_html(viz.clip_grid_video(clip)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 3 · Frames ↔ actions

        The game is keyboard-only. Each frame has a `{"keys": [...]}` line; the loader turns it into a
        multi-hot `(P, T, 9)` tensor over a fixed key vocabulary, OR-ing presses over each downsample
        window so frames and actions stay aligned. Below: when each key is held over the clip, with
        in-window events marked.
        """
    )
    return


@app.cell
def _(clip, viz):
    viz.keystroke_timeline(clip, perspective=0)
    return


@app.cell
def _(clip, mo):
    # `master_sec or 0.0` normalizes a -0.0 (kickoff at the very start) to 0.0.
    _rows = "\n".join(f"- `{e.event_name}` @ {e.master_sec or 0.0:.2f}s" for e in clip.events)
    # Built flush-left (not an indented triple-quote) so the spliced bullet list renders as markdown.
    mo.md(
        "## 4 · Game events on the master clock\n\n"
        "Discrete events (kickoffs, goals, demolitions, goal replays) live on a single match-wide "
        "master clock shared by all perspectives, and map onto each perspective's frames via its "
        "`recording_offset_sec`. `exclude_replays=True` skips clips overlapping goal-replay segments."
        "\n\nEvents overlapping this clip:\n\n" + (_rows or "_(none in this window)_")
    )
    return


@app.cell
def _(clip, json, mo, video_html, viz):
    _parts = [
        mo.md(
            """
            ## 5 · Per-frame game state

            Alongside video and actions, each frame carries the world state. `clip.physics` is one
            list of per-frame state dicts per perspective, frame-aligned with the video. Below: the
            state animated as a top-down arena radar + HUD, then the **actual fields** — one decoded
            frame of state and the per-perspective metadata.
            """
        )
    ]
    if clip.physics is not None:
        _fr = clip.physics[0][0]
        _peek = {
            "game": _fr["game"],
            "ball": _fr["ball"],
            "cars": [_fr["cars"][0], f"... ({len(_fr['cars'])} cars, one shown)"],
        }
        _parts += [
            video_html(viz.clip_grid_with_radar_video(clip)),
            mo.md("**One frame of state** — `clip.physics[0][0]` (the other cars share this shape):"),
            mo.md("```json\n" + json.dumps(_peek, indent=2) + "\n```"),
        ]
    else:
        _parts.append(mo.md("_This dataset carries no physics track._"))
    _parts += [
        mo.md("**Per-perspective metadata** — `clip.metadata[0]`:"),
        mo.md("```json\n" + json.dumps(clip.metadata[0], indent=2, default=str) + "\n```"),
    ]
    mo.vstack(_parts)
    return


if __name__ == "__main__":
    app.run()
