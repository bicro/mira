"""from_hub wiring, without network: snapshot_download is stubbed to return a local prefix tree."""

import json

import huggingface_hub
import pytest

from mira.data import RocketScienceDataset

MID = "2026-05-10T00-00-00Z-abcdef"


def _write_index(prefix_dir):
    prefix_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "match_id": MID,
        "shard": "dataset_00.tar",
        "n_players": 4,
        "chunk_frames": [20],
        "perspectives": [
            {"player_id": 400 + j, "team": j % 2, "frames": 20, "duration": 1.0} for j in range(4)
        ],
    }
    (prefix_dir / "index.json").write_text(json.dumps({"total_samples": 1, "entries": [entry]}))


def test_from_hub_downloads_prefix_and_loads(tmp_path, monkeypatch):
    _write_index(tmp_path / "train")
    calls = {}

    def fake_snapshot_download(repo_id, *, repo_type, revision, allow_patterns):
        calls.update(repo_id=repo_id, repo_type=repo_type, revision=revision, allow_patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    ds = RocketScienceDataset.from_hub("kyutai/rocket-science", split="train")
    assert ds.match_ids() == [MID]
    # only the requested split prefix is fetched, from the dataset repo
    assert calls == {
        "repo_id": "kyutai/rocket-science",
        "repo_type": "dataset",
        "revision": None,
        "allow_patterns": ["train/*"],
    }


def test_subdir_overrides_split(tmp_path, monkeypatch):
    _write_index(tmp_path / "custom")
    seen = {}
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kw: seen.update(kw) or str(tmp_path),
    )
    ds = RocketScienceDataset.from_hub("kyutai/rocket-science", split="train", subdir="custom")
    assert ds.match_ids() == [MID]
    assert seen["allow_patterns"] == ["custom/*"]  # subdir wins over split


def test_missing_index_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **kw: str(tmp_path))
    with pytest.raises(FileNotFoundError):
        RocketScienceDataset.from_hub("kyutai/rocket-science", split="train")  # no train/ written


def test_from_hub_nested_shard_paths(tmp_path, monkeypatch):
    """The index's subfolder shard paths (train/000/dataset_*.tar) flow through `shards=N`."""
    prefix_dir = tmp_path / "train"
    prefix_dir.mkdir(parents=True)
    entries = [
        {
            "match_id": f"{MID}-{i}",
            "shard": shard,
            "n_players": 4,
            "chunk_frames": [20],
            "perspectives": [
                {"player_id": 400 + j, "team": j % 2, "frames": 20, "duration": 1.0}
                for j in range(4)
            ],
        }
        for i, shard in enumerate(["000/dataset_00000.tar", "001/dataset_01000.tar"])
    ]
    (prefix_dir / "index.json").write_text(json.dumps({"total_samples": 2, "entries": entries}))

    patterns_seen = []

    def fake_snapshot_download(repo_id, *, repo_type, revision, allow_patterns):
        patterns_seen.append(allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    ds = RocketScienceDataset.from_hub("kyutai/rocket-science", split="train", shards=1)
    assert patterns_seen == [
        ["train/index.json"],
        ["train/index.json", "train/000/dataset_00000.tar"],
    ]
    assert ds.match_ids() == [f"{MID}-0"]


def test_allow_patterns_match_nested_shards():
    """`train/*` must keep matching nested shards: allow_patterns are fnmatch-based (`*` crosses
    `/`); a switch to path-aware globbing would make `from_hub(split=...)` silently miss them."""
    from huggingface_hub.utils import filter_repo_objects

    kept = list(
        filter_repo_objects(
            ["train/000/dataset_00000.tar", "train/index.json", "test/dataset_00.tar"],
            allow_patterns=["train/*"],
        )
    )
    assert kept == ["train/000/dataset_00000.tar", "train/index.json"]
