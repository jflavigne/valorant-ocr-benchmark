import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import dense_killfeed_timeline as dense
import valorant_killfeed_tracker as tracker


@dataclass
class Event:
    killer: str
    victim: str
    row_color: str = "green"
    probable_enemy_kill: bool = False
    raw_left: str = ""
    raw_right: str = ""
    t: float = 0.0
    weapon: str | None = None
    weapon_score: float | None = None
    weapon_margin: float | None = None
    weapon_vs: str | None = None
    row_band_index: int | None = None
    active: bool = True
    orientation: str | None = "broadcast_left_right"


def observation(ts, rows):
    frame = np.full((4, 4, 3), int(ts) % 255, dtype=np.uint8)
    return dense.build_observation(ts, frame, rows, source_sha256="s", config_sha256="c", elapsed=.1)


def test_source_grid_is_deterministic_and_nested():
    one = dense.timestamp_grid(1.2, 10.2, 1, 0)
    two = dense.timestamp_grid(1.2, 10.2, 2, 0)
    assert one == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert set(two).issubset(one)


def test_frame_safe_cache_key_changes_with_pixels_not_geometry():
    first = np.zeros((5, 8, 3), dtype=np.uint8)
    second = first.copy(); second[0, 0] = 255
    assert tracker.frame_safe_row_cache_key(first, 1, 2, 3) != tracker.frame_safe_row_cache_key(second, 1, 2, 3)
    assert tracker.frame_safe_row_cache_key(first, 1, 2, 3) == tracker.frame_safe_row_cache_key(first.copy(), 1, 2, 3)


def test_easyocr_warmup_runs_once_per_runtime_signature(monkeypatch=None):
    original_reader = tracker.get_easyocr_reader
    original_one = tracker.ocr_killfeed_row_easyocr_one_pass
    original_stacked = tracker.easyocr_parse_stacked_bands
    original_easyocr = tracker.easyocr
    calls = []
    try:
        tracker.easyocr = object()
        tracker._easyocr_warmed_signatures.clear()
        tracker.get_easyocr_reader = lambda: object()
        tracker.ocr_killfeed_row_easyocr_one_pass = lambda _strip: calls.append("one") or ("", "", "", "")
        tracker.easyocr_parse_stacked_bands = lambda _strips: calls.append("stacked") or []
        tracker.warm_easyocr_for_session("easyocr")
        tracker.warm_easyocr_for_session("easyocr")
        assert calls == ["one", "stacked"]
    finally:
        tracker.get_easyocr_reader = original_reader
        tracker.ocr_killfeed_row_easyocr_one_pass = original_one
        tracker.easyocr_parse_stacked_bands = original_stacked
        tracker.easyocr = original_easyocr
        tracker._easyocr_warmed_signatures.clear()


def test_multiple_rows_enter_leave_and_continue_independently():
    a = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0)
    c = Event("C", "D", raw_left="C", raw_right="D", row_band_index=1)
    timeline = dense.compact_observations([
        observation(1, [a]),
        observation(2, [a, c]),
        observation(3, [a, c]),
        observation(4, [c]),
    ], 1)
    runs = {(run["killer"], run["victim"]): run for run in timeline["runs"]}
    assert runs[("A", "B")]["observation_count"] == 3
    assert runs[("A", "B")]["first_observed"] == 1
    assert runs[("A", "B")]["last_observed"] == 3
    assert runs[("C", "D")]["observation_count"] == 3
    assert runs[("C", "D")]["first_observed"] == 2
    assert runs[("C", "D")]["last_observed"] == 4


def test_identical_text_in_different_rows_is_not_merged():
    first = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0)
    second = Event("A", "B", raw_left="A", raw_right="B", row_band_index=1)
    timeline = dense.compact_observations([observation(1, [first, second])], 1)
    assert len(timeline["runs"]) == 2
    assert {run["order"] for run in timeline["runs"]} == {0, 1}


def test_ambiguity_splits_only_affected_row():
    good = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0)
    ambiguous = Event("?", "B", raw_left="", raw_right="B", row_band_index=0)
    other = Event("C", "D", raw_left="C", raw_right="D", row_band_index=1)
    timeline = dense.compact_observations([
        observation(1, [good, other]), observation(2, [ambiguous, other]), observation(3, [good, other])
    ], 1)
    assert len([r for r in timeline["runs"] if r["order"] == 0]) == 3
    assert len([r for r in timeline["runs"] if r["order"] == 1]) == 1


def test_repeated_rows_ignore_unrelated_score_overlay_metadata():
    row = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0)
    first = observation(1, [row]); first["score_overlay"] = [5, 3]
    second = observation(2, [row]); second["score_overlay"] = [6, 3]
    timeline = dense.compact_observations([first, second], 1)
    assert len(timeline["runs"]) == 1
    assert timeline["runs"][0]["observation_count"] == 2


def test_missing_detection_and_ocr_failure_split_continuity():
    row = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0)
    missing = observation(2, [])
    failure = dense.error_observation(4, "s", "c", RuntimeError("fixture failure"))
    timeline = dense.compact_observations([
        observation(1, [row]), missing, observation(3, [row]), failure, observation(5, [row])
    ], 1)
    assert len(timeline["runs"]) == 3
    assert [sample["status"] for sample in timeline["samples"]] == [
        "detected", "not_detected", "detected", "error", "detected"
    ]


def test_isolated_wrong_reading_remains_inspectable():
    good = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0)
    wrong = Event("X", "Y", raw_left="X", raw_right="Y", row_band_index=0)
    timeline = dense.compact_observations([
        observation(1, [good]), observation(2, [wrong]), observation(3, [good])
    ], 1)
    assert [(run["killer"], run["victim"]) for run in timeline["runs"]] == [
        ("A", "B"), ("X", "Y"), ("A", "B")
    ]


def test_orientation_change_splits_continuity():
    left = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0,
                 orientation="broadcast_left_right")
    right = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0,
                  orientation="broadcast_right_left")
    timeline = dense.compact_observations([observation(1, [left]), observation(2, [right])], 1)
    assert len(timeline["runs"]) == 2
    assert [run["orientation"] for run in timeline["runs"]] == [
        "broadcast_left_right", "broadcast_right_left"
    ]


def test_timestamp_gap_and_nonuniform_samples_split_continuity():
    row = Event("A", "B", raw_left="A", raw_right="B", row_band_index=0)
    timeline = dense.compact_observations([
        observation(1, [row]), observation(2, [row]), observation(4.5, [row])
    ], 1)
    assert len(timeline["runs"]) == 2
    assert timeline["runs"][0]["observation_count"] == 2
    assert timeline["runs"][1]["first_observed"] == 4.5


def test_checkpoint_rejects_incompatible_configuration():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.jsonl"
        row = observation(1, [])
        dense.append_checkpoint(path, row)
        try:
            dense.load_checkpoint(path, "s", "different")
        except ValueError as error:
            assert "incompatible checkpoint" in str(error)
        else:
            raise AssertionError("incompatible checkpoint was accepted")


def test_process_resumes_and_preserves_grid_coverage():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        video = tmp_path / "video.mp4"; video.write_bytes(b"fixture-video")
        bands = tmp_path / "bands.json"; bands.write_text("[]")
        calls = []

        def reader(_video, timestamp):
            calls.append(timestamp)
            return np.full((4, 4, 3), int(timestamp), dtype=np.uint8)

        def parser(_frame, **kwargs):
            return [Event("A", "B", raw_left="A", raw_right="B", row_band_index=0, t=kwargs["now"])]

        out = tmp_path / "out"
        receipt = dense.process(video, out, 0, 2, row_bands=bands, frame_reader=reader, parser=parser)
        assert receipt["requested_samples"] == 3
        assert receipt["processed_samples"] == 3
        assert calls == [0, 1, 2]
        calls.clear()
        first_raw_hash = dense.digest_file(out / "observations.jsonl.gz")
        first_timeline_hash = dense.digest_file(out / "timeline.json")
        dense.process(video, out, 0, 2, row_bands=bands, frame_reader=reader, parser=parser)
        assert calls == []
        assert dense.digest_file(out / "observations.jsonl.gz") == first_raw_hash
        assert dense.digest_file(out / "timeline.json") == first_timeline_hash
        assert dense.validate(out)["ok"]
        timeline = json.loads((out / "timeline.json").read_text())
        assert timeline["coverage"]["grid_complete"] is True
        assert timeline["coverage"]["elimination_recall"] is None


class DenseKillfeedTests(unittest.TestCase):
    def test_functions(self):
        for function in (
            test_source_grid_is_deterministic_and_nested,
            test_frame_safe_cache_key_changes_with_pixels_not_geometry,
            test_easyocr_warmup_runs_once_per_runtime_signature,
            test_multiple_rows_enter_leave_and_continue_independently,
            test_identical_text_in_different_rows_is_not_merged,
            test_ambiguity_splits_only_affected_row,
            test_repeated_rows_ignore_unrelated_score_overlay_metadata,
            test_missing_detection_and_ocr_failure_split_continuity,
            test_isolated_wrong_reading_remains_inspectable,
            test_orientation_change_splits_continuity,
            test_timestamp_gap_and_nonuniform_samples_split_continuity,
            test_checkpoint_rejects_incompatible_configuration,
            test_process_resumes_and_preserves_grid_coverage,
        ):
            with self.subTest(function=function.__name__):
                function()


if __name__ == "__main__":
    unittest.main()
