import json
from pathlib import Path

import numpy as np
import pytest
from automatic_text_to_gesture.cli import main as cli_main
from automatic_text_to_gesture.core import (Clip, GestureBank, Rule, average_frame_cosine, calibration_report, center_pad, chunks, import_manual_map,
                                            load_manual_map, match_manual, mine_clips, mine_rules, normalize_pose, retrieve, threshold_from_percentile,
                                            window_scores)

ROOT = Path(__file__).resolve().parents[1]


def test_mine_and_retrieve():
    pose = np.zeros((4, 3, 2), np.float32); pose[:, 2, 0] = [1, 2, 3, 4]
    words = [{"word": "move", "start_frame": 0, "end_frame": 2}, {"word": "forward", "start_frame": 2, "end_frame": 4}]
    rules = mine_rules(pose, words, {"g": pose}, threshold=.99)
    assert rules[0].gesture_id == "g"
    vectors = {"move": np.array([1., 0.]), "forward": np.array([1., 0.])}
    assert retrieve("move forward", rules, vectors)[0]["gesture_id"] == "g"
    assert np.allclose(normalize_pose(pose)[:, 1], 0)
    short=np.ones((2,3,2),np.float32); padded=center_pad(short,4)
    assert np.allclose(padded[[0,3]],0) and np.allclose(padded[1:3],1)


def test_padding_aware_mean_scores_short_gesture_in_long_window():
    """Audit regression: a 30-frame gesture inside a 45-frame window scored at most 0.667."""
    rng = np.random.default_rng(0)
    gesture = rng.normal(size=(30, 6, 2)).astype(np.float32)
    window = rng.normal(size=(45, 6, 2)).astype(np.float32)
    window[7:37] = gesture  # centre-pad offset (45-30)//2
    assert average_frame_cosine(window, gesture) == pytest.approx(1.0, abs=1e-5)
    # A bank already zero-padded to 45 frames is treated the same way.
    assert average_frame_cosine(window, center_pad(gesture, 45)) == pytest.approx(1.0, abs=1e-5)
    # Vectorised bank scoring agrees with the reference function and the 30-frame unit is mined at 0.92.
    other = rng.normal(size=(45, 6, 2)).astype(np.float32)
    bank = {"short": gesture, "long": other}
    gb = GestureBank({k: normalize_pose(v) for k, v in bank.items()}, neck_joint=1)
    nw = normalize_pose(window)
    assert np.allclose(gb.scores(nw), [average_frame_cosine(nw, normalize_pose(v)) for v in bank.values()], atol=1e-5)
    words = [{"word": "hello", "start_frame": 10, "end_frame": 20}]
    rules = mine_rules(window, words, bank, threshold=.92)
    assert [r.gesture_id for r in rules] == ["short"] and rules[0].score > .99


def test_threshold_timestamps_and_vocabulary_guard():
    pose=np.zeros((4,3,2),np.float32); pose[:,2,0]=1
    words=[{"word":"move","start_frame":0,"end_frame":2},{"word":"again","start_frame":1,"end_frame":4}]
    with pytest.raises(ValueError,match="ordered"):
        mine_rules(pose,words,{"g":pose})
    with pytest.raises(ValueError,match="threshold"):
        mine_rules(pose,[],{"g":pose},threshold=1.1)
    rules=[Rule("move","g",1,0,4)]
    with pytest.raises(ValueError,match="overlap"):
        retrieve("unseen",rules,{"move":np.array([1.,0.])},oov="error")


def test_oov_chunk_goes_idle_or_is_skipped_instead_of_raising():
    rules = [Rule("move forward", "g", 1, 0, 4)]
    vectors = {"move": np.array([1., 0.]), "forward": np.array([1., 0.])}
    text = "move forward now please go zebra quantum blorp flux xylo"
    idle = retrieve(text, rules, vectors, audio_seconds=2.0)
    assert [s["gesture_id"] for s in idle] == ["g", "idle"] and idle[1]["map"] == "idle"
    assert idle[1]["start_seconds"] == pytest.approx(1.0)
    skipped = retrieve(text, rules, vectors, audio_seconds=2.0, oov="skip")
    assert [s["gesture_id"] for s in skipped] == ["g"]
    floor = retrieve("move", rules, {"move": np.array([1., 0.]), "forward": np.array([0., 1.])}, min_similarity=.9, idle_id="rest")
    assert floor[0]["gesture_id"] == "rest"


def test_chunking_drops_short_remainder_and_honours_size():
    assert chunks("a b c d e f g") == ["a b c d e"]
    assert chunks("a b c") == ["a b c"]
    assert chunks("a b c d e f", size=3) == ["a b c", "d e f"]


def test_manual_auto_hybrid_maps():
    manual = load_manual_map(ROOT / "examples" / "manual_map.json")
    imported = import_manual_map(ROOT / "examples" / "manual_map_nvbg.xml")
    assert [(r.keyword, r.patterns, r.gestures, r.priority) for r in imported] == [(r.keyword, r.patterns, r.gestures, r.priority) for r in manual]
    # Containment, not whole-chunk equality; higher priority wins.
    rule, pattern = match_manual("we will never stop today", manual)
    assert rule.keyword == "negation" and pattern == "never"
    assert match_manual("it is fine", manual) is None
    rules = [Rule("happy day", "auto_happy", 1, 0, 4), Rule("never again", "auto_never", 1, 0, 4)]
    vectors = {"happy": np.array([1., 0.]), "day": np.array([1., 0.]), "sunny": np.array([1., 0.]), "never": np.array([0., 1.])}
    text = "it is a sunny day we will never stop"
    auto = retrieve(text, rules, vectors, mode="auto")
    assert [s["gesture_id"] for s in auto] == ["auto_happy"]
    hybrid = retrieve("a sunny day for all", rules, vectors, mode="hybrid", manual=manual)
    assert hybrid[0]["map"] == "auto"
    hybrid = retrieve("we will never stop today", rules, vectors, mode="hybrid", manual=manual)
    assert hybrid[0]["map"] == "manual" and hybrid[0]["gesture_id"] == "negation_sweep"
    only_manual = retrieve("a sunny day for all", [], {}, mode="manual", manual=manual)
    assert only_manual[0]["map"] == "idle"
    with pytest.raises(ValueError, match="manual rules"):
        retrieve("x", rules, vectors, mode="manual")


def test_csv_import(tmp_path):
    table = tmp_path / "rules.csv"
    table.write_text("keyword,patterns,gestures,priority\nyes,yes|of course,nod|nod_small,2\n", encoding="utf-8")
    (rule,) = import_manual_map(table)
    assert rule.patterns == ("yes", "of course") and rule.gestures == ("nod", "nod_small") and rule.priority == 2


def _clip(rng, gesture, name):
    pose = rng.normal(size=(45, 4, 2)).astype(np.float32)
    pose[7:37] = gesture
    return Clip(name, pose, [{"word": name, "start_frame": 0, "end_frame": 5}])


def test_multi_clip_mining_and_calibration_report():
    rng = np.random.default_rng(3)
    bank = {f"g{i}": rng.normal(size=(n, 4, 2)).astype(np.float32) for i, n in enumerate((30, 40, 45))}
    clips = [_clip(rng, bank["g0"], "a"), _clip(rng, bank["g0"], "b")]
    rules, report = mine_clips(clips, bank, .92, seed=0)
    assert [r.source for r in rules] == ["a", "b"] and {r.gesture_id for r in rules} == {"g0"}
    assert report["windows"] == 2 and report["gestures"] == 3 and report["at_threshold"]["window_pass_rate"] == 1.0
    assert [c["rules"] for c in report["clips"]] == [1, 1]
    assert any(row["threshold"] == .92 for row in report["grid"])
    _, matrix = window_scores(clips[0].pose, bank)
    assert threshold_from_percentile([matrix], 100) == pytest.approx(matrix.max())
    degenerate = calibration_report([np.full((4, 5), .99)], .92)
    assert "degenerate" in degenerate["warning"]
    _, pct_report = mine_clips(clips, bank, None, threshold_percentile=50)
    assert pct_report["threshold_percentile"] == 50


def test_cli_multi_video_manifest_and_report(tmp_path, capsys):
    rng = np.random.default_rng(5)
    g = rng.normal(size=(30, 4, 2)).astype(np.float32)
    np.savez(tmp_path / "bank.npz", g=g, h=rng.normal(size=(45, 4, 2)).astype(np.float32))
    for name in ("one", "two"):
        clip = _clip(rng, g, name)
        np.savez(tmp_path / f"{name}.npz", pose=clip.pose, words_json=np.asarray(json.dumps(clip.words)))
    (tmp_path / "list.json").write_text(json.dumps(["two.npz"]), encoding="utf-8")
    cli_main(["mine", "--video", str(tmp_path / "one.npz"), "--manifest", str(tmp_path / "list.json"), "--bank", str(tmp_path / "bank.npz"),
              "--output", str(tmp_path / "rules.jsonl")])
    rules = [json.loads(x) for x in (tmp_path / "rules.jsonl").read_text().splitlines()]
    assert [r["source"] for r in rules] == ["one", "two"]
    report = json.loads((tmp_path / "rules.jsonl.calibration.json").read_text())
    assert report["threshold"] == .92 and "window pass rate" in capsys.readouterr().err
    cli_main(["config"])
    assert json.loads(capsys.readouterr().out)["chunk_words"] == 5
