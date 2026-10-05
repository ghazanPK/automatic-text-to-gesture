"""Paper-method preparation hook and prepared demo endpoints on tiny synthetic BEAT fixtures."""
import argparse
import contextlib
import io
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_fixture  # noqa: E402
import build_glove_subset  # noqa: E402
import paper_method_common as pm  # noqa: E402
import prepare_paper_method as prep  # noqa: E402

SPEAKERS = "1,2,3,4"


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = prep.main(argv)
    return code, json.loads(out.getvalue().strip().splitlines()[-1])


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    base = tmp_path_factory.mktemp("automatic")
    source = beat_fixture.make_processed(base / "processed", speakers=SPEAKERS.split(","), takes=2)
    glove = beat_fixture.make_glove(base / "glove.txt")
    argv = ["--processed", str(source), "--glove", str(glove), "--output-root", str(base / "out"), "--speakers", SPEAKERS]
    code, result = run(argv)
    return {"code": code, "result": result, "argv": argv, "glove": glove, "source": source}


def manifest_of(result):
    return json.loads((Path(result["server_args"][2]) / "manifest.json").read_text(encoding="utf-8"))


def test_processed_route_mines_disjoint_video_against_library_bank(prepared):
    assert prepared["code"] == 0 and prepared["result"]["ready"] is True, prepared["result"]
    folder = Path(prepared["result"]["server_args"][2])
    manifest = manifest_of(prepared["result"])
    roles = manifest["roles"]["roles"]
    assert manifest["roles"]["unit"] == "speaker" and not set(roles["library"]) & set(roles["video"])
    assert manifest["roles"]["probe_takes"] and len(manifest["files"]["clips"]) == 3  # one video take held out
    calibration = json.loads(next(folder.glob("calibration-p95.json")).read_text(encoding="utf-8"))
    assert manifest["threshold"] == round(calibration["threshold"], 2)
    bank = np.load(pm.resolve(manifest["files"]["bank"]))
    rules = [json.loads(x) for x in (folder / "rules.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rules and {r["gesture_id"] for r in rules} <= set(bank.files)
    assert {r["source"] for r in rules} <= {Path(c).stem for c in manifest["files"]["clips"]}
    assert all(len(r["phrase"].split()) <= 5 and r["score"] >= manifest["threshold"] for r in rules)
    metrics = manifest["metrics"]
    assert metrics["rules"] == len(rules) and metrics["heldout_windows"] > 0
    assert 0 <= metrics["heldout_top1"] <= 1 and metrics["heldout_chance"] == round(1 / metrics["distinct_rule_gestures"], 4)
    library = np.load(pm.resolve(manifest["files"]["library"]))
    assert set(map(str, library["ids"])) == set(bank.files) and library["motion"].shape[1:] == (45, 33)


def test_cached_and_paper_threshold(prepared, tmp_path):
    code, result = run(prepared["argv"])
    assert code == 0 and result["summary"]["cached"] is True
    code, result = run([*prepared["argv"][:-4], "--output-root", str(tmp_path), "--speakers", SPEAKERS, "--preset", "paper"])
    assert result["ready"] and manifest_of(result)["threshold"] == 0.92


def test_raw_bvh_textgrid_route(tmp_path, prepared):
    raw = beat_fixture.make_raw(tmp_path / "beat_english_v0.2.1", speakers=SPEAKERS.split(","), takes=2)
    code, result = run(["--beat-root", str(raw), "--glove", str(prepared["glove"]), "--output-root", str(tmp_path / "out"),
                        "--speakers", SPEAKERS])
    assert code == 0 and result["ready"], result
    assert manifest_of(result)["source"]["kind"] == "raw"


def test_missing_source_or_glove_is_not_ready(tmp_path, monkeypatch, prepared):
    for name in (pm.ENV_PROCESSED, pm.ENV_RAW, pm.ENV_GLOVE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    code, result = run(["--output-root", str(tmp_path / "out")])
    assert code == 0 and result["ready"] is False and "BEAT" in result["reason"]
    code, result = run(["--processed", str(prepared["source"]), "--output-root", str(tmp_path / "out")])
    assert result["ready"] is False and "GloVe" in result["reason"] and "glove.6B" in result["next_steps"][0]


def test_glove_subset_keeps_frequent_and_beat_words(tmp_path, prepared):
    glove = tmp_path / "glove.txt"
    glove.write_text("".join(f"w{i} 0.1 0.2\n" for i in range(50)) + "welcome 1 0\nzebra 0 1\n", encoding="utf-8")
    vocabulary = build_glove_subset.beat_vocabulary(prepared["source"])
    assert {"welcome", "the"} <= vocabulary
    report = build_glove_subset.build(glove, tmp_path / "subset.txt", top=5, vocabulary=vocabulary)
    lines = (tmp_path / "subset.txt").read_text(encoding="utf-8").splitlines()
    assert [l.split()[0] for l in lines] == ["w0", "w1", "w2", "w3", "w4", "welcome"] and report["words"] == 6
    raw = beat_fixture.make_raw(tmp_path / "raw", speakers=("1",), seconds=6)
    assert build_glove_subset.beat_vocabulary(raw) & set(beat_fixture.VOCABULARY)  # raw TextGrid route


@pytest.fixture(scope="module")
def server(prepared):
    import demo_server
    args = argparse.Namespace(prepared=Path(prepared["result"]["server_args"][2]), example=False, glove=None,
                              host="127.0.0.1", port=0)
    httpd = demo_server.make_server(args)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def test_prepared_server_library_threshold_and_idle(server, prepared):
    library = get(server + "/api/beat-library")
    assert library["ready"] and library["prepared"] and library["clips"] and library["suggested_queries"]
    threshold = library["default_threshold"]
    assert "percentile" in library["threshold_rule"] and threshold == manifest_of(prepared["result"])["threshold"]
    query = library["suggested_queries"][0]
    for path in ("/api/beat-query", "/api/query"):
        result = get(server + path + "?" + urllib.parse.urlencode({"text": query, "threshold": threshold, "seed": 0}))
        slot = result["slots"][0]
        assert slot["route"] == "mined_pose_rule" and slot["confidence"] == pytest.approx(1.0, abs=1e-4)
        assert slot["gesture_id"] in {c["id"] for c in library["clips"]} and len(slot["frames"]) == 45
    loose = get(server + "/api/beat-query?" + urllib.parse.urlencode({"text": query, "threshold": -1, "seed": 0}))
    assert loose["rule_count"] > library["metrics"]["rules"]  # the viewer threshold re-mines Algorithm 1
    idle = get(server + "/api/beat-query?" + urllib.parse.urlencode({"text": "zzzz qqqq"}))
    assert idle["no_match"] is True and idle["slots"][0]["route"] == "idle_no_match"
    with pytest.raises(urllib.error.HTTPError):
        get(server + "/api/beat-query?" + urllib.parse.urlencode({"text": "open", "threshold": 3}))
