import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_public_data import JOINTS, load_bvh, timed_words
from export_playback import make_playback
from export_flow_map import export_map
from automatic_text_to_gesture.core import Rule


def test_bvh_hierarchy_and_playback(tmp_path):
    lines=["HIERARCHY","ROOT Hips","{","OFFSET 0 0 0",
           "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]
    for joint in JOINTS[1:]:
        lines.extend([f"JOINT {joint}","{","OFFSET 0 1 0","CHANNELS 0"])
    lines.extend(["}"]*len(JOINTS))
    lines.extend(["MOTION","Frames: 2","Frame Time: 0.0666666667",
                  "0 0 0 0 0 0","0 0 0 90 0 0"])
    bvh=tmp_path/"example.bvh";bvh.write_text("\n".join(lines),encoding="utf-8")
    motion=load_bvh(bvh)
    assert motion.shape==(2,len(JOINTS),3)
    assert np.allclose(motion[:,1],0)
    assert not np.allclose(motion[0,2],motion[1,2])
    transcript=tmp_path/"words.jsonl"
    transcript.write_text(json.dumps({"word":"move","start_seconds":0,"end_seconds":.06})+"\n",encoding="utf-8")
    assert timed_words(transcript,15,2)[0]["end_frame"]==1
    sequence=[{"gesture_id":"clip","text":"move","similarity":.8}]
    playback=make_playback(sequence,{"clip":motion})
    assert playback["slots"][0]["frames"][1][2][0] != playback["slots"][0]["frames"][0][2][0]
    with pytest.raises(KeyError):
        make_playback(sequence,{})


def test_flow_rule_map_contains_mined_motion_and_only_requested_vectors(tmp_path):
    glove=tmp_path/"vectors.txt"
    glove.write_text("move 1 0\nforward 1 0\nunrelated 0 1\n",encoding="utf-8")
    rules=[Rule("move forward","clip",.95,0,2)]
    motion=np.zeros((2,3,2),np.float32)
    exported=export_map(rules,{"clip":motion},glove,["missing"])
    assert exported["rules"][0]["gesture"]=="clip"
    assert len(exported["rules"][0]["frames"][0][0])==3
    assert set(exported["vectors"])=={"move","forward"}
    assert exported["missing_vectors"]==["missing"]
