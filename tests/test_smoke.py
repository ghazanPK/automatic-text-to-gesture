import numpy as np
from automatic_text_to_gesture.core import Rule, center_pad, mine_rules, normalize_pose, retrieve


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
