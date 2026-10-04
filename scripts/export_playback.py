"""Join retrieval IDs to actual motion clips for the browser viewer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def make_playback(sequence: list[dict] | dict, library: dict[str, np.ndarray], fps: int = 15) -> dict:
    slots = sequence.get("gestures", []) if isinstance(sequence, dict) else sequence
    if fps <= 0 or not slots:
        raise ValueError("playback requires a positive FPS and non-empty retrieval sequence")
    output = []
    for slot in slots:
        gid = str(slot["gesture_id"])
        if gid not in library:
            raise KeyError(f"retrieved gesture {gid!r} is absent from the motion library")
        clip = np.asarray(library[gid], np.float32)
        if clip.ndim == 2:
            if clip.shape[-1] % 3:
                raise ValueError("flattened motion width must be divisible by 3")
            clip = clip.reshape(len(clip), -1, 3)
        if clip.ndim != 3 or clip.shape[-1] not in (2, 3) or not np.isfinite(clip).all():
            raise ValueError("motion clips must be finite [F,J,2|3] arrays")
        if clip.shape[-1] == 2:
            clip = np.pad(clip, ((0, 0), (0, 0), (0, 1)))
        output.append({"gesture_id": gid, "frames": clip.tolist(),
                       "text": slot.get("text", slot.get("english_text", "")),
                       "similarity": slot.get("similarity"),
                       "source": slot.get("source"),
                       "cluster_id": slot.get("cluster_id"),
                       "duration_seconds": slot.get("duration_seconds", len(clip) / fps)})
    canonical = ["Hips", "Neck", "Head", "LeftShoulder", "LeftArm",
                 "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand"]
    count = len(output[0]["frames"][0])
    return {"fps": fps, "joint_order": canonical if count == len(canonical) else [f"joint_{i}" for i in range(count)],
            "slots": output}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True, help="NPZ bank, units or training pairs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=15)
    a = p.parse_args()
    data = np.load(a.motion, allow_pickle=False)
    if "motion3d" in data:
        library = {str(gid): clip for gid, clip in zip(data["ids"], data["motion3d"])}
    elif "motion" in data:
        library = {str(gid): clip for gid, clip in zip(data["ids"], data["motion"])}
    else:
        library = {key: data[key] for key in data.files}
    sequence = json.loads(a.sequence.read_text(encoding="utf-8"))
    playback = make_playback(sequence, library, a.fps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(playback), encoding="utf-8")
    print(json.dumps({"slots": len(playback["slots"]), "output": str(a.output)}))


if __name__ == "__main__":
    main()
