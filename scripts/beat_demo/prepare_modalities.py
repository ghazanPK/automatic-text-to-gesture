"""Crop official BEAT audio and ARKit face data to a motion clip."""
import argparse
import hashlib
import json
from pathlib import Path
import wave

LUFA_CHANNELS = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft",
                 "browDownRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight")


def crop(sequence, source, output, frames=120, start_frame=0):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source / f"{sequence}.wav"), "rb") as incoming:
        if incoming.getframerate() != 16000 or incoming.getnchannels() != 1:
            raise ValueError("Expected official 16 kHz mono BEAT WAV")
        incoming.setpos(round(start_frame / 30 * incoming.getframerate()))
        audio = incoming.readframes(round(frames / 30 * incoming.getframerate()))
        wav_path = output / f"{sequence}.wav"
        with wave.open(str(wav_path), "wb") as outgoing:
            outgoing.setparams(incoming.getparams())
            outgoing.writeframes(audio)
    source_face = json.loads((source / f"{sequence}.json").read_text(encoding="utf-8"))
    names = source_face["names"]
    if len(source_face["frames"]) < (start_frame + frames) * 2:
        raise ValueError("Face take shorter than requested 60 Hz window")
    weights = [source_face["frames"][2*(start_frame+i)]["weights"] for i in range(frames)]
    nine = [names.index(name) for name in LUFA_CHANNELS]
    face_path = output / f"{sequence}-face.json"
    face_path.write_text(json.dumps({"schema": "paperreach.beat-face.v1", "fps": 30,
                                     "names": names, "weights": weights,
                                     "lufaNames": LUFA_CHANNELS,
                                     "lufaWeights": [[row[j] for j in nine] for row in weights],
                                     "sourceFaceFps": 60, "sourceFrameIndices": "2*(startFrame+i)",
                                     "startFrame": start_frame},
                                    separators=(",", ":")), encoding="utf-8")
    return [{"path": str(path), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (wav_path, face_path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--source", default="tmp/beat-demo/source")
    parser.add_argument("--output", default="tmp/beat-demo/sample")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(crop(args.sequence, args.source, args.output, args.frames, args.start_frame)))


if __name__ == "__main__":
    main()
