"""Prepare 30 Hz motion directly from one official BEAT BVH file."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from prepare import fk, matrix_to_quat


def rotation(axis, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    if axis == "X": return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == "Y": return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def parse_header(stream):
    names, parents, offsets, channels = [], [], [], []
    stack = []
    pending = None
    end_site = False
    for line in stream:
        parts = line.strip().split()
        if not parts: continue
        if parts[0] == "MOTION": break
        if parts[0] in ("ROOT", "JOINT"):
            pending = len(names)
            names.append(parts[1]); parents.append(stack[-1] if stack else -1)
            offsets.append([0.,0.,0.]); channels.append([])
        elif parts[:2] == ["End", "Site"]:
            end_site = True
        elif parts[0] == "{":
            stack.append(-2 if end_site else pending)
            pending = None
        elif parts[0] == "}":
            stack.pop(); end_site = False
        elif parts[0] == "OFFSET" and stack and stack[-1] >= 0:
            offsets[stack[-1]] = [float(x)/100 for x in parts[1:4]]
        elif parts[0] == "CHANNELS" and stack and stack[-1] >= 0:
            channels[stack[-1]] = parts[2:]
    frames = int(next(stream).split(":",1)[1])
    frame_time = float(next(stream).split(":",1)[1])
    return names, parents, offsets, channels, frames, frame_time


def emit(path, output, max_frames=120, start_frame=0):
    path, output = Path(path), Path(output)
    with path.open(encoding="utf-8") as stream:
        names, parents, offsets, channels, source_frames, dt = parse_header(stream)
        stride = round((1/dt)/30)
        if abs(stride*dt - 1/30) > 1e-4:
            raise ValueError(f"Cannot exact-stride BVH at {1/dt} Hz")
        stop = min(source_frames, (start_frame+max_frames)*stride)
        local_frames = []
        for source_index, line in enumerate(stream):
            if source_index >= stop: break
            if source_index < start_frame*stride or source_index % stride: continue
            values = [float(x) for x in line.split()]
            if len(values) != sum(map(len, channels)):
                raise ValueError(f"BVH channel count mismatch at frame {source_index}")
            cursor = 0
            mats = []
            root = np.zeros(3)
            for joint_channels in channels:
                mat = np.eye(3)
                for channel in joint_channels:
                    value = values[cursor]; cursor += 1
                    if channel.endswith("position"):
                        root["XYZ".index(channel[0])] = value/100
                    else:
                        mat = mat @ rotation(channel[0], value)
                mats.append(mat)
            mats = np.stack(mats)
            local_frames.append({"quaternions": [matrix_to_quat(m) for m in mats],
                                 "rootTranslation": root.tolist(),
                                 "positions": fk(mats, root, parents, np.asarray(offsets)).tolist()})
    if not local_frames: raise ValueError("No frames selected")
    payload = {"schema": "paperreach.beat-motion.v1", "dataset": "beat", "sequenceId": path.stem,
               "fps": 30, "takeLocalStartFrame": start_frame,
               "source": {"jointNames": names, "parents": parents, "offsets": offsets,
                          "restQuaternions": [[0,0,0,1] for _ in names],
                          "rootUnits": "metres", "rotationConvention": "BVH channel order XYZ local Euler"},
               "frames": local_frames}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload,separators=(",", ":")),encoding="utf-8")
    return len(local_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps({"frames": emit(args.bvh,args.output,args.frames,args.start_frame),"output":args.output}))


if __name__ == "__main__": main()
