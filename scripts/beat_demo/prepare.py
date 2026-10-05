"""Prepare small, local BEAT motion clips without loading speaker arrays into RAM."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np


def npz_member_view(path, name):
    """Map an uncompressed .npy member inside an NPZ by its payload offset."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}: {name} is compressed; convert source to uncompressed NPZ first")
        with open(path, "rb") as stream:
            stream.seek(info.header_offset + 26)
            filename_size = int.from_bytes(stream.read(2), "little")
            extra_size = int.from_bytes(stream.read(2), "little")
            stream.seek(info.header_offset + 30 + filename_size + extra_size)
            import numpy.lib.format as fmt
            version = fmt.read_magic(stream)
            reader = fmt.read_array_header_2_0 if version >= (2, 0) else fmt.read_array_header_1_0
            shape, fortran, dtype = reader(stream)
            if fortran:
                raise ValueError("Fortran ordered arrays are unsupported")
            offset = stream.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def sixd_to_matrix(values):
    """Gram-Schmidt columns, following the standard rotation 6D convention."""
    a = np.asarray(values, dtype=np.float64).reshape(-1, 6)
    x = a[:, :3]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = a[:, 3:] - np.sum(x * a[:, 3:], axis=1, keepdims=True) * x
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    z = np.cross(x, y)
    return np.stack((x, y, z), axis=-1)


def matrix_to_quat(matrix):
    """Return quaternion in xyzw order."""
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q[:] = ((matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s, s / 4)
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(0, 1 + matrix[i, i] - matrix[j, j] - matrix[k, k])) * 2
        q[i] = s / 4
        q[3] = (matrix[k, j] - matrix[j, k]) / s
        q[j] = (matrix[j, i] + matrix[i, j]) / s
        q[k] = (matrix[k, i] + matrix[i, k]) / s
    return (q / np.linalg.norm(q)).tolist()


def fk(local, root, parents, offsets):
    world = np.empty_like(local)
    positions = np.empty((len(parents), 3), dtype=np.float64)
    for i, parent in enumerate(parents):
        if parent < 0:
            world[i] = local[i]
            positions[i] = root + offsets[i]
        else:
            world[i] = world[parent] @ local[i]
            positions[i] = positions[parent] + world[parent] @ offsets[i]
    return positions


def clipped(items, start, end):
    result = []
    for item in items:
        left, right = max(start, item["frame_start"]), min(end, item["frame_end"])
        if right > left:
            result.append({**item, "frame_start": left-start, "frame_end": right-start})
    return result


def emit_clip(dataset, speaker, take, output, max_frames=150, start_frame=0):
    folder = Path(dataset) / str(speaker)
    metadata = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    record = next(x for x in metadata["takes"] if x["sequence_id"] == take)
    if start_frame < 0 or max_frames < 1 or start_frame >= record["frame_end"] - record["frame_start"]:
        raise ValueError("Invalid take-local frame selection")
    count = min(max_frames, record["frame_end"] - record["frame_start"] - start_frame)
    start, end = start_frame, start_frame + count
    motion = folder / "motion.npz"
    rotations = npz_member_view(motion, "rotations")
    roots = npz_member_view(motion, "root")
    parents = npz_member_view(motion, "parents").astype(int).tolist()
    offsets = npz_member_view(motion, "offsets").astype(float)
    names = list(metadata["joints"])
    frames = []
    for index in range(record["frame_start"] + start, record["frame_start"] + end):
        local = sixd_to_matrix(rotations[index])
        root = np.asarray(roots[index], dtype=float)
        frames.append({"quaternions": [matrix_to_quat(m) for m in local],
                       "rootTranslation": root.tolist(),
                       "positions": fk(local, root, parents, offsets).tolist()})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "paperreach.beat-motion.v1", "dataset": metadata["dataset"],
               "datasetVersion": metadata["version"], "speaker": str(speaker),
               "sequenceId": take, "fps": metadata["fps"],
               "takeLocalStartFrame": start,
               "source": {"jointNames": names, "parents": parents, "offsets": offsets.tolist(),
                          "restQuaternions": [[0, 0, 0, 1] for _ in parents],
                          "axisSigns": [-1, 1, 1],
                          "basisNote": "OmniMo Unity humanoid left-negative-X to MPFB anatomical left-positive-X; mirror X while preserving Y-up and forward Z.",
                          "rotationConvention": "local 6D Gram-Schmidt columns to xyzw quaternions",
                          "rootUnits": "metres"},
               "annotations": {"emotion": record.get("emotion"),
                               "words": clipped(record.get("words", []), start, end),
                               "phones": clipped(record.get("phones", []), start, end),
                               "semantic": clipped(record.get("semantic", []), start, end)},
               "modalities": record.get("modalities", {}), "frames": frames}
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return {"file": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "frames": count, "bytes": output.stat().st_size, "sequence_id": take}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Processed BEAT or BEAT2 folder")
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--take", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(emit_clip(args.dataset, args.speaker, args.take, args.output, args.max_frames, args.start_frame)))


if __name__ == "__main__":
    main()
