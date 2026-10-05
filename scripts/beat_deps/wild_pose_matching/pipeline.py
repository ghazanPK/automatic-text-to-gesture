from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np

PAPER_NOISE_VARIANCES = (0.001, 0.01, 0.1)


def add_noise(sequence: np.ndarray, variance: float, rng: np.random.Generator, mask: np.ndarray | None = None) -> np.ndarray:
    """Gaussian noise with the paper's *variance* (sigma = sqrt(variance)) on scale-normalised poses."""
    if variance < 0:
        raise ValueError("noise variance must be non-negative")
    x = np.asarray(sequence, np.float32)
    noise = rng.normal(0, np.sqrt(variance), x.shape).astype(np.float32)
    if mask is not None:
        noise[~np.asarray(mask, bool)] = 0
    return x + noise


def temporal_shift(sequence: np.ndarray, rng: np.random.Generator, mask: np.ndarray | None = None, crop: int = 30, buffer: int | None = None,
                   max_offset: int = 15, fill: str = "mean", offset: int | None = None, start: int | None = None) -> np.ndarray:
    """Paper temporal shift: a random contiguous ``crop``-frame segment of the real frames is placed at a random
    offset in 1..``max_offset`` of a ``buffer``-frame sequence (default: input length, 45 at 15 fps) that is
    pre-filled with the sequence's mean pose (``fill='mean'``) or zero poses (``fill='zero'``)."""
    x = np.asarray(sequence, np.float32)
    real = x[np.asarray(mask, bool)] if mask is not None else x
    if not len(real):
        raise ValueError("sequence has no real frames")
    buffer = len(x) if buffer is None else buffer
    crop = min(crop, len(real), buffer)
    if fill == "mean":
        out = np.broadcast_to(real.mean(0), (buffer, x.shape[-1])).copy()
    elif fill == "zero":
        out = np.zeros((buffer, x.shape[-1]), np.float32)
    else:
        raise ValueError("fill must be mean or zero")
    top = min(max_offset, buffer - crop)
    if offset is None:
        offset = int(rng.integers(1, top + 1)) if top >= 1 else 0
    if start is None:
        start = int(rng.integers(0, len(real) - crop + 1))
    out[offset:offset + crop] = real[start:start + crop]
    return out


def augment_projected(sequence: np.ndarray, rng: np.random.Generator, mask: np.ndarray | None = None, noise_variances=PAPER_NOISE_VARIANCES,
                      shift_prob: float = 0.5, fills=("mean", "zero"), crop: int = 30, max_offset: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """One training view of a projected 2D unit: optional paper temporal shift, then sqrt(variance) Gaussian noise.

    Returns the augmented sequence and its frame mask (all frames are real after a shift, because the
    fill is part of the augmented input).
    """
    x = np.asarray(sequence, np.float32)
    m = np.ones(len(x), bool) if mask is None else np.asarray(mask, bool)
    if shift_prob > 0 and rng.random() < shift_prob:
        x = temporal_shift(x, rng, m, crop=crop, max_offset=max_offset, fill=str(rng.choice(list(fills))))
        m = np.ones(len(x), bool)
    variance = float(rng.choice(list(noise_variances))) if len(noise_variances) else 0.0
    return add_noise(x, variance, rng, m), m


def normalize_neck(x: np.ndarray, neck: int, coords: int) -> np.ndarray:
    shape=x.shape; p=x.reshape(shape[0],-1,coords)
    if neck>=p.shape[1]: raise ValueError("neck index outside skeleton")
    return (p-p[:,neck:neck+1]).reshape(shape)


def sixgrams(text: str, size: int = 6):
    words=re.findall(r"[\w']+",text)
    return [" ".join(words[i:i+size]) for i in range(0,len(words),size)]


def cluster_latents(latents: np.ndarray, n_clusters: int, seed: int=0):
    """Bisecting K-Means on unit latents (scikit-learn's default largest-cluster bisection; cluster sizes are not balanced)."""
    from sklearn.cluster import BisectingKMeans
    n=min(n_clusters,len(latents))
    labels=BisectingKMeans(n_clusters=n,random_state=seed).fit_predict(latents)
    centroids=np.stack([latents[labels==i].mean(0) for i in range(n)])
    centroids/=np.linalg.norm(centroids,axis=1,keepdims=True).clip(1e-8)
    return labels,centroids


def build_rules(text_embeddings: np.ndarray, texts: list[str], wild_latents: np.ndarray, unit_latents: np.ndarray, unit_ids: list[str], cluster_labels: np.ndarray,
                min_pose_match: float | None = None):
    """(TextEmb, cluster) rules: each wild 2D sequence is assigned its nearest 3D unit in GestureCLR latent space."""
    scores=wild_latents@unit_latents.T; nearest=scores.argmax(1)
    return [{"text":t,"text_embedding":e.tolist(),"gesture_id":unit_ids[j],"cluster_id":int(cluster_labels[j]),"pose_match":float(scores[i,j])}
            for i,(t,e,j) in enumerate(zip(texts,text_embeddings,nearest)) if min_pose_match is None or scores[i,j]>=min_pose_match]


def retrieve(text: str, rules: list[dict], encode, clusters: dict[int,list[str]], seed: int=0, *, chunk_words: int = 6, min_similarity: float | None = None,
             idle_id: str | None = "idle", audio_seconds: float | None = None, unit_seconds: dict[str, float] | None = None):
    """Six-gram Sentence-BERT retrieval with random sampling inside the matched cluster.

    With ``audio_seconds`` each chunk gets a slot proportional to its word count (start and duration);
    ``unit_seconds`` adds the natural length of the sampled unit. Below ``min_similarity`` a chunk plays ``idle_id``.
    """
    if not rules:
        raise ValueError("rule map is empty")
    if audio_seconds is not None and audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive")
    rng=np.random.default_rng(seed); matrix=np.asarray([r["text_embedding"] for r in rules],np.float32); matrix/=np.linalg.norm(matrix,axis=1,keepdims=True).clip(1e-8)
    parts=sixgrams(text,chunk_words)
    if not parts:
        return []
    queries=np.asarray(encode(parts),np.float32).reshape(len(parts),-1)
    queries/=np.linalg.norm(queries,axis=1,keepdims=True).clip(1e-8)
    words=[len(p.split()) for p in parts]; total=sum(words); clock=0.0
    output=[]
    for chunk,q,n in zip(parts,queries,words):
        sims=matrix@q; idx=int(np.argmax(sims)); cluster=int(rules[idx]["cluster_id"]); similarity=float(sims[idx])
        entry={"text":chunk,"cluster_id":cluster,"similarity":similarity,"rule_text":rules[idx].get("text")}
        if min_similarity is not None and similarity<min_similarity:
            entry.update(gesture_id=idle_id,map="idle")
        else:
            entry.update(gesture_id=str(rng.choice(clusters[cluster])),map="rule")
            if unit_seconds and entry["gesture_id"] in unit_seconds:
                entry["unit_seconds"]=float(unit_seconds[entry["gesture_id"]])
        if audio_seconds is not None:
            duration=audio_seconds*n/total
            entry.update(start_seconds=clock,duration_seconds=duration); clock+=duration
            if "unit_seconds" in entry:
                entry["playback_rate"]=entry["unit_seconds"]/duration
        output.append(entry)
    return output


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text("".join(json.dumps(v)+"\n" for v in values),encoding="utf-8")
