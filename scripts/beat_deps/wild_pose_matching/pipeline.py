from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np


def augment_projected(sequence: np.ndarray, noise_std: float, shift: int, rng: np.random.Generator) -> np.ndarray:
    x=np.asarray(sequence,np.float32); out=np.full_like(x, x.mean(axis=0))
    keep=max(1,len(x)-abs(shift)); src=max(0,-shift); dst=max(0,shift)
    out[dst:dst+keep]=x[src:src+keep]
    return out+rng.normal(0,noise_std,out.shape).astype(np.float32)


def normalize_neck(x: np.ndarray, neck: int, coords: int) -> np.ndarray:
    shape=x.shape; p=x.reshape(shape[0],-1,coords)
    if neck>=p.shape[1]: raise ValueError("neck index outside skeleton")
    return (p-p[:,neck:neck+1]).reshape(shape)


def sixgrams(text: str):
    words=re.findall(r"[\w']+",text)
    return [" ".join(words[i:i+6]) for i in range(0,len(words),6)]


def cluster_latents(latents: np.ndarray, n_clusters: int, seed: int=0):
    from sklearn.cluster import BisectingKMeans
    n=min(n_clusters,len(latents))
    labels=BisectingKMeans(n_clusters=n,random_state=seed).fit_predict(latents)
    centroids=np.stack([latents[labels==i].mean(0) for i in range(n)])
    centroids/=np.linalg.norm(centroids,axis=1,keepdims=True).clip(1e-8)
    return labels,centroids


def build_rules(text_embeddings: np.ndarray, texts: list[str], wild_latents: np.ndarray, unit_latents: np.ndarray, unit_ids: list[str], cluster_labels: np.ndarray):
    scores=wild_latents@unit_latents.T; nearest=scores.argmax(1)
    return [{"text":t,"text_embedding":e.tolist(),"gesture_id":unit_ids[j],"cluster_id":int(cluster_labels[j]),"pose_match":float(scores[i,j])} for i,(t,e,j) in enumerate(zip(texts,text_embeddings,nearest))]


def retrieve(text: str, rules: list[dict], encode, clusters: dict[int,list[str]], seed: int=0):
    rng=np.random.default_rng(seed); matrix=np.asarray([r["text_embedding"] for r in rules],np.float32); matrix/=np.linalg.norm(matrix,axis=1,keepdims=True).clip(1e-8)
    output=[]
    for chunk in sixgrams(text):
        q=np.asarray(encode([chunk])[0],np.float32); q/=max(np.linalg.norm(q),1e-8); idx=int(np.argmax(matrix@q)); cluster=int(rules[idx]["cluster_id"]); choices=clusters[cluster]
        output.append({"text":chunk,"cluster_id":cluster,"gesture_id":str(rng.choice(choices)),"similarity":float(matrix[idx]@q)})
    return output


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text("".join(json.dumps(v)+"\n" for v in values),encoding="utf-8")
