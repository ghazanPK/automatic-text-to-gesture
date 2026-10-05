from __future__ import annotations
import re
import numpy as np

def extract_units(motion: np.ndarray,fps=15,min_seconds=2.,max_seconds=3.,variance_threshold=0.,closure_threshold=float("inf")):
    x=np.asarray(motion,np.float32); lo,hi=round(min_seconds*fps),round(max_seconds*fps); units=[]; start=0
    while len(x)-start>=lo:
        candidates=[]
        for length in range(lo,min(hi,len(x)-start)+1):
            clip=x[start:start+length]; closure=float(np.linalg.norm(clip[0]-clip[-1])); variance=float(np.var(clip,axis=0).mean())
            if closure<=closure_threshold and variance>=variance_threshold: candidates.append((closure,-variance,clip))
        if not candidates: start+=lo; continue
        _,_,clip=min(candidates,key=lambda q:(q[0],q[1])); units.append(clip); start+=len(clip)
    return units

def augment_2d(x,rng):
    noise=float(rng.choice([.001,.01,.1])); shift=int(rng.integers(1,16)); out=np.broadcast_to(x.mean(0),x.shape).copy(); keep=min(30,len(x)-shift); out[shift:shift+keep]=x[:keep]
    return out+rng.normal(0,noise,out.shape).astype(np.float32)

def bisect(latents,k,seed=0):
    from sklearn.cluster import BisectingKMeans
    labels=BisectingKMeans(n_clusters=min(k,len(latents)),random_state=seed).fit_predict(latents); n=labels.max()+1
    centers=np.stack([latents[labels==i].mean(0) for i in range(n)]); centers/=np.linalg.norm(centers,axis=1,keepdims=True).clip(1e-8); return labels,centers

def sixgrams(text):
    w=re.findall(r"[\w']+",text); return [" ".join(w[i:i+6]) for i in range(0,len(w),6)]

def require_english(text,source_language,translations):
    if source_language.lower().startswith("en"): return text
    if text not in translations: raise ValueError("non-English input requires an explicit English translation in --translations")
    return translations[text]

def retrieve(english_text,rules,encode,clusters,seed=0,min_similarity=None,idle_id=None):
    rng=np.random.default_rng(seed); bank=np.asarray([r["text_embedding"] for r in rules],np.float32); bank/=np.linalg.norm(bank,axis=1,keepdims=True).clip(1e-8); out=[]
    for chunk in sixgrams(english_text):
        q=np.asarray(encode([chunk])[0],np.float32); q/=max(np.linalg.norm(q),1e-8); sims=bank@q; i=int(sims.argmax()); cid=int(rules[i]["cluster_id"]); gid=idle_id if min_similarity is not None and sims[i]<min_similarity else str(rng.choice(clusters[cid])); out.append({"english_text":chunk,"gesture_id":gid,"cluster_id":cid,"similarity":float(sims[i]),"blend_frames":5})
    return out
