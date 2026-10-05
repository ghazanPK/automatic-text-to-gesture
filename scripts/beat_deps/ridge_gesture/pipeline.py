from __future__ import annotations
import re
import numpy as np

WORD=re.compile(r"[\w']+")
STOP={"the","a","an","and","or","but","to","of","in","on","at","for","is","are","was","were","be","it","that","this"}

def heuristic_phrases(text,min_words=3,max_words=10,limit=5):
    words=WORD.findall(text); candidates=[]
    for n in range(min_words,min(max_words,len(words))+1):
        for i in range(len(words)-n+1):
            span=words[i:i+n]; score=sum(w.casefold() not in STOP for w in span)/n + .02*n
            candidates.append((score,i," ".join(span)))
    chosen=[]; occupied=set()
    for _,i,phrase in sorted(candidates,reverse=True):
        ids=set(range(i,i+len(phrase.split())))
        if not ids&occupied: chosen.append(phrase); occupied|=ids
        if len(chosen)==limit: break
    return chosen

def align_phrase(phrase,words):
    target=[x.casefold() for x in WORD.findall(phrase)]; stream=[str(x["word"]).casefold() for x in words]
    for i in range(len(stream)-len(target)+1):
        if stream[i:i+len(target)]==target: return int(words[i]["start_frame"]),int(words[i+len(target)-1]["end_frame"])
    raise ValueError(f"phrase is not a contiguous transcript span: {phrase}")

def neck_normalize(x,neck=1,coords=3):
    x=np.asarray(x,np.float32); p=x.reshape(*x.shape[:-1],-1,coords)
    return (p-p[...,neck:neck+1,:]).reshape(x.shape)

def overlapping_segments(words,start,min_words=3,max_words=10):
    """Candidate spans sharing a start position, longest first."""
    limit=min(max_words,len(words)-start)
    if limit<min_words: return []
    return [" ".join(words[start:start+n]) for n in range(limit,min_words-1,-1)]

def fallback_segments(text,size=6):
    w=WORD.findall(text); return [" ".join(w[i:i+size]) for i in range(0,len(w),size)]

def hybrid_retrieve(text,rules,embed,rule_threshold,fallback_text_latents,fallback_ids,encode_fallback):
    if not rules: raise ValueError("rule map is empty")
    if len(fallback_text_latents)==0 or len(fallback_ids)!=len(fallback_text_latents): raise ValueError("fallback index is empty or inconsistent")
    bank=np.asarray([r["embedding"] for r in rules],np.float32); bank/=np.linalg.norm(bank,axis=1,keepdims=True).clip(1e-8); out=[]
    words=WORD.findall(text); start=0
    while start<len(words):
        candidates=overlapping_segments(words,start)
        if candidates:
            queries=np.asarray(embed(candidates),np.float32)
            if queries.ndim!=2 or queries.shape[1]!=bank.shape[1]: raise ValueError("query and rule embedding dimensions differ")
            queries/=np.linalg.norm(queries,axis=1,keepdims=True).clip(1e-8); scores=queries@bank.T; flat=int(scores.argmax()); ci,ri=np.unravel_index(flat,scores.shape)
            if scores[ci,ri]>=rule_threshold:
                phrase=candidates[ci]; out.append({"text":phrase,"gesture_id":rules[ri]["gesture_id"],"source":"rule","similarity":float(scores[ci,ri])}); start+=len(WORD.findall(phrase)); continue
        chunk=" ".join(words[start:start+6]); z=encode_fallback(chunk); score=fallback_text_latents@z; j=int(score.argmax()); out.append({"text":chunk,"gesture_id":fallback_ids[j],"source":"fallback","similarity":float(score[j])}); start+=len(WORD.findall(chunk))
    return out

class GCA:
    def __init__(self,text_clusters=100,gesture_clusters=20,seed=0): self.k=text_clusters; self.gk=gesture_clusters; self.seed=seed
    def fit(self,text,motion):
        from sklearn.cluster import BisectingKMeans
        self.text_model=BisectingKMeans(n_clusters=min(self.k,len(text)),random_state=self.seed).fit(text); labels=self.text_model.labels_; self.gesture_models={}
        for k in np.unique(labels):
            group=motion[labels==k]; n=min(self.gk,len(group)); self.gesture_models[int(k)]=BisectingKMeans(n_clusters=n,random_state=self.seed).fit(group)
        return self
    def score(self,text,motion):
        labels=self.text_model.predict(text); values=[]
        for k,g in zip(labels,motion):
            centers=self.gesture_models[int(k)].cluster_centers_; gn=g/max(np.linalg.norm(g),1e-8); cn=centers/np.linalg.norm(centers,axis=1,keepdims=True).clip(1e-8); values.append(float(np.max(cn@gn)))
        return float(np.mean(values))
