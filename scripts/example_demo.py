"""Author-created motion and transparent illustrative vectors for a local demo.

No model weights are loaded or claimed to be trained. The real research pipeline
is used with small hand-specified examples so controls remain observable.
"""
from __future__ import annotations
import numpy as np
from export_playback import make_playback

JOINTS=np.array([
    [0,0,0],[0,1,0],[0,1.5,0],[-.35,1.25,0],[-.55,1.1,0],
    [-.7,.9,0],[-.82,.75,0],[.35,1.25,0],[.55,1.1,0],
    [.7,.9,0],[.82,.75,0]],np.float32)


def library():
    clips={}
    for name in ("open_hands","point_right","rest"):
        frames=[]
        for t in np.linspace(0,1,45):
            pose=JOINTS.copy()
            pulse=np.sin(np.pi*t)
            if name=="open_hands":
                pose[[4,5,6],0]-=.15*pulse
                pose[[9,10],0]+=.15*pulse
                pose[[5,6,9,10],1]+=.28*pulse
            elif name=="point_right":
                pose[[9,10],0]+=.45*pulse
                pose[[9,10],1]+=.18*pulse
                pose[[9,10],2]+=.2*pulse
            else:
                pose[[6,10],1]+=.03*pulse
            frames.append(pose)
        clips[name]=np.stack(frames)
    return clips


def words_vector(text):
    tokens=text.lower().split()
    a=sum(token in {"open","hands","hand","welcome","together","move"} for token in tokens)
    b=sum(token in {"point","result","show","chart","look","see"} for token in tokens)
    if not a and not b:
        b=1  # explicit illustrative unknown-word route
    v=np.array([a,b],np.float32)
    return v/max(np.linalg.norm(v),1e-8)


def query(mode,text,params):
    motions=library()
    seed=int(params.get("seed",["0"])[0])
    if mode=="automatic":
        from automatic_text_to_gesture.core import mine_rules,retrieve
        video=np.concatenate([motions["open_hands"][..., :2],motions["point_right"][..., :2]])
        words=[{"word":"open","start_frame":0,"end_frame":20},
               {"word":"hands","start_frame":20,"end_frame":45},
               {"word":"point","start_frame":45,"end_frame":65},
               {"word":"result","start_frame":65,"end_frame":90}]
        threshold=float(params.get("threshold",["0.92"])[0])
        bank={key:value[...,:2] for key,value in motions.items() if key!="rest"}
        rules=mine_rules(video,words,bank,threshold,seed)
        if not rules:raise ValueError("no authored clips pass this pose threshold")
        vectors={word:words_vector(word) for word in ("open","hands","point","result","welcome","together","move","show","chart","look","see")}
        sequence=retrieve(text,rules,vectors,3)
        result=make_playback(sequence,bank)
        result.update({"algorithm":"frame-cosine rule mining + illustrative summed word vectors",
                       "threshold":threshold,"rule_count":len(rules),"trace":sequence})
    elif mode=="wild":
        from wild_pose_matching.pipeline import build_rules,cluster_latents,retrieve
        unit_ids=["open_hands","point_right"]
        latents=np.eye(2,dtype=np.float32)
        labels,_=cluster_latents(latents,2)
        text_rows=["open both hands","point to the result"]
        rules=build_rules(latents,text_rows,latents,latents,unit_ids,labels)
        clusters={int(labels[i]):[unit_ids[i]] for i in range(2)}
        sequence=retrieve(text,rules,lambda xs:np.stack([words_vector(x) for x in xs]),clusters,seed)
        result=make_playback(sequence,motions)
        result.update({"algorithm":"illustrative pose-unit vectors + Bisecting K-Means + six-word retrieval",
                       "cluster_count":2,"seed":seed,"trace":sequence})
    elif mode=="multilingual":
        from multilingual_gesture.pipeline import require_english,retrieve
        language=params.get("language",["en"])[0]
        translations={"결과를 보여 주세요":"point to the result","손을 펼쳐 주세요":"open both hands"}
        english=require_english(text,language,translations)
        rules=[{"english_text":"open both hands","text_embedding":[1.,0.],"cluster_id":0},
               {"english_text":"point to the result","text_embedding":[0.,1.],"cluster_id":1}]
        sequence=retrieve(english,rules,lambda xs:np.stack([words_vector(x) for x in xs]),
                          {0:["open_hands"],1:["point_right"]},seed)
        result=make_playback({"gestures":sequence},motions)
        result.update({"algorithm":"explicit example translation + illustrative clustered text vectors",
                       "source_text":text,"source_language":language,"english_text":english,
                       "cluster_count":2,"seed":seed,"trace":sequence})
    elif mode=="ridge":
        from ridge_gesture.pipeline import hybrid_retrieve
        threshold=float(params.get("threshold",["0.72"])[0])
        rules=[{"phrase":"open both hands","gesture_id":"open_hands","embedding":[1.,0.]}]
        latent=np.array([[0.,1.],[1.,0.]],np.float32)
        ids=["point_right","open_hands"]
        sequence=hybrid_retrieve(text,rules,lambda xs:np.stack([words_vector(x) for x in xs]),
                                 threshold,latent,ids,words_vector)
        result=make_playback(sequence,motions)
        result.update({"algorithm":"threshold-gated rule + illustrative fallback vectors",
                       "threshold":threshold,
                       "rule_count":sum(row["source"]=="rule" for row in sequence),
                       "fallback_count":sum(row["source"]=="fallback" for row in sequence),
                       "trace":sequence})
    else:raise ValueError(mode)
    result["data_label"]="Author-created example motion and illustrative vectors; no trained weights"
    return result
