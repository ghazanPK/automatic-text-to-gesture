"""Export mined motion rules in Flow Human's optional rule-map JSON contract."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
from automatic_text_to_gesture.core import read_rules

TOKEN = re.compile(r"[\w']+")


def export_map(rules, bank, glove_path=None, extra_words=(), fps=15):
    rows=[]
    for rule in rules:
        if rule.gesture_id not in bank:
            raise KeyError(f"missing gesture {rule.gesture_id} in bank")
        motion=np.asarray(bank[rule.gesture_id],np.float32)
        if motion.ndim!=3 or motion.shape[-1] not in (2,3) or not np.isfinite(motion).all():
            raise ValueError("bank clips must be finite [F,J,2|3]")
        if motion.shape[-1]==2:
            motion=np.pad(motion,((0,0),(0,0),(0,1)))
        rows.append({"phrase":rule.phrase,"gesture":rule.gesture_id,
                     "frames":motion.tolist(),"fps":fps,
                     "score":rule.score,"source":rule.source})
    result={"rules":rows}
    if glove_path:
        required={word for rule in rules for word in TOKEN.findall(rule.phrase.lower())}
        required.update(word.lower() for word in extra_words)
        vectors={}
        with Path(glove_path).open(encoding="utf-8") as handle:
            for line in handle:
                word, *values=line.strip().split()
                if word in required:
                    vectors[word]=[float(value) for value in values]
        result["vectors"]=vectors
        result["missing_vectors"]=sorted(required-vectors.keys())
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rules",type=Path,required=True)
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--glove",type=Path)
    p.add_argument("--extra-words",type=Path,help="optional query vocabulary, one word per line")
    p.add_argument("--fps",type=int,default=15)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.fps<=0:raise ValueError("fps must be positive")
    b=np.load(a.bank,allow_pickle=False)
    words=a.extra_words.read_text(encoding="utf-8").splitlines() if a.extra_words else ()
    result=export_map(read_rules(a.rules),{key:b[key] for key in b.files},a.glove,words,a.fps)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result),encoding="utf-8")
    print(json.dumps({"rules":len(result["rules"]),"vectors":len(result.get("vectors",{})),"output":str(a.output)}))


if __name__=="__main__":
    main()
