"""Prepared demo mode: serve rules mined from public BEAT to the browser viewer.

``scripts/demo_server.py --prepared outputs/paper-method/<key>`` loads the
artifacts written by ``prepare_paper_method.py`` and answers ``/api/beat-library``,
``/api/beat-query`` and ``/api/query`` with the repository's own Algorithm 2:
five-word chunks, all-MiniLM-L6-v2 phrase vectors (or the optional summed GloVe
vectors when the method was prepared with them), the most similar mined phrase
and its bank gesture. A different threshold or seed from the viewer re-mines the
prepared video clips (Algorithm 1) before retrieval.
"""
from __future__ import annotations

import json

import numpy as np

import paper_method_common as pm

ALGORITHM = ("Automatic Text-to-Gesture: frame-cosine mining of timed <=5-word phrases from projected+corrupted "
             "BEAT video-role takes against a frontal 2D gesture bank (Algorithm 1), five-word chunk retrieval by "
             "the most similar mined phrase (Algorithm 2) with all-MiniLM-L6-v2 phrase vectors in place of the "
             "paper's summed GloVe")
ALGORITHM_GLOVE = ALGORITHM.split(" with all-MiniLM")[0] + " with the paper's summed GloVe vectors"
DATA_LABEL = "Public BEAT, disjoint library/video speakers; projected BEAT motion stands in for video pose"


class PreparedDemo:
    def __init__(self, folder, args=None):
        pm.use_repository_package("automatic_text_to_gesture")
        from automatic_text_to_gesture.core import Clip, read_rules
        self.folder, self.manifest = pm.load_manifest(folder)
        files = self.manifest["files"]
        self.rules = read_rules(self.folder / files["rules"])
        self.mined = {(self.manifest["threshold"], self.manifest["seed"]): self.rules}
        bank = np.load(pm.resolve(files["bank"]))
        self.bank = {k: bank[k] for k in bank.files}
        self.clips = []
        for path in files["clips"]:
            d = np.load(pm.resolve(path))
            self.clips.append(Clip(str(d["clip_id"]), d["pose"], json.loads(str(d["words_json"]))))
        library = np.load(pm.resolve(files["library"]))
        self.motion = {str(i): m for i, m in zip(library["ids"], library["motion"])}
        self.info = {r["id"]: r for r in json.loads((self.folder / files["library_info"]).read_text(encoding="utf-8"))}
        self.rest = np.median(np.stack([m.reshape(len(m), -1, 3)[0] for m in self.motion.values()]), axis=0)
        # Prepared with Sentence-BERT (default) or the optional GloVe; older manifests name only a GloVe file.
        spec = self.manifest.get("text_encoder") or {"kind": "glove", "file": self.manifest.get("glove")}
        self.encoder_kind = "glove" if getattr(args, "glove", None) else spec["kind"]
        self.vectors = {}
        if self.encoder_kind == "glove":
            self.glove = pm.resolve(getattr(args, "glove", None) or spec.get("file") or self.manifest["glove"])
            if not self.glove.is_file():
                raise FileNotFoundError(f"GloVe file {self.glove} is missing; rerun prepare_paper_method.py --glove <file>")
            self.encoder_label = f"glove ({self.glove.name})"
        else:
            from automatic_text_to_gesture.core import SentenceEncoder
            model, why = pm.find_sbert(getattr(args, "sbert", None))
            if model is None:
                raise FileNotFoundError(why)
            self.encoder = SentenceEncoder(model)
            self.encoder_label = self.encoder.label
        self.algorithm = ALGORITHM_GLOVE if self.encoder_kind == "glove" else ALGORITHM

    def library(self):
        clips = [{"id": gid, "text": row.get("text", ""), "duration": len(self.motion[gid]) / pm.FPS,
                  "source": {k: row.get(k) for k in ("speaker", "take", "start_frame", "end_frame", "role", "kind",
                                                     "motion_url", "alignment_url")}}
                 for gid, row in self.info.items()]
        suggested = list(self.manifest["suggested_queries"]) + list(self.manifest["heldout_probes"])
        return {"ready": True, "prepared": True, "mode": "automatic", "clips": clips, "suggested_queries": suggested,
                "heldout_probes": self.manifest["heldout_probes"], "metrics": self.manifest["metrics"],
                "roles": self.manifest["roles"]["roles"], "default_threshold": self.manifest["threshold"],
                "threshold_rule": self.manifest["threshold_rule"], "algorithm": self.algorithm, "data_label": DATA_LABEL,
                "text_encoder": self.encoder_label}

    def _rules(self, threshold, seed):
        from automatic_text_to_gesture.core import mine_clips
        key = (threshold, seed)
        if key not in self.mined:
            if not -1 <= threshold <= 1:
                raise ValueError("threshold must be a cosine value in [-1, 1]")
            if len(self.mined) > 16:
                self.mined = {k: v for k, v in list(self.mined.items())[:1]}
            self.mined[key] = mine_clips(self.clips, self.bank, threshold, seed)[0]
        return self.mined[key]

    def _vectors(self, rules, text):
        if self.encoder_kind != "glove":
            return self.encoder
        from automatic_text_to_gesture.core import TOKEN, load_glove
        needed = {w for r in rules for w in TOKEN.findall(r.phrase.lower())} | set(TOKEN.findall(text.lower()))
        missing = needed - self.vectors.keys()
        if missing:
            self.vectors.update(load_glove(self.glove, missing))
            self.vectors.update({w: None for w in missing - self.vectors.keys()})
        return {w: v for w, v in self.vectors.items() if v is not None and w in needed}

    def query(self, text, params):
        from automatic_text_to_gesture.core import retrieve
        threshold = round(pm.param(params, "threshold", self.manifest["threshold"], float), 4)
        seed = pm.param(params, "seed", self.manifest["seed"], int)
        floor = pm.param(params, "min_similarity", self.manifest.get("min_similarity"), float)
        rules = self._rules(threshold, seed)
        if not rules:
            raise ValueError(f"no mined rule passes threshold {threshold}; lower the pose-cosine threshold")
        sequence = retrieve(text, rules, self._vectors(rules, text), mode="auto", chunk_words=5, seed=seed,
                            oov="idle", min_similarity=floor)
        if not sequence:
            raise ValueError("Query text has no words")
        slots = []
        for entry in sequence:
            if entry["map"] == "idle":
                slots.append(pm.idle_slot(entry["text"], self.rest, entry.get("reason", "no match")))
                continue
            gid = entry["gesture_id"]
            row = self.info.get(gid, {})
            slots.append({"gesture_id": gid, "text": entry["text"], "frames": pm.frames_m(self.motion[gid]),
                          "route": "mined_pose_rule", "confidence": round(entry["similarity"], 5),
                          "similarity": round(entry["similarity"], 5),
                          "source": {k: row.get(k) for k in ("speaker", "take", "start_frame", "end_frame", "kind",
                                                             "motion_url", "alignment_url")},
                          "rule_source": {"rule_phrase": entry.get("rule_phrase"), "threshold": threshold},
                          "blend_frames": 5})
        gestures = {r.gesture_id for r in rules}
        metrics = {k: self.manifest["metrics"].get(k) for k in ("bank_gestures", "heldout_top1", "heldout_chance")}
        metrics.update(rule_count=len(rules), distinct_rule_gestures=len(gestures), threshold=threshold, min_similarity=floor)
        return pm.query_result(slots, algorithm=self.algorithm, data_label=DATA_LABEL, metrics=metrics,
                               trace={"input": text, "retrieval_text": text, "seed": seed, "map": "auto",
                                      "text_encoder": self.encoder_label},
                               joints=pm.ingest().UPPER_BODY,
                               extra={"threshold": threshold, "rule_count": len(rules), "text_encoder": self.encoder_label})
