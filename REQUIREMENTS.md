# Requirements derived from the paper

## Paper facts

- Input video records contain timestamped words and 2D upper-body poses; gesture-bank entries are 2D projections of predefined 3D animations.
- Poses are centered on the neck. Gesture clips are padded to a common length.
- Rule mining advances by one gesture-length window, averages frame-wise cosine similarity, keeps candidates at or above `0.92`, and randomly selects one when several pass.
- The aligned phrase is at most five words. Runtime text is handled in five-word chunks.
- A phrase vector is the sum of its 300-dimensional pretrained GloVe word vectors; cosine similarity retrieves the rule.
- Gesture durations are scheduled from speech duration and the number of chunks. The paper also evaluates an optional manual-first hybrid map.

## Reimplementation decisions

- Data uses explicit `.npz` and JSONL contracts rather than the institute pipeline. Pose arrays are `[frames, joints, 2]`; words carry `start_frame` and `end_frame`.
- Deterministic random selection is exposed through `--seed`. Zero-norm frames contribute similarity zero.
- Manual rules, when supplied, win on exact normalized phrase matches; otherwise all entries compete semantically.
- No pose extraction, TTS, renderer, datasets, gesture assets, or pretrained vectors are distributed.

## Acceptance criteria

`mine` writes a JSONL rule map with provenance and score; `retrieve` writes ordered gesture slots and timing; malformed shapes and timestamps fail clearly; the included verification test covers normalization, threshold mining, and semantic retrieval.


## Interactive data handoff

`scripts/start_demo.py` prepares one official BEAT BVH/TextGrid take into an ignored, locally generated bank and runs the browser. Three seed gestures remain the entire playback bank; disjoint paired windows produce weak associations, yielding 21 local text rules in the current small sample. The trace exposes seed or association routes and selected IDs. This small simulation is distinct from the original GloVe and video-pose CLI contracts above; it does not establish paper-scale rule coverage or matching quality. The older `--example` server path remains an explicitly authored offline fixture. Speech is optional, and motion playback follows speech progress with short clip blends. No public recording or fitted artifact is bundled.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; speaking envelopes approximate mouth motion rather than phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
