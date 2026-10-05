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
- Deterministic random selection is exposed through `--seed`. Zero-norm window frames contribute similarity zero.
- The frame-cosine mean is taken over the gesture's real frames only. Centre-padding frames, and all-zero frames of a pre-padded bank, are excluded, so padding does not cap the score of short gestures.
- `mine` runs Algorithm 1's outer loop over many clips (`--video` files or `--manifest`). It writes a calibration report: score percentiles, per-threshold window pass rates and bank pass fractions, and a degeneracy warning. `--threshold-percentile` derives the threshold from the score distribution.
- The Manual map uses an NVBG-like format: a keyword holds several patterns and several gestures, with a priority. It can be imported from XML or CSV. `--map manual|auto|hybrid` selects the map. Hybrid applies manual keyword containment first and falls back to GloVe. Chunks with no vocabulary or no match go idle (or are skipped) instead of raising.
- Threshold, phrase length, chunk size, neck joint, seed and OOV policy are exposed as CLI flags and a JSON config.
- No pose extraction, TTS, renderer, datasets, gesture assets, or pretrained vectors are distributed.

## Acceptance criteria

`mine` writes a JSONL rule map with provenance and score, plus a calibration report. `retrieve` writes ordered gesture slots with route and timing. Malformed shapes and timestamps fail clearly. Tests cover:
- normalization;
- padding-aware scoring, including a 30-frame gesture inside a 45-frame window scoring about 1.0;
- multi-clip mining and calibration;
- Manual, Auto and Hybrid retrieval, and the XML/CSV importers;
- out-of-vocabulary idle and skip handling.


## Interactive data handoff

`scripts/start_demo.py` prepares one official BEAT BVH/TextGrid take into an ignored, locally generated bank and runs the browser. Three seed gestures remain the entire playback bank; disjoint paired windows produce weak associations, yielding 21 local text rules in the current small sample. The trace exposes seed or association routes and selected IDs. This small simulation is distinct from the original GloVe and video-pose CLI contracts above; it does not establish paper-scale rule coverage or matching quality. The older `--example` server path remains an explicitly authored offline fixture. Speech is optional, and motion playback follows speech progress with short clip blends. No public recording or fitted artifact is bundled.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; speaking envelopes approximate mouth motion rather than phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
