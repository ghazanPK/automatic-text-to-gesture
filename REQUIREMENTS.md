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
- The Manual map uses an NVBG-like format: a keyword holds several patterns and several gestures, with a priority. It can be imported from XML or CSV. `--map manual|auto|hybrid` selects the map. Hybrid applies manual keyword containment first and falls back to the phrase vectors. Chunks with no vocabulary or no match go idle (or are skipped) instead of raising.
- Threshold, phrase length, chunk size, neck joint, seed and OOV policy are exposed as CLI flags and a JSON config.
- **Text encoder substitution (owner decision, 6 October 2026).** Phrase and chunk vectors come from Sentence-BERT all-MiniLM-L6-v2 (about 92 MB) instead of summed 300-D GloVe vectors (an 820 MB download). Algorithm 1 is unchanged, and Algorithm 2 keeps five-word chunks, the argmax cosine over stored rule phrases and audio-divided timing. A chunk with no whole word in the model's vocabulary is out of vocabulary and idles. `scripts/start_demo.py` downloads the model into ignored `models/all-MiniLM-L6-v2` on first run (`--offline` skips it). GloVe remains an optional alternative (`--glove`, `--encoder glove`), so the paper's sum pooling can still be run.
- No pose extraction, TTS, renderer, datasets, gesture assets, or pretrained vectors are distributed; the MiniLM model is downloaded at first run, never committed.

## Acceptance criteria

`mine` writes a JSONL rule map with provenance and score, plus a calibration report. `retrieve` writes ordered gesture slots with route and timing. Malformed shapes and timestamps fail clearly. Tests cover:
- normalization;
- padding-aware scoring, including a 30-frame gesture inside a 45-frame window scoring about 1.0;
- multi-clip mining and calibration;
- Manual, Auto and Hybrid retrieval, and the XML/CSV importers;
- out-of-vocabulary idle and skip handling;
- the MiniLM phrase-vector path keeping Algorithm 2's chunking, argmax and timing, and the optional GloVe path.


## Interactive data handoff

`scripts/start_demo.py` downloads all-MiniLM-L6-v2 (about 92 MB) into ignored `models/` and a small official BEAT BVH/TextGrid sample (four takes, about 80 MB) on first run, prepares an ignored, locally generated bank and runs the browser. Every bank clip is a playback gesture; the three seed clips also form the manual map, and disjoint association windows are mined into auto rules. The trace exposes seed or mined routes, selected IDs and the text encoder. This small simulation is distinct from the video-pose CLI contracts above; it does not establish paper-scale rule coverage or matching quality. The older `--example` server path remains an explicitly authored offline fixture. Speech is optional, and motion playback follows speech progress with short clip blends. No public recording or fitted artifact is bundled.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; mouth shapes follow a rule-based text-to-phoneme-to-viseme track timed to speech playback, an approximation rather than forced phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
