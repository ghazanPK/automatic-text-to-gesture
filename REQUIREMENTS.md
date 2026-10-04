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

The browser queries the existing retrieval implementation and renders the selected motion frames with a pinned local Three.js module. Its immediate example mode is author-created motion plus explicitly illustrative, untrained vectors. Production mode accepts the documented public BVH/timed-transcript preparation outputs, real local encoder assets and trained checkpoints as appropriate. The preparation adapter preserves motion/transcript alignment and declares skeleton/FPS assumptions; it does not fabricate annotations or evaluation results. Speech is optional and replaceable (Kokoro-82M English/faster-whisper small CPU INT8, with browser voice/typed-input alternatives). Verification must cover clip serialization and algorithm routing, with model quality evaluation deferred to user-prepared public data.
