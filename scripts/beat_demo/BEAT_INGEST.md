# BEAT ingest and gesture-demo contracts

`beat_ingest.py` is the shared, numpy-only BEAT loader for the gesture-line paper
repositories. It is maintained in `PaperReach/tools/beat-demo/` and vendored into
each repository as `scripts/beat_demo/beat_ingest.py` by
`tools/integrate-beat-methods.py`. Do not edit the vendored copy; change the
canonical file and re-run the integration.

Data never enters Git. Recordings, prepared arrays, rules and fitted weights stay
in ignored folders (`outputs/`, `data/`).

## Sources

| Source | Layout | Notes |
|---|---|---|
| Processed OmniMo BEAT | `<root>/<speaker>/{meta.json,motion.npz}` | 30 fps, 52-joint Unity humanoid, local 6D rotations; take-local word frames. Loaded with vectorised FK. |
| Raw BEAT | `beat_english_v0.2.1/<speaker>/<take>.{bvh,TextGrid}` (a flat folder also works) | 120 Hz BVH in centimetres, Praat TextGrid `words` tier. The official files are on Hugging Face `H-Liu1997/BEAT`. |

`detect_kind()` picks the route automatically: `meta.json` means processed and
`*.bvh` means raw.

## Records

Every loader returns one plain dictionary per take:

```python
{"id": "1_wayne_0_1_1", "speaker": "1", "take": "1_wayne_0_1_1", "fps": 30.0,
 "joint_names": [...],                 # canonical names (see below)
 "positions": float32[T, J, 3],        # metres
 "words": [("the", 40, 44), ...],      # (word, start_frame, end_frame), end exclusive
 "source": {"kind": "processed"|"raw", "version": ..., "axis_signs": [...],
            "motion_url": ..., "alignment_url": ..., ...},
 "basis": "metres; Y up; +Z forward; subject left on +X (raw BEAT BVH basis)"}
```

- **Basis.** Both routes share the raw BVH basis. Processed Unity data is
  left-negative-X, so it is mirrored on X when loaded (`mirror_x=True`). The
  original basis is kept in `source.axis_signs`. Raw BEAT BVH roots sit at the
  origin, so hip height differs between the routes. Neck-centred features are
  comparable.
- **Joint names.** Joint names are canonicalised to BVH names:
  `LeftUpperArm→LeftArm`, `LeftLowerArm→LeftForeArm` and the leg equivalents.
  `UPPER_BODY` is the 11-joint contract order:
  `Hips, Neck, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand, RightShoulder, RightArm, RightForeArm, RightHand`.
- **Frame rates.** Output fps may be 30 or 15, or any positive value. An integer
  stride is exact (120→30, 30→15); any other rate is resampled linearly. Word
  frames are rescaled to match.

## Python API

| Function | Purpose |
|---|---|
| `list_takes(source, kind="auto", speakers=None, takes=None, max_takes_per_speaker=None)` | Lightweight descriptors. `speakers` accepts ids or names (`"1,2,wayne"`); `takes` accepts ids or globs (`"*_0_1_1"`). |
| `load_take(descriptor, fps=30, start_frame=0, max_frames=None, joints=None)` | Load one take. Also available directly as `load_processed(folder, take, ...)` and `load_bvh(bvh, textgrid, ...)`. |
| `iter_takes(source, ..., fps=..., joints=...)` | Lazy loader over many speakers and takes. |
| `resample(record, fps)` | Resample one record to a new frame rate. |
| `select_joints(record, names=UPPER_BODY)` | Keep only the named joints. Aliases are accepted. |
| `center(record, joint="Neck", per_frame=True, axes=(0,1,2))` and `center_positions(...)` | Neck centring, or root centring with `joint="Hips", axes=(0,2)`. |
| `motion_energy(positions, joint_names, fps)` | Mean neck-relative wrist speed in m/s. Used to reject near-static clips. |
| `assign_roles(items, spec, seed)` | Deterministic, disjoint role assignment. See [Roles](#roles). |
| `role_assignment(descriptors, spec, seed, unit="auto")` | Applies `assign_roles` by speaker when there are at least as many speakers as roles, otherwise by take. Explicit take ids force take units. |
| `parse_role_spec(["library=1,2", "train=0.5", "wild=rest", "heldout=n3"])` | Parses CLI role strings: an id list, a fraction, a count (`nK`) or `rest`. |
| `windows(record, length, stride=None, min_words=1, word_rule="midpoint", offset=0)` | Fixed windows with globally unique IDs `"<take>:<start>-<end>"`. See [Windows](#windows). |
| `project_2d(positions, yaw, pitch, focal=1.0, distance=3.0, pivot=None)` | Explicit pinhole camera. See [Camera](#camera). |
| `corrupt_2d(pose2d, seed, noise=0.02, jitter=1, dropout=0.05, fill="zero"\|"hold"\|"nan")` | OpenPose-like corruption. Returns `(pose, confidence)`. See [Camera](#camera). |
| `save_record(record, folder)`, `load_record(npz)`, `write_words_jsonl(words, path)`, `read_words_jsonl(path)` | NPZ and JSONL export. `<take>.npz` holds positions, joint names, fps, speaker, take and source JSON. `<take>.words.jsonl` holds `{"word","start_frame","end_frame"}` rows. |
| `export_automatic / export_wild / export_multilingual / export_ridge(source, output_dir, ...)` | Write each repository's existing data contract from many takes with disjoint roles. |

### Roles

`assign_roles` maps each role to one of:

- a list of items;
- a fraction;
- an integer count;
- `"rest"`.

Explicit lists are honoured first. The remaining items are shuffled with the seed
and allocated in spec order, with at least one item per role while items remain.

### Windows

- **Silence.** Windows with fewer than `min_words` words are skipped, not
  treated as errors.
- **`midpoint` rule.** Each word goes to the window that holds its midpoint, so
  adjacent windows never share a boundary word.
- **`overlap` rule.** Reproduces the legacy behaviour, where a word goes to
  every window it overlaps.

### Camera

- **`project_2d`.** The camera sits `distance` units in front of the subject on
  +Z and is orbited by `yaw` and `pitch`, given in degrees. Image x points to
  the camera's right (the subject's left), and image y points up.
  - **Scale.** Coordinates are scaled by `focal*distance/depth`, so one image
    unit equals one world unit at the pivot.
  - **Orthographic.** `distance=None` gives an orthographic projection.
- **`corrupt_2d`.** Applies OpenPose-like corruption:
  - Gaussian noise, relative to the RMS body extent;
  - temporal jitter of ±k frames;
  - joint dropout, with confidence 0.

## CLI

```bash
python scripts/beat_demo/beat_ingest.py list   --source E:/datasets/processed/beat --speakers 1,2
python scripts/beat_demo/beat_ingest.py ingest --source <root> --output data/beat-records \
       --speakers 1,2 --max-takes-per-speaker 3 --fps 15 --upper-body --center neck
python scripts/beat_demo/beat_ingest.py roles  --source <root> --role library=1 --role train=0.5 --role wild=rest
python scripts/beat_demo/beat_ingest.py export-wild --source <root> --output-dir data/beat-wild \
       --speakers 1,2,3,4,5,6 --max-takes-per-speaker 4 --fps 15 --unit-seconds 3 \
       --yaw 20 --pitch 5 --noise 0.02 --jitter 1 --dropout 0.05 --seed 0
```

Common flags:

| Flag | Meaning |
|---|---|
| `--kind auto\|processed\|raw` | Source route. |
| `--speakers`, `--takes`, `--max-takes-per-speaker` | Take selection. |
| `--fps` | Output frame rate. Default 15. |
| `--max-frames` | Quick tests. |
| `--role role=value` | Repeatable role override. |
| `--role-unit auto\|speaker\|take` | Role granularity. |
| `--seed` | Role and corruption seed. |
| `--units cm\|m` | Contract units. **The default is `cm`**, matching the existing BVH-derived contracts. |

## Repository contracts (`export-<repo>`)

All contracts share these properties:

- positions are per-frame **Neck-centred**, in `--units` (cm by default), using the `UPPER_BODY` joint order;
- windows are `--unit-seconds` long (3 s at 15 fps, so 45 frames);
- wordless windows are skipped;
- IDs take the form `<take>:<start>-<end>`, in output-fps frames;
- `joint_order.json` records joints, fps, units, cameras, corruption and the role assignment (`roles`, `takes`, `unit`, `seed`).

Every role draws on disjoint speakers, or on disjoint takes when there are too few speakers.

| Repo | Default roles | Files (keys) |
|---|---|---|
| automatic-text-to-gesture | `library`, `video` | `bank.npz` `{<window id>: [L,J,2]}`: library windows, frontal orthographic. `video.npz` `pose [T,J,2]`, `confidence [T,J]`, `words_json` (`[{"word","start_frame","end_frame"}]` in stream frames), `segments_json` (take boundaries): video-role takes concatenated, projected through the camera and corrupted. |
| wild-pose-matching | `library`, `train`, `wild` | `units.npz` `motion3d [N,L,J*3]`, `ids`, `dim2`, `texts`, `speakers` (library). `pairs.npz` `pose2d [N,L,J*2]`, `motion3d`, `ids`, `texts`, `yaw` (train; clean projection with random yaw ±30°). `wild.npz` `pose2d`, `texts`, `ids`, `confidence` (held-out wild role; camera plus corruption). |
| multilingual-gesture | `library`, `train`, `wild` | As for wild, plus `speaker_motion.npy [T,J*3]`: the continuous motion of the first library speaker, for `extract_units`. |
| ridge | `library`, `train`, `heldout` | See below. |

The RIDGE contract writes these files:

| File | Contents |
|---|---|
| `train_pairs.npz` | `motion [N,L,J*3]`, `ids`, `texts`, `speakers` (train role), plus `text_embeddings` **only** when `--sbert <local path or model name>` is given. Nothing is downloaded implicitly. |
| `heldout_pairs.npz` | The same keys, for the heldout role. |
| `speaker_motion.npy` | Continuous motion (library role). |
| `transcripts.jsonl` | One record per library take: `{record_id, speaker, text, words}`. |

The extra keys (`ids`, `texts`, `confidence`, `segments_json`, `speakers`, `yaw`)
are additive. The existing loaders ignore them.

## Shared demo bank (`build_library.py`)

The default bank still uses the single public take `1_wayne_0_1_1` and nine clips.
Its reviewed windows keep the cached semantic annotations valid. Larger banks are
selected with flags:

```bash
python scripts/prepare_beat_demo.py --processed E:/datasets/processed/beat --speakers 1,2,3 --max-takes-per-speaker 2
python scripts/prepare_beat_demo.py --raw-root D:/beat_english_v0.2.1 --speakers 2,4
python scripts/prepare_beat_demo.py --takes 1_wayne_0_1_1,2_scott_0_1_1   # public download, <=25 MB per file
```

**Scaling up.** One take per role gives small pools (about 9 bank clips and 50
association windows for speakers 1–3). Add `--speakers` or raise
`--max-takes-per-speaker` for more units, training pairs and rules.

**Rebuilds.** Changing these flags rebuilds the bank. A bank written by an older
builder is also rebuilt, from its stored selection, on the next unflagged run.
The current builder is `v3`.

**Cache key.** The adapter is refitted when its cache key changes. The key covers:

- the bank hash (which includes the streams' content hash);
- epochs and seed;
- the strong rules;
- the Sentence-BERT setting;
- `beat_methods.py`;
- the vendored paper-package sources.

The bank has the following properties:

- **Static clips.** Windows with mean wrist speed below `--min-energy`
  (default 0.08 m/s) never enter the bank.
- **Speaker roles.** With three or more takes, takes are assigned to
  `library`, `train` and `wild` roles, by speaker when possible. Bank clips
  come only from `library`, and associations are tagged `train` or `wild`.
- **Single take.** With one take, the earlier association windows are tagged
  `train` and the later ones `wild`.
- **No shared words.** Bank clips keep every overlapping word. Association
  windows drop any word that overlaps a bank window, so no boundary word is
  shared.
- **Provenance.** Each clip records speaker, take, window id, route, motion
  energy and source URLs. The bank records the takes, roles and rejected static
  windows.
- **Streams (builder v3).**
  - When roles come from separate takes, the continuous `library` and `train`
    takes are saved next to the bank in `bank-streams.npz`.
  - The positions use the same transform as the clips.
  - `bank.json` holds `streams` metadata: the takes, their words and a content
    hash.
  - A single shared take stores no streams.

## Retrieval adapters (`beat_methods.py`)

Each mode is a thin adapter over its vendored paper package (`scripts/beat_deps/`).
The adapters reuse the package's own functions. They add only data
preparation, response formatting and the idle floor.

All four modes share these rules.

- **Text encoders.**
  - **Sentence-BERT.** Used only from a local model folder, given by
    `BEAT_SBERT_MODEL` or `prepare_beat_demo.py --sbert`. Nothing is downloaded.
  - **TF-IDF fallback.** Used otherwise. The response says so in
    `text_encoder` (`tfidf-fallback (...)`).
  - **Automatic mode** uses GloVe when `BEAT_GLOVE_PATH` names a local file,
    and labelled bag-of-words vectors otherwise.
- **Idle floor.**
  - A chunk with no in-vocabulary content word plays an explicit `idle` slot.
    So does a chunk scoring below `min_similarity`: 0.2 with TF-IDF or
    bag-of-words, 0.35 with Sentence-BERT.
  - An idle slot has route `idle_no_match`, confidence 0 and a still neutral
    pose.
  - `no_match` is true when every slot idles.
- **Reporting.** `metrics.route_counts` and `text_encoder` come with every
  query.
- **Suggested queries.** These are chosen at prepare time and checked by
  running them through `query`.
- **Gesture ids.**
  - Automatic plays the three bank seed clips: `beat_01`–`beat_03`.
  - Wild and Multilingual play extracted units, with window ids
    `<take>:<start>-<end>` in 30 fps take frames.
  - RIDGE plays bank clips, plus phrase spans with ids
    `<clip>:<start>-<end>` in clip frames.
  - The playback bank is written as `outputs/beat-library/<mode>/<mode>-bank.json`.
    `/api/beat-library` lists it.

### Automatic (`automatic_text_to_gesture.core`)

- **Mining.** `mine_clips` (Algorithm 1) mines every association window with a
  padding-aware `GestureBank`.
  - The window stride equals the gesture length.
  - Phrases come from `aligned_phrase` and have at most 5 words.
  - Among the passing gestures, one is picked at random with the seed.
- **Threshold.** The default is the 80th percentile from
  `threshold_from_percentile`, rounded down to the UI's 0.01 step.
  `calibration_report` is stored. A different UI threshold or seed re-mines at
  query time.
- **Retrieval.** Retrieval uses `retrieve` (Algorithm 2) with the hybrid map:
  - **Manual map.** The seed clips' own phrases, as `ManualRule` entries
    (route `seed_rule`).
  - **Auto map.** The mined rules (route `mined_pose_rule`).
  - **Chunking.** Paper chunks of 5 words. A trailing piece under 5 words is
    dropped.
  - Out-of-vocabulary chunks play idle.
  - `map=manual|auto` selects one map only.
- **Deviation from the paper.**
  - **Change.** Poses are frontal arm and hand XY with the dataset mean pose
    removed before the cosine.
  - **Why.** On raw neck-relative poses, every bank gesture scored about 0.96
    against every window, so all rules collapsed onto one gesture.
  - **Index note.** The index records this in `threshold_rule`.
- **Metrics.** Reported metrics are `rule_usage`, `max_clip_share`,
  `pair_pass_rate` and `mean_passing_gestures`.

### Wild (`wild_pose_matching`)

- **Units.** `units.extract_units` (Algorithm 3) extracts units from the library
  stream at 15 fps.
  - The variance threshold is the 25th percentile of window variances.
  - Poses are scaled with `Normalization`.
  - Without streams, each bank window yields one unit.
- **Training.**
  - **Data.** `dense_windows` over the train stream, or train associations
    without streams. Each pair gets a random camera yaw of ±30°.
  - **Trainer.** `training.train_gestureclr`, with the demo preset, the paper's
    augmentation and a validation split.
  - **Budget.** Capped at 100 steps of batch 64, about 30–50 s on a CPU.
  - The checkpoint is written with `save_checkpoint`.
- **Rules.** `wild` associations are projected at yaw 20° and pitch 5°, then
  corrupted with noise, jitter and dropout. `build_rules` maps them to their
  nearest unit, and units are grouped with `cluster_latents`.
- **Retrieval.** `pipeline.retrieve` matches 6-word chunks. It samples a unit at
  random inside the matched cluster, with a numpy generator seeded once per
  query.

### Multilingual (`multilingual_gesture`)

- **Units.** `extract_unit_spans` (Algorithm 1) extracts units with the
  automatic elbow threshold. Units keep their natural 2–3 s length.
- **Training.** The package's own `cli train` command trains GestureCLR. Each
  sample draws one augmentation condition. Models are loaded with
  `load_gestureclr`, encoded with `encode_batches` and clustered with `bisect`.
- **Retrieval.** `multilingual_retrieve` handles the request.
  - Input over 30 words is split into sentence chunks.
  - Each chunk is translated to English.
  - Retrieval runs per 6-word chunk, with the idle floor.
  - Every slot uses `blend_frames` 5.
- **Translator.** Translation goes through a `DictTranslator` first. Its table
  is `examples/beat-translations.json` plus the request's `translation_map`
  and `english_text`.
  - **Fallback.** `BEAT_TRANSLATOR=http|local` adds the package's MT client
    behind the dictionary.
  - **Settings.** These variables configure it: `BEAT_TRANSLATOR_URL`,
    `BEAT_TRANSLATOR_MODEL`, `BEAT_TRANSLATOR_API`,
    `BEAT_TRANSLATOR_API_KEY_ENV` and `BEAT_MT_MODEL_PATH`.
  - **Missing translation.** A missing translation raises an error.

### RIDGE (`ridge_gesture`)

- **Rules.**
  - **Source.** Strong rules come from `beat_semantics` and its cached
    annotations, which carry `llm_json` or `manual_annotation` provenance.
  - **Binding.** Each rule is bound to its phrase-timed span with
    `align_phrase`, at least 20 frames long.
  - **Fallback rules.** When no cached phrase occurs in the bank, the package's
    `annotate_record` heuristic is used, with route `heuristic_rule` and
    provenance `heuristic_annotation`.
  - **Live extraction.** `beat_semantics --endpoint` uses the paper's verbatim
    prompt.
- **Retrieval.** `hybrid_retrieve` scores every 3–10-word span and accepts rule
  spans greedily by score. The remaining words go to the fallback in chunks of
  at most 6 words.
- **Fallback.**
  - **Model.** `RidgeModel` is trained by the package's `cli train` in two
    stages, `pretrain` then `finetune --init`.
  - **Data.** It trains on association windows only, so the bank is held out.
  - **Early stopping.** It uses validation loss only with at least 100 pairs.
- **Confidence.** Confidence is the softmax share of the best clip multiplied
  by vocabulary coverage. The raw cosine is reported as `similarity`.
- **Metrics.** `heldout_top1` and `heldout_chance` score bank transcript →
  bank motion. `heldout_gca_retrieved` and `heldout_gca_ground_truth` are
  computed with `GCA`.

### Small-data caveat

These are small local simulations, not benchmark reproductions. With speakers
1–3 and one take per role, the measured signals were weak:

- **Wild and Multilingual.**
  - Held-out cross-view top-1 was 0.12, against 0.04 chance.
  - Learned rules concentrated on 4 units for Wild and 7 for Multilingual.
- **RIDGE.** The fallback's held-out top-1 was at chance or below.

More takes or speakers, a local Sentence-BERT, or the `paper` training presets
are needed for meaningful numbers.

## Launcher hook for paper-method preparation (phase 2)

`start_demo.py` contains a managed block, installed idempotently by the
integration script. The block runs in this order:

1. It runs `scripts/prepare_beat_demo.py`, unless `--skip-beat` is given. A
   failure prints a clear message and the demo continues with the existing cache
   or the authored starter.
2. If `scripts/prepare_paper_method.py` exists, and `--skip-paper-method` is
   absent, it runs that script non-fatally.

The hook script prints its human-readable progress, then **one JSON object as its
last stdout line**:

```json
{"ready": true, "server_args": ["scripts/demo_server.py", "--data-dir", "outputs/paper-method",
  "--rules", "outputs/paper-method/rules.jsonl", "--clusters", "outputs/paper-method/clusters.npz"],
 "summary": {"rules": 120, "data": "BEAT speakers 1-6 via beat_ingest export-wild"}}
```

The launcher handles the result as follows:

- **Success.** On exit code 0 with `"ready": true`, the object is exported as
  the environment variable `PAPER_METHOD_RESULT`.
- **Server arguments.** If `server_args` is present, it replaces the default
  server command, for example `scripts/demo_server.py --example`. The paths are
  relative to the repository root, and `--port` is still appended.
- **Failure.** On any other outcome the default command runs unchanged. That
  includes a non-zero exit, unparsable output or `ready` false.

Keep the hook cheap when its outputs are already current: cache by an input hash,
as `beat_runtime.setup` does.
