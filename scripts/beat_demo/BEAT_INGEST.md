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
python scripts/beat_demo/beat_ingest.py list   --source <path-to>/processed/beat --speakers 1,2
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
| `train_pairs.npz` | `motion [N,L,J*3]`, `ids`, `texts`, `speakers` (train role), plus `text_embeddings` **only** when `--sbert <local path or model name>` is given. The export downloads nothing itself. |
| `heldout_pairs.npz` | The same keys, for the heldout role. |
| `speaker_motion.npy` | Continuous motion (library role). |
| `transcripts.jsonl` | One record per library take: `{record_id, speaker, text, words}`. |

The extra keys (`ids`, `texts`, `confidence`, `segments_json`, `speakers`, `yaw`)
are additive. The existing loaders ignore them.

## Shared demo bank (`build_library.py`)

An unflagged run (what `start_demo.py` does) builds the **default bank**:

- **Public default.** Four named takes from three speakers (`DEFAULT_TAKES`):
  `1_wayne_0_1_1` (library), `2_scott_0_1_1` (train), `4_lawrence_0_1_1` and
  `4_lawrence_0_2_2` (wild). That is about 80 MB of BVH plus word alignments from
  Hugging Face, with a 25 MB cap per file and a 120 MB cap on the download folder.
  The result is 9 bank clips and about 85 association windows.
- **Processed default.** When `BEAT_PROCESSED_ROOT` names a local OmniMo
  collection, speakers 1–6 with two takes each are used instead (about 160
  association windows).
- **Roles.** With three or more speakers, the speaker of `1_wayne_0_1_1` is the
  `library`, so the nine reviewed windows and the cached semantic annotations stay
  valid. About 40% of the other speakers become `train` and the rest `wild`. The
  wild pool supplies the pose modes' rule text, so it gets the larger share.

Other banks are selected with flags:

```bash
python scripts/prepare_beat_demo.py --processed <path-to>/processed/beat --speakers 1,2,3 --max-takes-per-speaker 2
python scripts/prepare_beat_demo.py --raw-root <path-to>/beat_english_v0.2.1 --speakers 2,4
python scripts/prepare_beat_demo.py --takes 1_wayne_0_1_1,2_scott_0_1_1   # public download, <=25 MB per file
```

**Why several takes.** A single take gave nine bank windows and 18 association
windows. That was too little transcript text for any application line to share
vocabulary with a rule, and too little motion to train GestureCLR (held-out
cross-view top-1 equalled chance, and 8 of 9 wild rules mapped to one unit).

**Rebuilds.**

- Changing the flags rebuilds the bank.
- An unflagged run keeps a bank that was built from explicit flags; an older
  builder rebuilds it from its stored selection.
- A bank from the former single-take default is upgraded to the current default.
- A bank without build settings (assembled by hand) is kept.
- The current builder is `v4`.

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

- **Text encoders.** Local models are picked up automatically. The adapters
  download nothing; `scripts/start_demo.py` fetches the default model first (see
  [Default model download](#default-model-download)).
  - **Sentence-BERT** comes from `prepare_beat_demo.py --sbert`,
    `BEAT_SBERT_MODEL`, `SBERT_MODEL` (the variable the paper-method scripts
    read), or `<repo>/models/all-MiniLM-L6-v2`.
  - **TF-IDF fallback** is used otherwise. The response says so in
    `text_encoder` (`tfidf-fallback (...)`).
  - **Model names only.** Responses and `index.json` name the model (for
    example `sentence-bert (all-MiniLM-L6-v2)`), never its absolute path. The
    folder is resolved again from the same settings at query time.
  - **Automatic mode** keeps Algorithm 2 (five-word chunks, argmax cosine over
    the rule phrases). By default it uses all-MiniLM-L6-v2 phrase vectors through
    the package's `SentenceEncoder`, the owner-approved substitution for the
    paper's summed GloVe vectors. A chunk needs a content word in the model's
    word-piece vocabulary, so gibberish and untranslated Hangul stay
    out-of-vocabulary and idle. GloVe stays optional: `BEAT_GLOVE_PATH`,
    `GLOVE_PATH` or `<repo>/models/glove.*.txt` selects it (word vectors summed
    per phrase). Without either model it uses labelled bag-of-words vectors.
  - **Unavailable model at query time.** An index prepared with Sentence-BERT
    stores rule embeddings from that model. If the server later runs without it
    (no setting and no `models/all-MiniLM-L6-v2`, or no sentence-transformers),
    queries do not fail: the rule texts are re-encoded with a TF-IDF fallback, the
    TF-IDF idle floor applies, RIDGE's trained fallback idles, and
    `text_encoder` plus `trace.encoder_note` name the cause and the fix.
- **Idle semantics.** Each mode follows its paper.
  - **Automatic (Algorithm 2).** The best rule is played whenever a chunk shares
    vocabulary with the rule map. There is no similarity floor. Only true OOV
    chunks idle.
  - **Wild and Multilingual.** The best rule's cluster is played. The papers'
    optional low-similarity fallback idles a chunk below `min_similarity`:
    - TF-IDF: `1e-6`, so only a chunk sharing no content word idles.
    - Sentence-BERT: 0.15. Default application lines score 0.16–0.43 against
      BEAT rules, and gibberish scores 0.10–0.12.
    - A chunk with no in-vocabulary content word also idles.
  - **RIDGE.** Unchanged. Its trained fallback answers any chunk with an
    in-vocabulary word, by design. Fallback slots carry `similarity` and a
    softmax-share `confidence`.
  - **Overrides.** A request's `min_similarity` overrides the default in every
    mode.
  - **Idle slots.** An idle slot has route `idle_no_match`, confidence 0, a
    still neutral pose and a `rule_source.reason`. `no_match` is true when every
    slot idles.
- **Reporting.** `metrics.route_counts` and `text_encoder` come with every
  query.
- **Suggested queries.**
  - They are chosen at prepare time and kept only when the query retrieves at
    least one clip and no idle slot.
  - Wild and Multilingual offer up to five, one per cluster. Each must retrieve
    through its own rule (`rule_source.rule_text`).
  - The runtime adds Korean lines from `examples/beat-translations.json` only
    when they retrieve a clip.
- **Model loading.** Sentence-BERT and RIDGE models load once per process,
  behind a lock. A failed load is not cached, so concurrent first requests on a
  fresh server are safe.
- **Gesture ids.**
  - Automatic plays every bank clip: `beat_01`–`beat_09`. The three seed clips
    also form the manual map.
  - Wild and Multilingual play extracted units, with window ids
    `<take>:<start>-<end>` in 30 fps take frames.
  - RIDGE plays bank clips, plus phrase spans with ids
    `<clip>:<start>-<end>` in clip frames.
  - The playback bank is written as `outputs/beat-library/<mode>/<mode>-bank.json`.
    `/api/beat-library` lists it.

### Automatic (`automatic_text_to_gesture.core`)

- **Mining.** `mine_clips` (Algorithm 1) mines every association window with a
  padding-aware `GestureBank` built from every bank clip.
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
  - Out-of-vocabulary chunks play idle. There is no similarity floor.
  - `map=manual|auto` selects one map only.
- **Deviation from the paper.**
  - **Change.** Poses are frontal arm and hand XY. Before the cosine, the
    dataset mean pose is removed and each coordinate is divided by its dataset
    standard deviation.
  - **Why.** On raw neck-relative poses every bank gesture scored about 0.96
    against every window. With only the mean removed, the high-variance
    coordinates (raised hands) still dominated the cosine. Mined rules then
    piled onto one gesture: all 11 rules went to `beat_02` with the single-take
    bank.
  - **Effect.** The criterion is unchanged: mean frame cosine above the
    threshold, with a random pick among passing gestures. With the default bank,
    29 rules spread over all nine clips, and no clip takes more than 17%.
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
  - **Centring.** Both steps use per-modality mean-centred latents. The
    demo-budget latents are anisotropic: unit latents share a mean pairwise
    cosine of about 0.8, so a plain nearest-unit match sent most wild windows to
    one hub unit. Centring keeps the cosine-argmax criterion.
  - **Multilingual.** The Multilingual adapter does the same.
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
  - **Missing translation.** An untranslated chunk is not an error. It plays an
    explicit idle slot, and `trace.translation_note` says how to add a
    translation. The same happens when a configured MT service fails.
    Explicit routes (`english_text`, `translation_map`, the examples table) are
    unchanged.

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

These are small local simulations, not benchmark reproductions. These numbers
were measured on the public default bank (60 wild windows), with Sentence-BERT:

- **Wild.**
  - Held-out cross-view top-1 was 0.23, against 0.017 chance.
  - 10 of 21 units were matched; the largest unit share was 0.40.
- **Multilingual.**
  - Held-out cross-view top-1 was 0.18, against 0.017 chance.
  - 14 of 20 units were matched; the largest unit share was 0.28.
- **RIDGE.** The fallback's held-out top-1 was at chance.

More takes or speakers, a local Sentence-BERT, or the `paper` training presets
are needed for meaningful numbers.

## Launcher hook for paper-method preparation (phase 2)

`start_demo.py` contains a managed block, installed idempotently by the
integration script. The block runs in this order:

0. In repositories whose demo uses Sentence-BERT (modes `automatic`, `wild`,
   `multilingual`, `ridge`) it runs `scripts/beat_demo/fetch_models.py` (see
   [Default model download](#default-model-download)). The result never stops
   the launcher.
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

## Default model download

`fetch_models.py` (vendored as `scripts/beat_demo/fetch_models.py`; standard
library only) makes the small default text model available on first run:

- **Model.** `sentence-transformers/all-MiniLM-L6-v2` at a pinned revision, about
  92 MB (Apache-2.0), into the repository's ignored `models/all-MiniLM-L6-v2`.
  The weights are checked against their published size and SHA-256. The files
  go to a partial folder first, so an interrupted download leaves nothing behind.
- **Reuse.** An existing folder with `modules.json` and weights is reused.
- **Opt-out.** `start_demo.py --offline`, `PAPERREACH_OFFLINE=1` or
  `HF_HUB_OFFLINE=1` skips the download. `BEAT_SBERT_MODEL` or `SBERT_MODEL`
  (your own model) also skips it. `HF_ENDPOINT` selects a mirror.
- **Failure.** A failed download prints the reason, the fallback (labelled
  TF-IDF or bag-of-words text matching) and the retry command. The demo starts
  anyway.
- **Scope.** Only this model is automatic. Larger optional models (bert-base for
  Context-Aware `--train-starter`, Kokoro, Whisper, LLMs, the paper-named mpnet
  models in ASAP, GloVe) stay opt-in and are documented in each README.
- **Never committed.** `models/` is in every integrated repository's
  `.gitignore`.
