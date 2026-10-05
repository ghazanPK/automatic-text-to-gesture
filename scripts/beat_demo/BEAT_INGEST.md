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

Changing these flags rebuilds the bank. The adapter is refitted when its cache key
changes. The cache key covers:

- the bank hash;
- epochs and seed;
- the strong rules;
- `beat_methods.py`;
- the paper-package sources.

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

## Retrieval adapters (`beat_methods.py`)

All four modes share these rules:

- **Similarity floor and idle.** A chunk that shares no content word with any
  rule, or scores below `min_similarity` (default 0.2), plays an explicit
  `idle` slot (route `idle_no_match`, confidence 0, a still neutral pose).
  `no_match` is true when every slot idles. RIDGE idles when the chunk has no
  in-vocabulary content word.
- **Route counts.** `metrics.route_counts` is reported with every query.

### Automatic

- **Mining.** Each association window is split into phrases of at most 5
  words, using word timing. Each phrase's pose span is compared with the three
  bank gestures by padding-aware mean frame cosine on neck-relative arm and hand
  XY, after removing the mean pose.
- **Rules.** Every gesture at or above the threshold is a candidate, and one is
  picked at random with the seed.
- **Threshold.** The default is the 80% quantile of phrase–gesture cosines
  (`default_threshold`). A UI threshold or seed re-mines at query time.
- **Chunks.** Queries use 5-word chunks.
- **Metrics.** Reported metrics are `rule_usage`, `max_clip_share` and
  `pair_pass_rate`.

### Wild and Multilingual

- **Training.** The encoders train on `train` associations, with a random
  camera yaw of ±30° per pair. Multilingual also uses the paper's `augment_2d`.
- **Mining.** Rules are mined from `wild` associations, projected at yaw 20°
  and pitch 5° with noise, jitter and dropout.
- **Suggested queries.** Suggestions are learned-rule phrases whose first
  route is `learned_pose_rule`, plus one bank transcript.
- **Metrics.** Reported metrics are `rule_routes`, `learned_rule_usage`,
  `cluster_sizes`, and `heldout_cross_view_top1` with `heldout_chance`.

### RIDGE

- **Training.** The fallback trains on association windows only. The bank
  clips it retrieves are held out.
- **Confidence.** Confidence is the softmax share of the best clip multiplied
  by vocabulary coverage. The raw cosine is reported as `similarity`.
- **Metrics.** Reported metrics are `heldout_top1` and `heldout_chance`, where
  bank transcript → bank motion is evaluated without training on it.

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
