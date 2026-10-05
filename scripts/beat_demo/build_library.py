"""Build a small natural co-speech bank and a separate association pool locally.

Dataset recordings stay in ignored outputs/data directories. The default public
route downloads one BVH and its word alignment, not the full BEAT archive;
``--takes``/``--speakers`` select more takes from a processed OmniMo collection,
a local raw ``beat_english_v0.2.1`` folder, or named public takes.

Bank clips and association windows are disjoint 2.5 s windows. Bank clips keep
every word overlapping them; association windows drop any word that overlaps a
bank window and otherwise own a word only when they hold its midpoint, so no
boundary word is shared between the bank and the associations. Near-static
windows (mean wrist speed below ``--min-energy`` m/s) never enter the bank.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import beat_ingest as ingest

TAKE = '1_wayne_0_1_1'
BASE = ingest.HF_BASE + '1/'
ROOT_URL = ingest.HF_BASE
WINDOW = 75
OFFSET = 30
REVIEWED_STARTS = [1080, 480, 1005, 1905, 1680, 1755, 780, 1830, 1230]
BUILDER = 'v3'  # v3: continuous library/train streams in <bank>-streams.npz


def fetch(name, folder, cap=25_000_000, url=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        try:
            with urllib.request.urlopen(url or BASE + name, timeout=60) as response, part.open('wb') as output:
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > cap:
                        raise ValueError('Named source exceeds the download size cap')
                    output.write(block)
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    if path.stat().st_size > cap:
        raise ValueError('Cached source exceeds size cap')
    return path


def textgrid_words(path):
    """Legacy helper: word dicts with 30 fps frame_start/frame_end."""
    return [{'text': w, 'frame_start': s, 'frame_end': e} for w, s, e in ingest.parse_textgrid(path, 30)]


def _names(value):
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(',')
    return [str(v).strip() for v in value if str(v).strip()]


def load_records(processed=None, source=Path('outputs/beat-library/source'), *, speakers=None, takes=None,
                 max_takes_per_speaker=1, raw_root=None, max_frames=3000):
    """Load the selected takes (30 fps, all joints, metres, canonical basis)."""
    speakers, takes = _names(speakers), _names(takes)
    if not speakers and not takes:
        takes = [TAKE]
    options = {'fps': 30, 'max_frames': max_frames}
    if processed or raw_root:
        root = Path(processed or raw_root)
        rows = ingest.list_takes(root, kind='processed' if processed else 'raw', speakers=speakers or None,
                                 takes=takes or None,
                                 max_takes_per_speaker=None if takes else max_takes_per_speaker)
        if not rows:
            raise ValueError(f'No BEAT takes matched in {root}')
        return [ingest.load_take(row, **options) for row in rows]
    if not takes or any(any(ch in t for ch in '*?[') for t in takes):
        raise ValueError('The public download route needs explicit take ids, e.g. --takes 1_wayne_0_1_1,2_scott_0_1_1; '
                         'use --processed or --raw-root to select by speaker')
    records = []
    for take in takes:
        speaker = take.split('_')[0]
        bvh = fetch(take + '.bvh', source, url=ROOT_URL + f'{speaker}/{take}.bvh')
        grid = fetch(take + '.TextGrid', source, 500_000, url=ROOT_URL + f'{speaker}/{take}.TextGrid')
        records.append(ingest.load_bvh(bvh, grid, fps=30, max_frames=max_frames, speaker=speaker))
    return records


def _roles(records, seed):
    """Library takes feed the bank; other takes feed train/wild association pools."""
    descriptors = [{'speaker': r['speaker'], 'take': r['take']} for r in records]
    if len(records) == 1:
        return {records[0]['take']: 'shared'}, 'single take'
    if len(records) == 2:
        return {records[0]['take']: 'library', records[1]['take']: 'associations'}, 'take'
    mapping, unit = ingest.role_assignment(descriptors, {'library': 1 / 3, 'train': 1 / 3, 'wild': 'rest'}, seed)
    return mapping, unit


def _descriptor(clip, names):
    hands = [names.index(n) for n in ('LeftHand', 'RightHand')]
    return np.concatenate([clip[:, hands].mean(0).ravel(), clip[:, hands].std(0).ravel(),
                           np.diff(clip[:, hands], axis=0).std(0).ravel() * 8])


def disjoint(clips, associations):
    """True when no association window overlaps a bank window or uses a word overlapping one."""
    spans = {}
    for c in clips:
        spans.setdefault(c['source']['take'], []).append((c['source']['start_frame'], c['source']['end_frame']))
    for a in associations:
        take, start = a['source']['take'], a['source']['start_frame']
        for lo, hi in spans.get(take, []):
            if a['source']['end_frame'] > lo and start < hi:
                return False
            if any(start + w['end_frame'] > lo and start + w['start_frame'] < hi for w in a['words']):
                return False
    return True


def build(processed=None, source=Path('outputs/beat-library/source'), count=9, *, speakers=None, takes=None,
          max_takes_per_speaker=1, raw_root=None, min_energy=0.08, max_associations=160, seed=0,
          max_frames=3000):
    if not 3 <= count <= 12:
        raise ValueError('The small demo supports 3–12 bank entries')
    records = load_records(processed, source, speakers=speakers, takes=takes,
                           max_takes_per_speaker=max_takes_per_speaker, raw_root=raw_root, max_frames=max_frames)
    names = records[0]['joint_names']
    if any(r['joint_names'] != names for r in records):
        raise ValueError('Selected takes use different skeletons; build processed and raw banks separately')
    roles, role_unit = _roles(records, seed)
    candidates = []
    streams = []
    for record in records:
        positions = ingest.center_positions(record['positions'], names, 'Neck')
        positions[:, :, 1] += 1.35
        centred = dict(record, positions=positions)
        # Continuous library/train takes let the pose demos run the papers' unit extraction
        # (Algorithm 1/3) and dense training windows; a single shared take stores none.
        if role_unit != 'single take' and roles.get(record['take']) in ('library', 'train'):
            streams.append({'take': record['take'], 'speaker': record['speaker'], 'role': roles[record['take']],
                            'start_frame': int(record['source'].get('take_frame_start', 0)),
                            'frames': len(positions), 'route': record['source']['kind'],
                            'motion_url': record['source']['motion_url'],
                            'alignment_url': record['source']['alignment_url'],
                            'words': [{'text': w, 'start_frame': int(s), 'end_frame': int(e)}
                                      for w, s, e in record['words']],
                            'positions': np.round(positions, 4).astype(np.float32)})
        for window in ingest.windows(centred, WINDOW, offset=OFFSET, min_words=3, word_rule='overlap'):
            window['energy'] = ingest.motion_energy(window['positions'], names, 30)
            window['descriptor'] = _descriptor(window['positions'], names)
            window['record'] = record
            window['role'] = roles.get(record['take'])
            candidates.append(window)
    shared = role_unit == 'single take'
    library = [i for i, c in enumerate(candidates) if shared or c['role'] == 'library']
    eligible = [i for i in library if candidates[i]['energy'] >= min_energy]
    if len(eligible) < count:
        raise ValueError(f'Only {len(eligible)} bank windows pass the motion-energy floor {min_energy}; '
                         'lower --min-energy or add takes')
    feature = np.stack([candidates[i]['descriptor'] for i in eligible])
    feature = (feature - feature.mean(0)) / np.maximum(feature.std(0), .01)
    # Stable selections keep the small reviewed semantic-annotation cache usable
    # with both the raw public take and its processed humanoid counterpart;
    # reviewed windows below the motion-energy floor are replaced greedily.
    lookup = {(c['take'], c['start']): i for i, c in enumerate(candidates)}
    picked = []
    selection = 'greedy hand-descriptor diversity over library windows above the motion-energy floor'
    if count == 9 and len(records) == 1 and records[0]['take'] == TAKE:
        position = {index: k for k, index in enumerate(eligible)}
        picked = [position[lookup[(TAKE, s)]] for s in REVIEWED_STARTS
                  if lookup.get((TAKE, s)) in position]
        selection = (f'{len(picked)} reviewed diverse hand windows for the named take above the motion-energy '
                     'floor, completed by greedy diversity')
    if not picked:
        picked = [int(np.argmax(np.linalg.norm(feature[:, 6:12], axis=1)))]
    while len(picked) < count:
        distances = np.min(np.linalg.norm(feature[:, None] - feature[picked][None], axis=-1), axis=1)
        distances[picked] = -1
        picked.append(int(distances.argmax()))
    selected = [eligible[i] for i in picked]
    chosen = set(selected)
    # Association windows never reuse a word that overlaps a bank window, and
    # among themselves each word belongs to the window holding its midpoint.
    bank_spans = {}
    for i in selected:
        bank_spans.setdefault(candidates[i]['take'], []).append((candidates[i]['start'], candidates[i]['end']))
    for index, c in enumerate(candidates):
        if index in chosen:
            continue
        spans = bank_spans.get(c['take'], [])
        kept = []
        for word, s, e in c['words']:
            a, b = c['start'] + s, c['start'] + e
            if any(b > lo and a < hi for lo, hi in spans) or not c['start'] <= (a + b) / 2 < c['end']:
                continue
            kept.append((word, s, e))
        c['words'], c['text'] = kept, ' '.join(w for w, _, _ in kept)
    pool = [i for i, c in enumerate(candidates) if i not in chosen and (shared or c['role'] != 'library')
            and len(c['words']) >= 3]
    if len(pool) > max_associations:
        pool = [pool[int(k)] for k in np.linspace(0, len(pool) - 1, max_associations).round()]
    if len(pool) < 4:
        raise ValueError('Insufficient aligned motion windows for bank and disjoint associations')
    if shared:
        # One take: earlier association windows train, later ones are the held-out "wild" pool.
        for position, i in enumerate(pool):
            candidates[i]['role'] = 'train' if position < len(pool) // 2 else 'wild'
        for i in selected:
            candidates[i]['role'] = 'library'
    elif role_unit == 'take' and len(records) == 2:
        for position, i in enumerate(pool):
            candidates[i]['role'] = 'train' if position < len(pool) // 2 else 'wild'

    def entry(c, identifier):
        record = c['record']
        start = int(record['source'].get('take_frame_start', 0)) + c['start']
        return {'id': identifier, 'text': c['text'], 'role': c['role'],
                'positions': np.round(c['positions'], 4).tolist(),
                'words': [{'text': w, 'start_frame': s, 'end_frame': e} for w, s, e in c['words']],
                'source': {'dataset': 'BEAT', 'speaker': record['speaker'], 'take': record['take'],
                           'window_id': ingest.window_id(record['take'], start, start + WINDOW),
                           'start_frame': start, 'end_frame': start + WINDOW, 'route': record['source']['kind'],
                           'motion_energy': round(c['energy'], 4),
                           'motion_url': record['source']['motion_url'],
                           'alignment_url': record['source']['alignment_url']}}
    clips = [entry(candidates[i], f'beat_{j + 1:02d}') for j, i in enumerate(selected)]
    associations = [entry(candidates[i], f'paired_{k:03d}') for k, i in enumerate(pool)]
    if not disjoint(clips, associations):
        raise AssertionError('Bank and association windows must not share frames or words')
    rejected = sum(1 for i in library if candidates[i]['energy'] < min_energy)
    takes_used = [{'speaker': r['speaker'], 'take': r['take'], 'role': roles.get(r['take']),
                   'route': r['source']['kind'], 'version': r['source'].get('version'),
                   'frames': len(r['positions'])} for r in records]
    return {'schema': 'paperreach.beat-gesture-bank.v2', 'fps': 30, 'joint_names': names,
            'axisSigns': [1, 1, 1], 'clips': clips, 'associations': associations,
            'seed_count': 3, 'base_ids': [c['id'] for c in clips[:3]],
            'selection': selection + (f'; disjoint {WINDOW}-frame windows; association windows exclude words '
                                      'overlapping bank windows and own words by midpoint'),
            'provenance': {'takes': takes_used, 'role_unit': role_unit, 'basis': ingest.BASIS,
                           'min_energy': min_energy, 'rejected_static_windows': rejected,
                           'candidate_windows': len(candidates)},
            'limits': ('Small local simulation, not benchmark evaluation. Aligned text is observed '
                       'association, not a semantic guarantee.'),
            '_streams': streams}


def save(bank, output):
    """Write bank.json; continuous streams (private ``_streams`` key) go to a compact sibling NPZ."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bank = dict(bank)
    streams = bank.pop('_streams', None) or []
    npz = output.with_name(output.stem + '-streams.npz')
    if streams:
        digest = hashlib.sha256()
        arrays, takes = {}, []
        for i, stream in enumerate(streams):
            key = f's{i}'
            arrays[key] = np.asarray(stream['positions'], np.float32)
            digest.update(arrays[key].tobytes())
            takes.append({k: v for k, v in stream.items() if k != 'positions'} | {'key': key})
        np.savez_compressed(npz, **arrays)
        bank['streams'] = {'file': npz.name, 'content_sha256': digest.hexdigest(), 'takes': takes}
    else:
        npz.unlink(missing_ok=True)
        bank.pop('streams', None)
    output.write_text(json.dumps(bank, separators=(',', ':')), encoding='utf-8')
    return output


def settings(processed=None, count=9, speakers=None, takes=None, max_takes_per_speaker=1, raw_root=None,
             min_energy=0.08, max_associations=160):
    return {'processed': str(processed) if processed else None, 'raw_root': str(raw_root) if raw_root else None,
            'count': count, 'speakers': _names(speakers), 'takes': _names(takes),
            'max_takes_per_speaker': max_takes_per_speaker, 'min_energy': min_energy,
            'max_associations': max_associations, 'builder': BUILDER}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, help='Processed OmniMo BEAT root (<speaker>/meta.json)')
    parser.add_argument('--raw-root', type=Path, help='Local raw beat_english_v0.2.1 folder (<speaker>/<take>.bvh)')
    parser.add_argument('--speakers', help='Comma list of speaker ids/names (processed or raw-root)')
    parser.add_argument('--takes', help='Comma list of take ids (glob patterns with --processed/--raw-root)')
    parser.add_argument('--max-takes-per-speaker', type=int, default=1)
    parser.add_argument('--source', type=Path, default=Path('outputs/beat-library/source'))
    parser.add_argument('--output', type=Path, default=Path('outputs/beat-library/bank.json'))
    parser.add_argument('--count', type=int, default=9)
    parser.add_argument('--min-energy', type=float, default=0.08, help='Bank motion-energy floor (mean wrist speed, m/s)')
    parser.add_argument('--max-associations', type=int, default=160)
    args = parser.parse_args()
    bank = build(args.processed, args.source, args.count, speakers=args.speakers, takes=args.takes,
                 max_takes_per_speaker=args.max_takes_per_speaker, raw_root=args.raw_root,
                 min_energy=args.min_energy, max_associations=args.max_associations)
    bank['build_settings'] = settings(args.processed, args.count, args.speakers, args.takes,
                                      args.max_takes_per_speaker, args.raw_root, args.min_energy, args.max_associations)
    save(bank, args.output)
    print(json.dumps({'output': str(args.output), 'bank': len(bank['clips']), 'association_windows': len(bank['associations']),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
