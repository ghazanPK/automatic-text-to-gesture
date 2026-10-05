"""Build a small natural co-speech bank and a separate association pool locally.

Dataset recordings stay in ignored outputs/data directories. The public route
downloads one BVH and its word alignment, not the full BEAT archive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import urllib.request
import numpy as np
from prepare import emit_clip
from prepare_bvh import emit as emit_bvh

TAKE = '1_wayne_0_1_1'
BASE = 'https://huggingface.co/datasets/H-Liu1997/BEAT/resolve/main/beat_english_v0.2.1/beat_english_v0.2.1/1/'


def fetch(name, folder, cap=25_000_000):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        try:
            with urllib.request.urlopen(BASE + name, timeout=60) as response, part.open('wb') as output:
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
    text = path.read_text(encoding='utf-8')
    word_tier = next((x for x in re.split(r'\bitem \[\d+\]:', text) if 'name = "words"' in x), None)
    if word_tier is None:
        raise ValueError('BEAT TextGrid has no words tier')
    values = re.findall(r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "((?:[^"]|"")*)"', word_tier)
    return [{'text': word.replace('""', '"'), 'frame_start': round(float(start)*30),
             'frame_end': round(float(end)*30)} for start, end, word in values if word.strip()]


def load_take(processed, source):
    with tempfile.TemporaryDirectory() as temporary:
        file = Path(temporary)/'motion.json'
        if processed:
            metadata = json.loads((processed/'1/meta.json').read_text(encoding='utf-8'))
            take = next(t for t in metadata['takes'] if t['sequence_id'] == TAKE)
            emit_clip(processed, '1', TAKE, file, take['frame_end']-take['frame_start'])
            words = take['words']
        else:
            bvh = fetch(TAKE+'.bvh', source)
            grid = fetch(TAKE+'.TextGrid', source, 500_000)
            emit_bvh(bvh, file, max_frames=3000)
            words = textgrid_words(grid)
        motion = json.loads(file.read_text(encoding='utf-8'))
    return motion, words


def build(processed=None, source=Path('outputs/beat-library/source'), count=9):
    if not 3 <= count <= 12:
        raise ValueError('The small demo supports 3–12 bank entries')
    motion, words = load_take(processed, source)
    positions = np.asarray([f['positions'] for f in motion['frames']], np.float32)
    names = motion['source']['jointNames']
    neck = names.index('Neck')
    positions -= positions[:, neck:neck+1]
    positions[:, :, 1] += 1.35
    hands = [names.index(n) for n in ('LeftHand', 'RightHand')]
    # Disjoint 2.5s candidates, with no bank/association overlap.
    candidates = []
    for start in range(30, len(positions)-75, 75):
        text = ' '.join(w.get('text', w.get('word', '')) for w in words
                        if w['frame_end'] > start and w['frame_start'] < start+75)
        if len(text.split()) < 3:
            continue
        clip = positions[start:start+75]
        descriptor = np.concatenate([clip[:, hands].mean(0).ravel(), clip[:, hands].std(0).ravel(),
                                     np.diff(clip[:, hands], axis=0).std(0).ravel()*8])
        candidates.append({'text': text, 'positions': clip, 'descriptor': descriptor, 'start': start})
    if len(candidates) < count+4:
        raise ValueError('Insufficient aligned motion windows for bank and disjoint associations')
    feature = np.stack([c['descriptor'] for c in candidates])
    feature = (feature-feature.mean(0))/np.maximum(feature.std(0), .01)
    selected = [int(np.argmax(np.linalg.norm(feature[:, 6:12], axis=1)))]
    while len(selected) < count:
        distances = np.min(np.linalg.norm(feature[:, None]-feature[selected][None], axis=-1), axis=1)
        distances[selected] = -1
        selected.append(int(distances.argmax()))
    # Stable selections make the small reviewed semantic annotation cache usable
    # with both the raw public take and its processed humanoid counterpart.
    reviewed_starts = [1080, 480, 1005, 1905, 1680, 1755, 780, 1830, 1230]
    lookup = {c['start']: i for i, c in enumerate(candidates)}
    if count == 9 and all(start in lookup for start in reviewed_starts):
        selected = [lookup[start] for start in reviewed_starts]
    def entry(c, identifier):
        return {'id': identifier, 'text': c['text'], 'positions': np.round(c['positions'], 5).tolist(),
                'words': [{'text': w.get('text', w.get('word', '')), 'start_frame': max(0, w['frame_start']-c['start']),
                           'end_frame': min(75, w['frame_end']-c['start'])} for w in words
                          if w['frame_end'] > c['start'] and w['frame_start'] < c['start']+75],
                'source': {'dataset': 'BEAT', 'take': TAKE, 'start_frame': c['start'], 'end_frame': c['start']+75,
                           'motion_url': BASE+TAKE+'.bvh', 'alignment_url': BASE+TAKE+'.TextGrid'}}
    clips = [entry(candidates[i], f'beat_{j+1:02d}') for j, i in enumerate(selected)]
    associations = [entry(c, f'paired_{i:02d}') for i, c in enumerate(candidates) if i not in selected]
    # Processed Unity is left-negative-X; the raw BVH is left-positive-X.
    signs = motion['source'].get('axisSigns', [1, 1, 1])
    return {'schema': 'paperreach.beat-gesture-bank.v1', 'fps': 30, 'joint_names': names,
            'axisSigns': signs, 'clips': clips, 'associations': associations,
            'seed_count': 3, 'base_ids': [c['id'] for c in clips[:3]],
            'selection': 'reviewed diverse hand windows for named take; greedy diversity for other counts; disjoint 75-frame windows',
            'limits': 'Small same-speaker simulation, not benchmark evaluation. Aligned text is observed association, not a semantic guarantee.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, help='Existing processed BEAT folder; otherwise fetch the named public take')
    parser.add_argument('--source', type=Path, default=Path('outputs/beat-library/source'))
    parser.add_argument('--output', type=Path, default=Path('outputs/beat-library/bank.json'))
    parser.add_argument('--count', type=int, default=9)
    args = parser.parse_args()
    bank = build(args.processed, args.source, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, separators=(',', ':')), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'bank': len(bank['clips']), 'association_windows': len(bank['associations']),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
