"""Standalone local BEAT setup and HTTP boundary for paper/application demos."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE/'beat_deps'))
import beat_methods


def paths(repo_root, mode):
    root = Path(repo_root)
    bank = root/'outputs/beat-library/bank.json'
    artifact = root/'outputs/beat-library'/mode
    return bank, artifact


def setup(repo_root, mode, *, processed=None, epochs=80, strong_rules=None, rebuild=False):
    bank, artifact = paths(repo_root, mode)
    if processed or rebuild or not bank.exists():
        sys.path.insert(0, str(Path(repo_root)/'scripts/beat_demo'))
        from build_library import build
        result = build(Path(processed) if processed else None, bank.parent/'source')
        bank.parent.mkdir(parents=True, exist_ok=True)
        bank.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    if mode == 'wearable':
        return {'bank': str(bank), 'method': 'exact text rules over three seed clips'}
    explicit_rules = bool(strong_rules)
    if mode == 'ridge' and not strong_rules:
        from beat_semantics import resolve
        annotations = json.loads((HERE/'beat-semantic-annotations.json').read_text(encoding='utf-8'))
        rules = resolve(json.loads(bank.read_text(encoding='utf-8')), annotations)
        strong_rules = bank.parent/'strong-rules.json'
        strong_rules.write_text(json.dumps(rules, indent=2), encoding='utf-8')
    cached = json.loads((artifact/'index.json').read_text(encoding='utf-8')) if (artifact/'index.json').exists() else {}
    if not cached or explicit_rules or rebuild or cached.get('origin_bank_sha256', cached.get('bank_sha256')) != hashlib.sha256(bank.read_bytes()).hexdigest():
        return beat_methods.prepare(bank, artifact, mode, epochs=epochs, strong_rules_path=strong_rules)
    return {'artifact_dir': str(artifact), 'cached': True}


def library(repo_root, mode):
    bank, artifact = paths(repo_root, mode)
    if not bank.exists() or (mode != 'wearable' and not (artifact/'index.json').exists()):
        return {'ready': False, 'message': 'Prepare the local BEAT library with scripts/prepare_beat_demo.py.'}
    data = json.loads(bank.read_text(encoding='utf-8'))
    info = json.loads((artifact/'index.json').read_text(encoding='utf-8')) if mode != 'wearable' else {}
    if mode == 'multilingual':
        refined = Path(info['bank_path'])
        if not refined.is_absolute():
            refined = artifact/refined
        data = json.loads(refined.read_text(encoding='utf-8'))
    available = info.get('playback_ids', data['base_ids'])
    clips = [{'id': c['id'], 'text': c['text'], 'duration': len(c['positions'])/data['fps'],
              'source': c['source']} for c in data['clips'] if c['id'] in available]
    examples = [c['text'] for c in (clips if mode in {'wild','multilingual'} else clips[:3])]
    if mode == 'ridge' and info.get('strong_rules'):
        examples = [info['strong_rules'][0]['phrase'], clips[4]['text'], clips[-1]['text']]
    if mode == 'automatic':
        mined = [r['text'] for r in info.get('rules', []) if r.get('route') == 'weak_pose_rule']
        examples += mined[:1]
    if len(examples) >= 3:
        examples.append('. '.join(examples[:3]))
    if mode == 'multilingual':
        translations = Path(repo_root)/'examples/beat-translations.json'
        if translations.exists():
            examples = list(json.loads(translations.read_text(encoding='utf-8')))[:3] + examples
    metrics = {'seed_pairs': 3, 'bank_clips': len(clips), 'association_windows': len(data['associations']),
               'rules': len(info.get('rules', info.get('strong_rules', []))), 'training_loss': info.get('training_loss')}
    return {'ready': True, 'mode': mode, 'clips': clips, 'suggested_queries': examples,
            'metrics': metrics, 'algorithm': info.get('algorithm', 'Exact string rules over seed gestures')}


def query_application(repo_root, mode, text, params=None):
    bank_path, artifact = paths(repo_root, mode)
    if not library(repo_root, mode)['ready']:
        raise ValueError('Local BEAT library is not prepared. Run python scripts/prepare_beat_demo.py.')
    params = {k: (v[0] if isinstance(v, list) else v) for k, v in (params or {}).items()}
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError('Supply 1–2000 characters of query text')
    if mode == 'wearable':
        bank = json.loads(bank_path.read_text(encoding='utf-8'))
        rules = [c for c in bank['clips'] if c['id'] in bank['base_ids']]
        slots = []
        triggers = [('here', rules[0]), ('this', rules[0]), ('i', rules[1]), ('we', rules[2])]
        for phrase in text.split('.'):
            phrase = phrase.strip()
            if not phrase:
                continue
            found = next((c for c in rules if c['text'].casefold() in phrase.casefold() or phrase.casefold() == c['text'].casefold()), None)
            if found is None:
                tokens = phrase.casefold().split()
                found = next((clip for trigger, clip in triggers if trigger in tokens), None)
            if found:
                slots.append({'gesture_id': found['id'], 'text': phrase, 'frames': found['positions'],
                              'route': 'exact_seed_rule', 'source': found['source']})
        if not slots:
            # The early method really has no generic fallback: do not manufacture a match.
            return {'ready': True, 'fps': bank['fps'], 'joint_order': bank['joint_names'], 'axisSigns': bank['axisSigns'],
                    'slots': [], 'trace': {'routes': [], 'unmatched': text}, 'algorithm': 'Exact seed phrase matching'}
        return {'ready': True, 'fps': bank['fps'], 'joint_order': bank['joint_names'], 'axisSigns': bank['axisSigns'],
                'slots': slots, 'trace': {'routes': ['exact_seed_rule']*len(slots)}, 'algorithm': 'Exact seed phrase matching'}
    if mode == 'multilingual':
        params['source_language'] = params.get('source_language', params.get('language', 'en'))
        translations = Path(repo_root)/'examples/beat-translations.json'
        if translations.exists():
            params.setdefault('translation_map', json.loads(translations.read_text(encoding='utf-8')))
    result = beat_methods.query(text, params, mode, artifact)
    result['ready'] = True
    if mode == 'multilingual':
        result.update(english_text=result['trace']['retrieval_text'], source_language=params['source_language'])
    return result


def serve_beat(handler, repo_root, mode):
    parsed = urlsplit(handler.path)
    if parsed.path not in {'/api/beat-library', '/api/beat-query'}:
        return False
    try:
        if parsed.path == '/api/beat-library':
            result = library(repo_root, mode)
        else:
            params = parse_qs(parsed.query)
            if handler.command == 'POST':
                size = int(handler.headers.get('Content-Length', '0'))
                if not 0 < size <= 32_000:
                    raise ValueError('Text query must be under 32 KB')
                params.update(json.loads(handler.rfile.read(size)))
            text = params.pop('text', '')
            if isinstance(text, list):
                text = text[0]
            result = query_application(repo_root, mode, text, params)
        status = 200
    except (ValueError, FileNotFoundError) as error:
        result, status = {'error': str(error)}, 400
    body = json.dumps(result, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=HERE.parent)
    parser.add_argument('--mode', choices=['wearable', 'automatic', 'wild', 'multilingual', 'ridge'], required=True)
    parser.add_argument('--processed', type=Path)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--strong-rules', type=Path)
    parser.add_argument('--query')
    parser.add_argument('--rebuild', action='store_true', help='Regenerate the local bank and refit this demo adapter')
    args = parser.parse_args()
    result = query_application(args.repo, args.mode, args.query) if args.query else setup(args.repo, args.mode, processed=args.processed,
                                                                                       epochs=args.epochs, strong_rules=args.strong_rules, rebuild=args.rebuild)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
