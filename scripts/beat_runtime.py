"""Standalone local BEAT setup and HTTP boundary for paper/application demos."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE/'beat_deps'))
import beat_methods

BUILD_DEFAULTS = {'processed': None, 'raw_root': None, 'speakers': None, 'takes': None,
                  'max_takes_per_speaker': 1, 'count': 9, 'min_energy': 0.08}
# With a local processed OmniMo BEAT collection (BEAT_PROCESSED_ROOT) the unflagged default bank uses more
# speakers and takes: speaker 1 feeds the bank, the others become disjoint train/wild association pools.
PROCESSED_DEFAULT = {'speakers': '1,2,3,4,5,6', 'max_takes_per_speaker': 2}
ENV_PROCESSED = 'BEAT_PROCESSED_ROOT'


def default_selection():
    """Bank selection for an unflagged run: a larger processed selection when BEAT_PROCESSED_ROOT names a
    local OmniMo collection, otherwise the multi-take public default of build_library (DEFAULT_TAKES)."""
    options = dict(BUILD_DEFAULTS)
    root = os.environ.get(ENV_PROCESSED)
    if root and Path(root).is_dir() and any(Path(root).glob('*/meta.json')):
        options.update(processed=root, **PROCESSED_DEFAULT)
    return options


def _legacy_default(settings):
    """A bank written by an older builder from the former single-take public default (no selection flags)."""
    return bool(settings) and not settings.get('processed') and not settings.get('raw_root') \
        and not settings.get('speakers') and settings.get('takes') in ([], ['1_wayne_0_1_1'], None) \
        and settings.get('count', 9) == 9 and settings.get('min_energy', 0.08) == 0.08


def paths(repo_root, mode):
    root = Path(repo_root)
    bank = root/'outputs/beat-library/bank.json'
    artifact = root/'outputs/beat-library'/mode
    return bank, artifact


def _build_module():
    for folder in (HERE.parent/'beat-demo', HERE/'beat_demo'):  # tools/ checkout, then vendored scripts/
        if folder.is_dir() and str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    import build_library
    return build_library


def _build_settings(options):
    return _build_module().settings(options['processed'], options['count'], options['speakers'], options['takes'],
                                    options['max_takes_per_speaker'], options['raw_root'], options['min_energy'])


def _read_settings(bank):
    try:
        return json.loads(bank.read_text(encoding='utf-8')).get('build_settings')
    except (OSError, ValueError):
        return None


def setup(repo_root, mode, *, processed=None, raw_root=None, speakers=None, takes=None, max_takes_per_speaker=1,
          count=9, min_energy=0.08, epochs=80, strong_rules=None, rebuild=False, seed=7, sbert=None):
    """Build (or reuse) the local bank, then prepare (or reuse) this mode's adapter.

    Without selection flags the default bank is used (``default_selection``: the
    multi-take public default, or a larger processed selection when
    BEAT_PROCESSED_ROOT is set); a bank built from explicit flags keeps its
    selection (rebuilt by a newer builder), a bank from the former single-take
    default is upgraded, and a hand-built bank without build settings is kept.
    Selection flags that differ from the cached bank rebuild it. The adapter is refit when
    its cache key (bank hash, epochs, seed, strong rules, Sentence-BERT setting,
    adapter and paper-package code) changes.
    """
    bank, artifact = paths(repo_root, mode)
    options = {'processed': processed, 'raw_root': raw_root, 'speakers': speakers, 'takes': takes,
               'max_takes_per_speaker': max_takes_per_speaker, 'count': count, 'min_energy': min_energy}
    flagged = any(options[k] != v for k, v in BUILD_DEFAULTS.items())
    builder = _build_module()
    current = _read_settings(bank) if bank.exists() else None
    default = False
    if not flagged:
        if current and not current.get('default') and not _legacy_default(current):
            # A bank built from explicit flags: keep that selection (rebuilt by a newer builder if needed).
            options.update({k: current.get(k, v) for k, v in BUILD_DEFAULTS.items()})
            options['speakers'] = ','.join(options['speakers']) if isinstance(options['speakers'], list) else options['speakers']
            options['takes'] = ','.join(options['takes']) if isinstance(options['takes'], list) else options['takes']
            options = {k: (v or None) if k in {'speakers', 'takes', 'processed', 'raw_root'} else v for k, v in options.items()}
        else:
            # Default selection (public multi-take, or processed when BEAT_PROCESSED_ROOT is set); a bank from
            # the former single-take default is upgraded.
            options, default = default_selection(), True
    wanted = dict(_build_settings(options), default=default)
    # A bank without build settings was assembled by hand or by another tool: an unflagged run keeps it.
    stale = current != wanted and not (not flagged and bank.exists() and current is None)
    if rebuild or not bank.exists() or stale:
        result = builder.build(Path(options['processed']) if options['processed'] else None, bank.parent/'source',
                               options['count'], speakers=options['speakers'], takes=options['takes'],
                               max_takes_per_speaker=options['max_takes_per_speaker'],
                               raw_root=Path(options['raw_root']) if options['raw_root'] else None,
                               min_energy=options['min_energy'])
        result['build_settings'] = wanted
        builder.save(result, bank)
    if mode == 'wearable':
        return {'bank': str(bank), 'method': 'exact text rules over three seed clips'}
    if mode == 'ridge' and not strong_rules:
        from beat_semantics import resolve
        annotations = json.loads((HERE/'beat-semantic-annotations.json').read_text(encoding='utf-8'))
        rules = resolve(json.loads(bank.read_text(encoding='utf-8')), annotations)
        strong_rules = bank.parent/'strong-rules.json'
        text = json.dumps(rules, indent=2)
        if not strong_rules.exists() or strong_rules.read_text(encoding='utf-8') != text:
            strong_rules.write_text(text, encoding='utf-8')
    key = beat_methods.cache_key(mode, bank, epochs=epochs, seed=seed, strong_rules_path=strong_rules, sbert=sbert)
    try:
        cached = json.loads((artifact/'index.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cached = {}
    if rebuild or cached.get('cache_key') != key:
        return beat_methods.prepare(bank, artifact, mode, epochs=epochs, seed=seed, strong_rules_path=strong_rules,
                                    sbert=sbert)
    return {'artifact_dir': str(artifact), 'cached': True, 'cache_key': key}


def _ready(repo_root, mode):
    bank, artifact = paths(repo_root, mode)
    return bank.exists() and (mode == 'wearable' or (artifact/'index.json').exists())


_VERIFIED: dict = {}


def _verified_translations(repo_root, mode):
    """Source-language lines from examples/beat-translations.json that retrieve at least one recorded clip
    and no idle slot with this repository's prepared adapter (checked once per index and table version)."""
    translations = Path(repo_root)/'examples/beat-translations.json'
    index = paths(repo_root, mode)[1]/'index.json'
    if not translations.exists() or not index.exists():
        return []
    key = (str(translations.resolve()), translations.stat().st_mtime_ns, index.stat().st_mtime_ns)
    if key not in _VERIFIED:
        table = json.loads(translations.read_text(encoding='utf-8'))
        good = []
        for text in table:
            if not any('가' <= c <= '힣' for c in text):
                continue  # these suggestions demonstrate the Korean translation route
            try:
                result = query_application(repo_root, mode, text, {'source_language': 'ko'})
            except ValueError:
                continue
            if result['slots'] and all(s['gesture_id'] != beat_methods.IDLE_ID for s in result['slots']):
                good.append(text)
        _VERIFIED.clear()
        _VERIFIED[key] = good
    return list(_VERIFIED[key])


def library(repo_root, mode):
    bank, artifact = paths(repo_root, mode)
    if not _ready(repo_root, mode):
        return {'ready': False, 'message': 'Prepare the local BEAT library with scripts/prepare_beat_demo.py.'}
    data, _ = beat_methods._read_json(bank)
    info, _ = beat_methods._read_json(artifact/'index.json') if mode != 'wearable' else ({}, None)
    playback = data
    if info.get('bank_path'):
        # Pose modes play extracted units and RIDGE adds phrase-timed spans: list the playback bank.
        refined = Path(info['bank_path'])
        playback, _ = beat_methods._read_json(refined if refined.is_absolute() else artifact/refined)
    available = set(info.get('playback_ids', data['base_ids']))
    clips = [{'id': c['id'], 'text': c.get('text', ''), 'duration': len(c['positions'])/playback['fps'],
              'source': c.get('source', {})} for c in playback['clips'] if c['id'] in available]
    if mode == 'wearable':
        examples = [c['text'] for c in clips[:3]]
    else:
        examples = list(info.get('suggested_queries') or [c['text'] for c in clips[:3] if c['text']])
    if mode == 'multilingual':
        korean = _verified_translations(repo_root, mode)
        examples = korean[:1] + examples + korean[1:3]
    # playable_clips: what this mode can play; bank_clips: windows in the local bank (they differ when a mode
    # plays a subset, e.g. the wearable seed clips, or derived units/spans).
    metrics = {'seed_pairs': 3, 'playable_clips': len(clips), 'bank_clips': len(data['clips']),
               'association_windows': len(data['associations']),
               'rules': len(info.get('rules', info.get('strong_rules', []))), 'training_loss': info.get('training_loss')}
    metrics.update({k: v for k, v in info.get('metrics', {}).items() if k != 'learned_rule_usage'})
    if mode == 'wearable':
        metrics['rules'] = 3
    result = {'ready': True, 'mode': mode, 'clips': clips, 'suggested_queries': examples,
              'metrics': metrics, 'algorithm': info.get('algorithm', 'Exact string rules over seed gestures'),
              'provenance': data.get('provenance')}
    if mode != 'wearable':
        result['text_encoder'] = beat_methods.encoder_label(info)
    if mode == 'automatic':
        result['default_threshold'] = info.get('default_threshold')
        result['threshold_rule'] = info.get('threshold_rule')
    return result


def query_application(repo_root, mode, text, params=None):
    bank_path, artifact = paths(repo_root, mode)
    if not _ready(repo_root, mode):
        raise ValueError('Local BEAT library is not prepared. Run python scripts/prepare_beat_demo.py.')
    params = {k: (v[0] if isinstance(v, list) and v else v) for k, v in (params or {}).items()}
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
        base = {'ready': True, 'fps': bank['fps'], 'joint_order': bank['joint_names'],
                'axisSigns': bank.get('axisSigns', [1, 1, 1]), 'algorithm': 'Exact seed phrase matching'}
        if not slots:
            # The early method really has no generic fallback: do not manufacture a match.
            return {**base, 'slots': [], 'no_match': True,
                    'trace': {'routes': ['idle_no_match'], 'unmatched': text}}
        return {**base, 'slots': slots, 'no_match': False, 'trace': {'routes': ['exact_seed_rule']*len(slots)}}
    if mode == 'multilingual':
        # source_language (or language) selects the route; a caller-supplied
        # english_text is an explicit translation for this one line.
        # Callers that forward only the text (e.g. an application's own route) still take the translation
        # route for Hangul input; an explicit source_language always wins.
        detected = 'ko' if any('가' <= c <= '힣' for c in text) else 'en'
        language = params.get('source_language') or params.get('language') or detected
        params['source_language'] = str(language)
        translations = Path(repo_root)/'examples/beat-translations.json'
        table = dict(json.loads(translations.read_text(encoding='utf-8'))) if translations.exists() else {}
        if isinstance(params.get('translation_map'), dict):
            table.update(params['translation_map'])
        if isinstance(params.get('english_text'), str) and params['english_text'].strip():
            table[text] = params['english_text'].strip()
        params['translation_map'] = table
    else:
        # Translation fields are accepted and ignored by modes that do not translate.
        for key in ('source_language', 'language', 'english_text', 'translation_map', 'mode'):
            params.pop(key, None)
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
                size = int(handler.headers.get('Content-Length', '0') or 0)
                if not 0 < size <= 32_000:
                    raise ValueError('Text query must be under 32 KB')
                body = json.loads(handler.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('POST body must be a JSON object such as {"text": "..."}')
                params.update(body)
            text = params.pop('text', '')
            if isinstance(text, list):
                text = text[0] if text else ''
            result = query_application(repo_root, mode, text, params)
        status = 200
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        result, status = {'error': str(error) or type(error).__name__}, 400
    except Exception as error:  # keep the demo server alive and report the failure
        result, status = {'error': f'{type(error).__name__}: {error}'}, 500
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
    parser.add_argument('--processed', type=Path, help='Processed OmniMo BEAT root (<speaker>/meta.json)')
    parser.add_argument('--raw-root', type=Path, help='Local raw beat_english_v0.2.1 folder')
    parser.add_argument('--speakers', help='Comma list of speaker ids/names for the bank (processed or raw-root)')
    parser.add_argument('--takes', help='Comma list of take ids (public download route needs explicit ids)')
    parser.add_argument('--max-takes-per-speaker', type=int, default=1)
    parser.add_argument('--count', type=int, default=9, help='Bank clips (3-12)')
    parser.add_argument('--min-energy', type=float, default=0.08, help='Bank motion-energy floor (m/s)')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--strong-rules', type=Path)
    parser.add_argument('--sbert', help='Local Sentence-BERT folder for text matching (default: $BEAT_SBERT_MODEL; '
                                        'otherwise a labelled TF-IDF fallback). Nothing is downloaded.')
    parser.add_argument('--query')
    parser.add_argument('--rebuild', action='store_true', help='Regenerate the local bank and refit this demo adapter')
    args = parser.parse_args()
    if args.query:
        result = query_application(args.repo, args.mode, args.query)
    else:
        result = setup(args.repo, args.mode, processed=args.processed, raw_root=args.raw_root, speakers=args.speakers,
                       takes=args.takes, max_takes_per_speaker=args.max_takes_per_speaker, count=args.count,
                       min_energy=args.min_energy, epochs=args.epochs, strong_rules=args.strong_rules,
                       rebuild=args.rebuild, sbert=args.sbert)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
