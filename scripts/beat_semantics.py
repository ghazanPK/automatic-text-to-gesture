"""Resolve cached LLM annotations or call a user's compatible local/hosted LLM.

Only transcript text is sent to an explicitly supplied endpoint. Responses must
be contiguous spans of the supplied clip transcripts; arbitrary gestures cannot
be invented by the annotation model.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request


def resolve(bank, raw):
    provenance = raw['provenance']
    if provenance.get('kind') not in {'llm_json', 'manual_annotation'}:
        raise ValueError('Annotation provenance must identify its real source')
    rules = []
    for phrase in raw['phrases']:
        words = phrase.casefold().split()
        if not 3 <= len(words) <= 10:
            raise ValueError('Strong phrases must contain 3–10 words')
        for clip in bank['clips']:
            source = clip['text'].casefold().split()
            if any(source[i:i+len(words)] == words for i in range(len(source)-len(words)+1)):
                rules.append({'phrase': phrase, 'gesture_id': clip['id'], 'provenance': provenance})
                break
    return {'rules': rules, 'provenance': provenance}


def extract_with_paper_prompt(bank, endpoint, model):
    """RIDGE Section 3.1 extraction (verbatim prompt, ridge_gesture.annotate) over each bank clip transcript."""
    deps = Path(__file__).resolve().parent/'beat_deps'
    if deps.is_dir() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    from ridge_gesture.annotate import PROMPT_SHA256, annotate_record
    rules = []
    provenance = {'kind': 'llm_json', 'producer': model, 'prompt': 'RIDGE paper Section 3.1 (verbatim)',
                  'prompt_sha256': PROMPT_SHA256,
                  'scope': 'Live extraction from supplied BEAT clip transcripts via ridge_gesture.annotate'}
    for clip in bank['clips']:
        words = [{'word': w.get('text', w.get('word')), 'start_frame': w['start_frame'], 'end_frame': w['end_frame']}
                 for w in clip.get('words') or []]
        if not words:
            continue
        row = annotate_record({'record_id': clip['id'], 'text': clip['text'], 'words': words}, mode='llm',
                              endpoint=endpoint, model=model, api_key_env='BEAT_LLM_API_KEY')
        rules += [{'phrase': p['phrase'], 'gesture_id': clip['id'], 'provenance': provenance} for p in row['phrases']]
    return {'rules': rules, 'provenance': provenance}


def extract(bank, endpoint, model):
    try:
        return extract_with_paper_prompt(bank, endpoint, model)
    except ImportError:
        pass  # ridge_gesture not importable: fall back to the compact JSON prompt below
    transcripts = [{'gesture_id': c['id'], 'text': c['text']} for c in bank['clips']]
    prompt = ('Return JSON {"phrases":[...]} containing at most five salient contiguous spans of 3–10 words '
              'copied exactly from these transcripts. Select meaningful content, not filler. The motion is '
              'observed alongside text; do not infer gesture semantics or invent text.\n'+json.dumps(transcripts))
    headers = {'Content-Type': 'application/json'}
    token = os.environ.get('BEAT_LLM_API_KEY')
    if token:
        headers['Authorization'] = 'Bearer '+token
    request = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
        data=json.dumps({'model': model, 'messages': [{'role':'user','content':prompt}],
                         'temperature':0, 'response_format':{'type':'json_object'}}).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.load(response)
    raw = json.loads(answer['choices'][0]['message']['content'])
    raw['provenance'] = {'kind':'llm_json','producer':model,'scope':'Live extraction from supplied BEAT clip transcripts'}
    result = resolve(bank, raw)
    if len(result['rules']) != len(raw['phrases']):
        raise ValueError('LLM returned an unaligned or duplicate phrase; review its annotation')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, default=Path('outputs/beat-library/bank.json'))
    parser.add_argument('--annotations', type=Path, default=Path(__file__).with_name('beat-semantic-annotations.json'))
    parser.add_argument('--endpoint', help='Explicit OpenAI-compatible base URL, e.g. http://localhost:1234/v1')
    parser.add_argument('--model')
    parser.add_argument('--output', type=Path, default=Path('outputs/beat-library/strong-rules.json'))
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text(encoding='utf-8'))
    if args.endpoint:
        if not args.model:
            parser.error('--endpoint requires --model')
        result = extract(bank, args.endpoint, args.model)
    else:
        result = resolve(bank, json.loads(args.annotations.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'rules':len(result['rules']),'output':str(args.output)}))


if __name__ == '__main__':
    main()
