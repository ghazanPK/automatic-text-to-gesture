"""Start this repository's bundled example; private data and model weights are optional."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--skip-beat', action='store_true', help='Skip preparation; use the existing local cache or authored fixture when absent')
parser.add_argument('--skip-paper-method', action='store_true', help='Do not run scripts/prepare_paper_method.py; serve the BEAT demo adapter')
args = parser.parse_args()
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
prepare = ROOT / 'scripts' / 'prepare_viewer.py'
if prepare.exists() and not all((ROOT / 'static' / 'vendor' / name).is_file() for name in ('three.module.js', 'GLTFLoader.js', 'BufferGeometryUtils.js')):
    subprocess.run([sys.executable, str(prepare)], check=True)

# >>> paperreach beat preparation (managed by tools/integrate-beat-methods.py)
# BEAT preparation and the optional paper-method hook are non-fatal: on failure
# the server still starts with the existing local cache or the authored starter.
paper_args = None
if not args.skip_beat:
    prepared = subprocess.run([sys.executable, str(ROOT/'scripts/prepare_beat_demo.py')])
    if prepared.returncode:
        print('BEAT preparation failed (see the message above); continuing with the existing local cache '
              'or the bundled authored starter. Retry with: python scripts/prepare_beat_demo.py', flush=True)
paper_hook = ROOT/'scripts/prepare_paper_method.py'
if paper_hook.exists() and not args.skip_paper_method:
    hook = subprocess.run([sys.executable, str(paper_hook)], stdout=subprocess.PIPE, text=True)
    print(hook.stdout or '', end='', flush=True)
    lines = [line for line in (hook.stdout or '').splitlines() if line.strip()]
    try:
        paper_result = json.loads(lines[-1]) if hook.returncode == 0 and lines else None
    except ValueError:
        paper_result = None
    if isinstance(paper_result, dict) and paper_result.get('ready'):
        os.environ['PAPER_METHOD_RESULT'] = json.dumps(paper_result)
        if isinstance(paper_result.get('server_args'), list) and paper_result['server_args']:
            paper_args = [str(arg) for arg in paper_result['server_args']]
    else:
        print('Paper-method preparation did not complete; serving the BEAT demo adapter instead. '
              'Details: python scripts/prepare_paper_method.py', flush=True)
# <<< paperreach beat preparation

print(f'Open http://127.0.0.1:{args.port}/ — bundled starter samples are ready.', flush=True)
raise SystemExit(subprocess.call([sys.executable, *(paper_args or ['scripts/demo_server.py', '--example']), '--port', str(args.port)]))
