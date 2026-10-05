"""Start this repository's bundled example; private data and model weights are optional."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--skip-beat', action='store_true', help='Skip preparation; use the existing local cache or authored fixture when absent')
args = parser.parse_args()
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
prepare = ROOT / 'scripts' / 'prepare_viewer.py'
if prepare.exists() and not all((ROOT / 'static' / 'vendor' / name).is_file() for name in ('three.module.js', 'GLTFLoader.js', 'BufferGeometryUtils.js')):
    subprocess.run([sys.executable, str(prepare)], check=True)

if not args.skip_beat:
    subprocess.run([sys.executable, str(ROOT/'scripts/prepare_beat_demo.py')], check=True)

print(f'Open http://127.0.0.1:{args.port}/ — bundled starter samples are ready.', flush=True)
raise SystemExit(subprocess.call([sys.executable, *['scripts/demo_server.py', '--example'], '--port', str(args.port)]))
