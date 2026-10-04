"""Fetch pinned Three.js library locally; no dataset, avatar or model downloads."""
from pathlib import Path
from urllib.request import urlopen
import argparse

def prepare(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for filename in ('three.module.js','three.core.js'):
        try:
            with urlopen('https://cdn.jsdelivr.net/npm/three@0.170.0/build/'+filename,timeout=60) as response:
                content=response.read()
        except Exception:
            if filename=='three.core.js':continue  # 0.170 is a self-contained module.
            raise
        (destination/filename).write_bytes(content)
    print('Three.js 0.170.0 prepared in',destination)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='static/vendor');prepare(parser.parse_args().out)
