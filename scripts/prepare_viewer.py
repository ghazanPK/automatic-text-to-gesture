"""Fetch pinned Three.js library locally; no dataset, avatar or model downloads."""
from pathlib import Path
from urllib.request import urlopen
import argparse

def prepare(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    modules={'three.module.js':'build/three.module.js','GLTFLoader.js':'examples/jsm/loaders/GLTFLoader.js','BufferGeometryUtils.js':'examples/jsm/utils/BufferGeometryUtils.js'}
    for filename,source in modules.items():
        with urlopen('https://cdn.jsdelivr.net/npm/three@0.170.0/'+source,timeout=30) as response:
            content=response.read()
        if filename!='three.module.js':
            content=content.decode().replace("from 'three'","from './three.module.js'").replace("from '../utils/BufferGeometryUtils.js'","from './BufferGeometryUtils.js'").encode()
        (destination/filename).write_bytes(content)
    print('Three.js 0.170.0 prepared in',destination)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='static/vendor');prepare(parser.parse_args().out)
