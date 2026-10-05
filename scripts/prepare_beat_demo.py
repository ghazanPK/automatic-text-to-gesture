"""Prepare this paper's small local BEAT demonstration."""
from pathlib import Path
import sys
from beat_runtime import main
if __name__ == '__main__':
    sys.argv[1:1] = ['--repo', str(Path(__file__).resolve().parents[1]), '--mode', 'automatic']
    main()
