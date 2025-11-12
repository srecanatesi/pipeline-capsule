#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
import shutil

def main():
    if len(sys.argv) < 2:
        print("Usage: preprocessing_nwb.py <SESSION_ID>", file=sys.stderr)
        sys.exit(2)

    session_id = str(sys.argv[1])
    nwb_dir = Path('/data/sessions_nwb')
    target = nwb_dir / f'session_{session_id}.nwb'
    if not target.exists():
        print(f"ERROR: Missing NWB file: {target}", file=sys.stderr)
        sys.exit(1)

    files = sorted(nwb_dir.glob('session_*.nwb'))
    try:
        idx = files.index(target)
    except ValueError:
        print(f"ERROR: Could not map {target} to index", file=sys.stderr)
        sys.exit(1)

    # Run the original script (it expects an index and uses ./data → set cwd='/')
    cmd = ['python', '/code/code/preprocessing_nwb.py', str(idx)]
    subprocess.run(cmd, check=True, cwd='/')

    # Move the output from /data → /results so Code Ocean collects it
    src = Path('/data/sessions_preprocessed') / f'df_{session_id}.pkl'
    dst_dir = Path('/results/sessions_preprocessed')
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst_dir / src.name)
        print(f"Saved: {dst_dir / src.name}")
    else:
        print(f"WARNING: Expected output not found: {src}", file=sys.stderr)

if __name__ == '__main__':
    main()