#!/usr/bin/env python3
"""
Download a single session NWB file from Allen Brain Observatory S3.

This script downloads the NWB file for a specific session ID, along with
supporting CSV files (sessions.csv and unit_table_all.csv).
"""

import os
import sys
import shutil
from pathlib import Path

from utils import SESSION_NWB_KEY, SESSIONS_TABLE_KEY, download_file


def main():
    """
    Main function to download a single session NWB file.
    
    The script:
    1. Determines the data directory (container vs local)
    2. Downloads sessions.csv if needed
    3. Finds and copies unit_table_all.csv if available
    4. Downloads the session NWB file if it doesn't exist
    """
    # Get session ID from environment variable or command line argument
    sid = os.environ.get('SESSION_ID') or (len(sys.argv) > 1 and sys.argv[1])
    if not sid:
        print('Usage: SESSION_ID=<id> download_single_session.py', file=sys.stderr)
        sys.exit(1)

    session_id = str(int(sid))
    
    # Determine data directory (support both container /data and local data/)
    data_dir_abs = Path('/data')
    data_dir_rel = Path(__file__).resolve().parents[1] / 'data'
    
    # Prefer relative path if it exists (for local development)
    # Otherwise use absolute path (for container environments)
    if data_dir_rel.exists() and data_dir_rel.is_dir():
        out_dir = data_dir_rel
    elif data_dir_abs.exists() and data_dir_abs.is_dir():
        out_dir = data_dir_abs
    else:
        # Default to relative for local development
        out_dir = data_dir_rel
    
    # Create NWB directory
    nwb_dir = out_dir / 'sessions_nwb'
    nwb_dir.mkdir(parents=True, exist_ok=True)

    # Ensure sessions.csv exists in data directory
    sessions_csv = out_dir / 'sessions.csv'
    if not sessions_csv.exists() or sessions_csv.stat().st_size == 0:
        if sessions_csv.exists() and sessions_csv.stat().st_size == 0:
            print(f"[Session {session_id}] sessions.csv exists but is empty, re-downloading...")
        else:
            print(f"[Session {session_id}] Fetching sessions.csv → {sessions_csv}")
        download_file(SESSIONS_TABLE_KEY, str(sessions_csv), show_progress=False, session_id=int(session_id))
    else:
        print(f"[Session {session_id}] sessions.csv already exists, skipping download")

    # Search recursively for unit_table_all.csv in data folder
    # This file contains unit metadata and should be in the repository
    unit_table_csv = out_dir / 'unit_table_all.csv'
    if not unit_table_csv.exists():
        # Search recursively in data folder for unit_table_all.csv
        candidates = []
        if out_dir.exists():
            candidates = list(out_dir.rglob('unit_table_all.csv'))
        if candidates:
            src = candidates[0]
            print(f"[Session {session_id}] Found unit_table_all.csv at {src}, copying to {unit_table_csv}")
            shutil.copy2(src, unit_table_csv)
        else:
            print(f"[Session {session_id}] WARNING: unit_table_all.csv not found recursively in {out_dir}; downstream preprocessing may fail.")

    # Download NWB file if it doesn't exist
    key = SESSION_NWB_KEY.format(sid=session_id)
    dest = nwb_dir / f'session_{session_id}.nwb'
    
    # Skip download if file already exists and is not empty
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[Session {session_id}] NWB file already exists ({dest.stat().st_size / (1024*1024):.1f} MB): {dest}")
        print(f"[Session {session_id}] Skipping download")
        return

    # Remove empty file if it exists
    if dest.exists() and dest.stat().st_size == 0:
        print(f"[Session {session_id}] NWB file exists but is empty, re-downloading...")
        dest.unlink()

    # Download the NWB file
    print(f"[Session {session_id}] Downloading NWB file to {dest} ...")
    download_file(key, str(dest), show_progress=True, session_id=int(session_id))
    
    # Verify download completed successfully
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[Session {session_id}] ✓ Download complete ({dest.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"[Session {session_id}] ⚠ Warning: Download may have failed (file size: {dest.stat().st_size if dest.exists() else 0} bytes)")


if __name__ == '__main__':
    main()
