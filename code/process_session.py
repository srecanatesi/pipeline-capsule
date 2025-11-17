#!/usr/bin/env python3
"""
Main pipeline script for processing neural state sessions.

This script orchestrates the complete processing pipeline:
1. Downloads sessions.csv and filters for functional_connectivity sessions
2. Finds txt files (format: <number>.txt) recursively in data folder
3. For each txt file, processes the corresponding session:
   - Downloads NWB file
   - Preprocesses neural data
   - Runs HMM cross-validation
   - Runs HMM analysis

The number in the txt filename determines which session to process:
- 1.txt -> 1st functional_connectivity session (index 0)
- 2.txt -> 2nd functional_connectivity session (index 1)
- N.txt -> Nth functional_connectivity session (index N-1)

Usage:
    python process_session.py

The script will automatically:
- Find all <number>.txt files in the data directory
- Process each corresponding session through the full pipeline
- Save results to the results directory

Environment Support:
- Supports both container environments (/data, /results) and local development (./data, ./results)
- Automatically detects which environment is being used
"""

import sys
import subprocess
import pandas as pd
import re
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# Import configuration and utilities
import config
from utils import SESSIONS_TABLE_KEY, download_file

# Set numpy random seed for reproducibility
np.random.seed(config.RANDOM_SEED)


# ============================================================================
# Constants
# ============================================================================

TXT_FILE_PATTERN = re.compile(r'^(\d+)\.txt$')  # Pattern for <number>.txt files
SESSION_TYPE_FILTER = 'functional_connectivity'  # Session type to filter for
SEPARATOR_LENGTH = 60  # Length of separator lines for output formatting


# ============================================================================
# Helper Functions
# ============================================================================

def run_command(cmd: List[str]) -> None:
    """
    Execute a shell command and print it.
    
    Args:
        cmd: List of command arguments (e.g., ['python', 'script.py', 'arg'])
        
    Raises:
        CalledProcessError: If command fails (propagated from subprocess.run)
        FileNotFoundError: If command executable is not found
        
    Example:
        >>> run_command(['python', 'script.py', 'session_id'])
        >> python script.py session_id
    """
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_output_directories(data_dir: Path, results_dir: Path) -> None:
    """
    Ensure all required output directories exist.
    
    Creates the following directories if they don't exist:
    - data_dir/sessions_nwb
    - results_dir/sessions_preprocessed
    - results_dir/sessions_hmm_crossval
    - results_dir/sessions_hmm_analysis
    - results_dir/logs
    
    Args:
        data_dir: Path to data directory
        results_dir: Path to results directory
    """
    directories = [
        data_dir / config.NWB_FOLDER_NAME,
        results_dir / config.PREPROCESSED_FOLDER_NAME,
        results_dir / config.HMM_CROSSVAL_FOLDER_NAME,
        results_dir / 'sessions_hmm_analysis',
        results_dir / config.LOGS_FOLDER_NAME,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def find_txt_files(data_dir: Path) -> List[Tuple[int, Path]]:
    """
    Find all txt files matching pattern <number>.txt recursively in data directory.
    
    Args:
        data_dir: Directory to search recursively
        
    Returns:
        List of tuples (run_number, file_path) sorted by run number.
        Run number is extracted from filename (e.g., "1.txt" -> 1)
        
    Example:
        >>> files = find_txt_files(Path('./data'))
        >>> # Returns: [(1, Path('./data/1.txt')), (2, Path('./data/2.txt'))]
    """
    txt_files = []
    print(f'Looking for txt files recursively in: {data_dir}')
    
    for txt_file in data_dir.rglob('*.txt'):
        filename = txt_file.name
        match = TXT_FILE_PATTERN.match(filename)
        if match:
            run_num = int(match.group(1))
            txt_files.append((run_num, txt_file))
            print(f'  Found: {txt_file} -> run number {run_num}')
        else:
            print(f'  Skipping: {txt_file} (does not match pattern <number>.txt)')
    
    txt_files.sort(key=lambda x: x[0])  # Sort by run number
    return txt_files


def download_sessions_table(data_dir: Path) -> Path:
    """
    Download sessions.csv file if it doesn't exist or is empty.
    
    Args:
        data_dir: Directory where sessions.csv should be saved
        
    Returns:
        Path to sessions.csv file
        
    Raises:
        FileNotFoundError: If download fails
        IOError: If downloaded file is corrupted
    """
    sessions_csv = data_dir / config.SESSIONS_CSV_NAME
    
    if not sessions_csv.exists() or (sessions_csv.exists() and sessions_csv.stat().st_size == 0):
        if sessions_csv.exists() and sessions_csv.stat().st_size == 0:
            print(f'sessions.csv exists but is empty, re-downloading...')
        else:
            print(f'Downloading sessions.csv → {sessions_csv}')
        download_file(SESSIONS_TABLE_KEY, str(sessions_csv), show_progress=True, session_id=0)
    else:
        file_size_kb = sessions_csv.stat().st_size / 1024
        print(f'sessions.csv already exists ({file_size_kb:.1f} KB), skipping download')
    
    return sessions_csv


def load_and_filter_sessions(sessions_csv: Path) -> pd.DataFrame:
    """
    Load sessions table and filter for functional_connectivity sessions.
    
    Args:
        sessions_csv: Path to sessions.csv file
        
    Returns:
        Filtered DataFrame containing only functional_connectivity sessions,
        sorted by session ID and reset index.
        
    Raises:
        FileNotFoundError: If sessions.csv doesn't exist
        pd.errors.EmptyDataError: If sessions.csv is empty
        KeyError: If required columns ('id', 'session_type') are missing
    """
    print(f'Loading sessions table from {sessions_csv}')
    df = pd.read_csv(sessions_csv)
    
    # Sort by id (ecephys_session_id)
    df = df.sort_values('id').reset_index(drop=True)
    
    # Filter by session_type == "functional_connectivity"
    df_filtered = df[df['session_type'] == SESSION_TYPE_FILTER].reset_index(drop=True)
    
    print(f'Total sessions: {len(df)}')
    print(f'Functional connectivity sessions: {len(df_filtered)}')
    
    if len(df_filtered) == 0:
        print('ERROR: No functional_connectivity sessions found', file=sys.stderr)
        sys.exit(1)
    
    return df_filtered


def get_session_id_from_index(df_filtered: pd.DataFrame, session_idx: int) -> str:
    """
    Get session ID from filtered dataframe by index.
    
    Args:
        df_filtered: Filtered sessions DataFrame
        session_idx: Zero-based index into filtered sessions
        
    Returns:
        Session ID as string
        
    Raises:
        IndexError: If session_idx is out of range
    """
    return str(int(df_filtered.iloc[session_idx]['id']))


def format_time(seconds: float) -> str:
    """
    Format time duration in a human-readable way.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted string (e.g., "120.5s (2.0 minutes)")
    """
    minutes = seconds / 60
    return f'{seconds:.1f}s ({minutes:.1f} minutes)'


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_single_session(session_id: str) -> None:
    """
    Process a single session through the complete pipeline.
    
    Pipeline steps:
    1. Download NWB file (if not already downloaded)
    2. Preprocess NWB data (extract spikes, filter units, extract behavioral data)
    3. Run HMM cross-validation (determine optimal states and fit final model)
    4. Run HMM analysis (post-HMM analysis including dimensionality analysis)
    
    Each step is executed as a separate subprocess to ensure proper isolation
    and error handling.
    
    Args:
        session_id: Session ID string (e.g., "766640955")
        
    Raises:
        CalledProcessError: If any pipeline step fails
        FileNotFoundError: If required scripts are not found
        
    Example:
        >>> process_single_session("766640955")
        [SESSION 766640955] Starting processing pipeline
        ...
    """
    start_time = time.time()
    
    # Get directories using config helper functions
    data_dir = config.get_data_dir()
    results_dir = config.get_results_dir()
    
    # Print session header
    separator = '=' * SEPARATOR_LENGTH
    print(f'\n{separator}')
    print(f'[SESSION {session_id}] Starting processing pipeline')
    print(f'{separator}')
    print(f'[SESSION {session_id}] Data directory: {data_dir}')
    print(f'[SESSION {session_id}] Results directory: {results_dir}')
    
    # Ensure output directories exist
    print(f'[SESSION {session_id}] Creating output directories...')
    ensure_output_directories(data_dir, results_dir)
    print(f'[SESSION {session_id}] [OK] Output directories ready')
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # STEP 1: Download NWB file
    print(f'\n[SESSION {session_id}] STEP 1/4: Downloading NWB file...')
    print(f'[SESSION {session_id}] {"-" * SEPARATOR_LENGTH}')
    step_start = time.time()
    run_command(['python', str(script_dir / 'download_single_session.py'), str(session_id)])
    step_time = time.time() - step_start
    print(f'[SESSION {session_id}] [OK] STEP 1/4 completed in {format_time(step_time)}')
    
    # STEP 2: Preprocess NWB data
    print(f'\n[SESSION {session_id}] STEP 2/4: Preprocessing NWB data...')
    print(f'[SESSION {session_id}] {"-" * SEPARATOR_LENGTH}')
    step_start = time.time()
    run_command(['python', str(script_dir / 'preprocessing_nwb.py'), str(session_id)])
    step_time = time.time() - step_start
    print(f'[SESSION {session_id}] [OK] STEP 2/4 completed in {format_time(step_time)}')
    
    # STEP 3: Run HMM cross-validation
    print(f'\n[SESSION {session_id}] STEP 3/4: Running HMM cross-validation...')
    print(f'[SESSION {session_id}] {"-" * SEPARATOR_LENGTH}')
    step_start = time.time()
    # Pass session_id to hmm_crossvalidation.py to process only this specific session
    run_command(['python', str(script_dir / 'hmm_crossvalidation.py'), '--session-id', str(session_id)])
    step_time = time.time() - step_start
    print(f'[SESSION {session_id}] [OK] STEP 3/4 completed in {format_time(step_time)}')
    
    # STEP 4: Run HMM analysis
    print(f'\n[SESSION {session_id}] STEP 4/4: Running HMM analysis...')
    print(f'[SESSION {session_id}] {"-" * SEPARATOR_LENGTH}')
    step_start = time.time()
    # Pass session_id to hmm_analysis.py to process only this specific session
    run_command(['python', str(script_dir / 'hmm_analysis.py'), '--session-id', str(session_id)])
    step_time = time.time() - step_start
    print(f'[SESSION {session_id}] [OK] STEP 4/4 completed in {format_time(step_time)}')
    
    # Print summary
    total_time = time.time() - start_time
    print(f'\n[SESSION {session_id}] {separator}')
    print(f'[SESSION {session_id}] [OK] ALL STEPS COMPLETED')
    print(f'[SESSION {session_id}] Total processing time: {format_time(total_time)}')
    print(f'[SESSION {session_id}] {separator}')


def main() -> None:
    """
    Main function: orchestrates processing of multiple sessions.
    
    Workflow:
    1. Determine data directory (container vs local)
    2. Download sessions.csv if needed
    3. Load and filter for functional_connectivity sessions
    4. Find txt files recursively in data folder
    5. Process each session corresponding to found txt files
    
    The script processes sessions based on txt files found in the data directory.
    Each txt file named <number>.txt corresponds to the Nth functional_connectivity
    session (where N is the number in the filename).
    
    Exit codes:
        0: Success
        1: No functional_connectivity sessions found
        1: Subprocess error (from pipeline steps)
        
    Example:
        If data/ contains 1.txt and 2.txt:
        - 1.txt -> processes 1st functional_connectivity session
        - 2.txt -> processes 2nd functional_connectivity session
    """
    # Get data directory using config helper function
    data_dir = config.get_data_dir()
    print(f'Using data directory: {data_dir}')
    
    # Download sessions table if needed
    sessions_csv = download_sessions_table(data_dir)
    
    # Load and filter sessions
    df_filtered = load_and_filter_sessions(sessions_csv)
    
    # Find txt files to process
    txt_files = find_txt_files(data_dir)
    
    # Exit gracefully if no txt files found
    if len(txt_files) == 0:
        print(f'\nNo txt files matching pattern <number>.txt found in {data_dir}')
        print('Nothing to process. Exiting gracefully.')
        return
    
    print(f'\nFound {len(txt_files)} txt file(s) to process')
    
    # Process sessions corresponding to each txt file
    # Mapping: txt file N.txt -> session index N-1 (Nth session in filtered list)
    pipeline_start = time.time()
    
    separator = '=' * SEPARATOR_LENGTH
    print(f'\n{separator}')
    print(f'PIPELINE SUMMARY')
    print(f'{separator}')
    print(f'Total sessions to process: {len(txt_files)}')
    print(f'Functional connectivity sessions available: {len(df_filtered)}')
    print(f'{separator}\n')
    
    sessions_processed = 0
    sessions_skipped = 0
    
    for idx, (run_num, txt_file) in enumerate(txt_files, 1):
        session_idx = run_num - 1  # Convert to 0-based index
        
        # Skip if session index exceeds available sessions
        if session_idx >= len(df_filtered):
            print(f'\n[RUN {run_num}] WARNING: Session index {session_idx} (from {txt_file.name}) '
                  f'exceeds available sessions ({len(df_filtered)}). Skipping.', file=sys.stderr)
            sessions_skipped += 1
            continue
        
        # Get session ID from filtered dataframe
        try:
            session_id = get_session_id_from_index(df_filtered, session_idx)
        except IndexError as e:
            print(f'\n[RUN {run_num}] ERROR: Failed to get session ID for index {session_idx}: {e}', 
                  file=sys.stderr)
            sessions_skipped += 1
            continue
        
        # Print run header
        run_separator = '#' * SEPARATOR_LENGTH
        print(f'\n{run_separator}')
        print(f'[RUN {run_num}/{len(txt_files)}] Processing session from {txt_file.name}')
        print(f'[RUN {run_num}/{len(txt_files)}] Session ID: {session_id} (index {session_idx} in filtered list)')
        print(f'{run_separator}')
        
        # Process the session
        try:
            process_single_session(session_id)
            sessions_processed += 1
            print(f'\n[RUN {run_num}/{len(txt_files)}] [OK] Completed run {run_num}')
        except subprocess.CalledProcessError as e:
            print(f'\n[RUN {run_num}] ERROR: Pipeline step failed: {e}', file=sys.stderr)
            sessions_skipped += 1
            continue
        except Exception as e:
            print(f'\n[RUN {run_num}] ERROR: Unexpected error: {e}', file=sys.stderr)
            sessions_skipped += 1
            continue
    
    # Print final summary
    total_pipeline_time = time.time() - pipeline_start
    print(f'\n{separator}')
    print('PIPELINE COMPLETE')
    print(f'{separator}')
    print(f'Total sessions processed: {sessions_processed}')
    print(f'Total sessions skipped: {sessions_skipped}')
    print(f'Total pipeline time: {format_time(total_pipeline_time)}')
    if sessions_processed > 0:
        avg_time = total_pipeline_time / sessions_processed
        print(f'Average time per session: {format_time(avg_time)}')
    print(f'{separator}')


if __name__ == '__main__':
    main()
