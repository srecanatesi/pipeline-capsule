#!/usr/bin/env bash
set -euo pipefail

# Run the process_session.py script
# It will automatically:
# - Download sessions.csv if needed
# - Filter for functional_connectivity sessions
# - Process sessions based on txt files in data/run1/, data/run2/, data/run3/
python /code/process_session.py




