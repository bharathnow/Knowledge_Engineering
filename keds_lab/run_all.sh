#!/usr/bin/env bash
# Runs every experiment and saves its console output as output.txt in the experiment folder.
# Usage:  bash run_all.sh
set -e
cd "$(dirname "$0")"
python3 make_data.py
for d in exp01 exp02 exp03 exp04 exp05 exp06 exp07 exp08 exp09 exp10; do
  script=$(ls $d/*.py)
  echo "Running $script ..."
  python3 "$script" > "$d/output.txt" 2>&1
done
echo "Done. See exp*/output.txt"
