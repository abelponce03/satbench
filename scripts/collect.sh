#!/usr/bin/env bash
# Download every results artifact of the given workflow runs into data/runs/<run-id>/.
# Usage: scripts/collect.sh RUN_ID [RUN_ID ...]
set -euo pipefail
for run in "$@"; do
  mkdir -p "data/runs/$run"
  gh run download "$run" --pattern 'results-job-*' --dir "data/runs/$run"
done
echo "now: satbench blocks-pending --results-dir data/runs --out study/blocks-pending.csv"
