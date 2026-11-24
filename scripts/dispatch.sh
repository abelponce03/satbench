#!/usr/bin/env bash
# Dispatch the blocked design in ranges of at most 256 jobs (GitHub matrix limit).
# Usage: scripts/dispatch.sh IMAGE_DIGEST_REF [MAX_PARALLEL] [FIRST] [LAST]
set -euo pipefail
image="$1"; maxp="${2:-10}"
total=$(python3 -c "import pandas as pd; print(int(pd.read_csv('study/blocks.csv').job.max()))")
first="${3:-0}"; last="${4:-$total}"
for ((s = first; s <= last; s += 256)); do
  e=$(( s + 255 < last ? s + 255 : last ))
  echo "dispatching jobs $s..$e"
  gh workflow run blocks.yml -f first="$s" -f last="$e" -f image="$image" -f max_parallel="$maxp"
done
