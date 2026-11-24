#!/usr/bin/env bash
# Run jobs of the blocked design locally, inside the runtime image and WITHOUT network.
# Instances must already be in the cache (download them on the host first; solvers are
# third-party code and never get network access here).
#
# Usage: scripts/local_job.sh DESIGN BLOCKS JOBS CACHE_DIR OUT_DIR
#   e.g. scripts/local_job.sh data/smoke/design.toml study/blocks.csv 0,1,2 data/smoke/instances data/smoke/out
set -euo pipefail
design="$1"; blocks="$2"; jobs="$3"; cache="$4"; out="$5"
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$out"
abs() { (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")"); }
design=$(abs "$design"); blocks=$(abs "$blocks"); cache=$(abs "$cache"); out=$(abs "$out")
exec docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$root/src:/opt/satbench/src:ro" \
  -v "$root/solvers/out:/opt/solvers:ro" \
  -v "$design:$design:ro" -v "$blocks:$blocks:ro" \
  -v "$cache:$cache" -v "$out:$out" \
  satbench-runtime:2026.11 \
  satbench blocks-run --design "$design" --blocks "$blocks" --jobs "$jobs" \
    --solvers-dir /opt/solvers --instances "$cache" \
    --results "$out/results.jsonl" --logs "$out/logs"
