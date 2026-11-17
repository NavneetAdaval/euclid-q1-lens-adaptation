#!/usr/bin/env bash
# run_parallel.sh -- launch N parallel arc-on-real generator workers.
# Each worker makes N_TARGET/N of the lenses, writes to its own shard band.
# Usage:  bash run_parallel.sh 6      (6 workers; pick ~ cores/2 to leave headroom)

N=${1:-6}
cd "$(dirname "$0")/.." || exit 1     # repo root (script lives in scripts/)

echo "launching $N generator workers..."
mkdir -p logs
pids=()
for w in $(seq 0 $((N-1))); do
    python scripts/generate_arc_on_real.py --worker "$w" --n_workers "$N" \
        > "logs/gen_w${w}.log" 2>&1 &
    pids+=($!)
    echo "  worker $w -> PID $! (log: logs/gen_w${w}.log)"
done

echo "all $N workers running. Monitor with:  tail -f logs/gen_w0.log"
echo "waiting for all workers to finish..."
for pid in "${pids[@]}"; do wait "$pid"; done
echo "ALL WORKERS DONE."
