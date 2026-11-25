#!/bin/bash
# Status of the time-step-000000 recovery run (read-only). Usage:  bash scripts/status_000000.sh [JOBID]
# Prints: scheduler state, levels done, parts present, merged files, errors, memory trend, run_info tail.
set -uo pipefail
main() {
W=${W:-/lustre/orion/cli900/world-shared/projects/XNR1K/audit/zonalmean/vod2uv.recovery}
R=$W/run_000000
J=${1:-3503423}
echo "== $(date -u +%FT%TZ) on $(hostname); user lookup: $(getent passwd "$(id -u)" | cut -d: -f1 || echo FAILED)"
echo "== scheduler"
squeue -u "$(id -u)" 2>&1 | head
sacct -j "$J" -X -n -o JobID,State,Elapsed,ExitCode,NodeList 2>&1 | head -3
sacct -j "$J" -n -o JobID%16,State,Elapsed,ExitCode 2>&1 | grep -vE "COMPLETED +[0-9:]+ +0:0" | head -20
echo "== levels"
echo "done lines in log: $(grep -c 'done in' "$R/log.txt"); skipped: $(grep -c 'parts exist, skipped' "$R/log.txt")"
nu=$(ls "$R"/parts/u_L*.grib 2>/dev/null | wc -l); nv=$(ls "$R"/parts/v_L*.grib 2>/dev/null | wc -l)
echo "parts: u $nu, v $nv (of 137); .tmp: $(ls "$R"/parts/*.tmp 2>/dev/null | wc -l)"
missing=""; for L in $(seq 1 137); do l=$(printf %03d $L)
  [ -s "$R/parts/u_L$l.grib" ] && [ -s "$R/parts/v_L$l.grib" ] || missing="$missing $L"; done
echo "levels missing u or v:${missing:- none}"
echo "== merged files"
ls -la "$R"/*recovered.grib 2>/dev/null || echo "none yet"
echo "== errors in logs"
grep -nE "Bus error|SIGBUS|Caught signal|Traceback|Error|error:" "$R/log.txt" "$W/run_000000_$J.out" 2>/dev/null | head -20 || true
echo "== last log lines"
tail -4 "$R/log.txt"
echo "== memory between chunks (last 8)"
tail -8 "$R/nodes_mem.txt" 2>/dev/null
echo "== run_info tail"
tail -12 "$R/run_info.txt"
}
main "$@"; exit $?
