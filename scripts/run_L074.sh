#!/usr/bin/env bash
# run_L074.sh -- first XNR1K recovery: level 74, time step 000000, native O8000 GRIB output.
# Run inside an allocation (or the interactive srun --pty node), venv active.
#   bash scripts/run_L074.sh [ranks]        default 32 ranks, one level group
# Inputs are read only. Refuses to run if the output directory already exists (no overwrite).
# Gate A met on 7 Oct (M2a, M4, M3; owner's decisions recorded in the plan).
set -euo pipefail
NP=${1:-32}
W=${W:?set W to the vod2uv.recovery working directory}
X=${X:-/lustre/orion/cli900/world-shared/projects/XNR1K/data/h3f7-ndjf/global/O8000/ml-dgg}
OUT=$W/run_L074
VO=$X/vo/vo_138_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
D=$X/d/d_155_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
TU=$X/u/u_131_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
TV=$X/v/v_132_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
for f in "$VO" "$D" "$TU" "$TV"; do [ -r "$f" ] || { echo "missing or unreadable: $f" >&2; exit 1; }; done
[ -e "$OUT" ] && { echo "$OUT exists; move it aside first (this script never overwrites)" >&2; exit 1; }
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# inside an interactive srun --pty shell nested srun needs these (see runbook step 3)
export SLURM_OVERLAP=${SLURM_OVERLAP:-1} SLURM_CPU_BIND=${SLURM_CPU_BIND:-none}
{
  echo "# run_L074 $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) job ${SLURM_JOB_ID:-none} ranks $NP"
  echo "# recover_uv.py md5 $(md5sum < "$W/recover/recover_uv.py" | cut -d' ' -f1)"
  echo "# gribio.py md5 $(md5sum < "$W/recover/gribio.py" | cut -d' ' -f1)  core.py md5 $(md5sum < "$W/recover/core.py" | cut -d' ' -f1)"
  echo "# python $(command -v python3) prefix $(python3 -c 'import sys; print(sys.prefix)') ($(df -PT "$(python3 -c 'import sys; print(sys.prefix)')" | awk 'NR==2{print $2}'))"
  echo "# eccodes $(python3 -c 'import eccodes, findlibs; print(eccodes.codes_get_api_version(), findlibs.find("eccodes"))')"
  for f in "$VO" "$D" "$TU" "$TV"; do echo "# input $(stat -c '%s %y' "$f") $f"; done
} > "$OUT/run_info.txt"
start=$(date +%s)
srun -n "$NP" -c 1 python3 "$W/recover/recover_uv.py" --vo "$VO" --d "$D" \
     --template-u "$TU" --template-v "$TV" --levels 74 --group-size "$NP" --out-dir "$OUT" \
     2>&1 | tee "$OUT/log.txt"
echo "# wall time $(( $(date +%s) - start )) s" | tee -a "$OUT/run_info.txt"
ls -l "$OUT" | tee -a "$OUT/run_info.txt"
