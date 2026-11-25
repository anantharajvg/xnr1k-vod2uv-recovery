#!/bin/bash
#SBATCH -J xnr1k_uv_000000
#SBATCH -A stf053
#SBATCH -N 4
#SBATCH -t 06:00:00
#SBATCH -o run_000000_%j.out
# All 137 levels of XNR1K u, v, time step 000000, native O8000 GRIB output.
# One level group of 32 ranks per node; levels dealt round-robin to groups.
#
# Version 4 (8 Oct): third SIGBUS (job 3503383, step 11) came AFTER a chunk had finished, inside MPI_Finalize
#   (UCX uct_mm_ep_flush after a TCP endpoint failure); venv was on NFS, so the Lustre explanation is refuted. v4 runs
#   Open MPI without UCX by default (MPI_STACK=ob1: pml ob1, btl self,vader,tcp; MPI_STACK=ucx restores v3), and
#   tolerates a non-zero srun exit if every level of the chunk has both parts (parts are written tmp+rename).
# Version 3 (8 Oct): refuses a venv on Lustre and records python/ecCodes paths (backtrace: SIGBUS in ecCodes
#   pack_double; libraries were mapped from Lustre). Version 2 (8 Oct 2026), after two bus errors (SIGBUS in UCX shared memory, both after ~9-10 levels per node):
#   levels run in short chunks, each its own srun step, so MPI/UCX shared memory and process memory start fresh
#   every chunk; node memory and /dev/shm use are logged between chunks (nodes_mem.txt); the per-level log line
#   reports root memory, node MemAvailable and /dev/shm use. Restartable as before: finished levels are skipped.
# Estimate: 137 levels x ~370 s / groups, plus ~30 s start-up per chunk.
# Submit from $W:  sbatch [-x badnode] scripts/run_000000.sbatch      (VENV must point to the venv)
set -euo pipefail
W=${W:?set W}; VENV=${VENV:?set VENV to the NFS venv, e.g. /ccs/proj/stf053/\$USER/venvs/andes/vod2uv}
X=/lustre/orion/cli900/world-shared/projects/XNR1K/data/h3f7-ndjf/global/O8000/ml-dgg
OUT=$W/run_000000
ROUNDS=${ROUNDS:-2}                          # levels per group per srun step
module load gcc openmpi miniforge3
source "$VENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MPI_STACK=${MPI_STACK:-ob1}
if [ "$MPI_STACK" = ob1 ]; then export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=^ucx; fi
# v3: refuse to run from a Python environment on Lustre (both bus errors were inside libraries mapped from Lustre)
fstype=$(df -PT "$(python3 -c 'import sys; print(sys.prefix)')" | awk 'NR==2{print $2}')
if [ "$fstype" = lustre ] && [ "${ALLOW_LUSTRE_VENV:-0}" != 1 ]; then
  echo "venv $(python3 -c 'import sys; print(sys.prefix)') is on Lustre; use one on NFS (set VENV), or ALLOW_LUSTRE_VENV=1" >&2
  exit 1
fi
NN=$SLURM_JOB_NUM_NODES; NP=$((32 * NN)); CH=$((NN * ROUNDS))
mkdir -p "$OUT"
{
  echo "# run_000000 v4 $(date -u +%Y-%m-%dT%H:%M:%SZ) job ${SLURM_JOB_ID:-none} nodes $NN ($SLURM_JOB_NODELIST) chunk $CH levels"
  for f in recover/recover_uv.py recover/gribio.py recover/core.py; do echo "# $f md5 $(md5sum < "$W/$f" | cut -d' ' -f1)"; done
  echo "# python $(command -v python3) prefix $(python3 -c 'import sys; print(sys.prefix)') ($fstype)"
  echo "# libeccodes $(python3 -c 'import findlibs; print(findlibs.find("eccodes"))')"
  echo "# MPI_STACK $MPI_STACK pml=${OMPI_MCA_pml:-default} btl=${OMPI_MCA_btl:-default} osc=${OMPI_MCA_osc:-default}"
  echo "# numpy $(python3 -c 'import numpy; print(numpy.__version__, numpy.__file__)')"
  for v in vo d u v; do for f in $X/$v/*_000000.ml-dgg.O8000.global.grib; do echo "# input $(stat -c '%s %y' "$f") $f"; done; done
} >> "$OUT/run_info.txt"
ARGS=(--vo $X/vo/vo_138_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
      --d  $X/d/d_155_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
      --template-u $X/u/u_131_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
      --template-v $X/v/v_132_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
      --group-size 32 --out-dir "$OUT")
nodemem() {   # one line per node: memory and /dev/shm
  srun -N "$NN" -n "$NN" --ntasks-per-node=1 bash -c \
    'echo "$(date -u +%H:%M:%S) $(hostname) $1 | $(free -g | awk "/Mem:/{print \"mem used \"\$3\" free \"\$4\" avail \"\$7\" GB\"}") | shm $(df -h /dev/shm | awk "NR==2{print \$3\" used of \"\$2}")"' _ "$1" \
    >> "$OUT/nodes_mem.txt" 2>&1 || true
}
start=$(date +%s)
for ((a = 1; a <= 137; a += CH)); do
  b=$((a + CH - 1)); [ $b -gt 137 ] && b=137
  nodemem "before $a-$b"
  set +e
  srun -N "$NN" -n "$NP" -c 1 python3 "$W/recover/recover_uv.py" "${ARGS[@]}" --levels "$a-$b" --no-merge \
       2>&1 | tee -a "$OUT/log.txt"
  rc=${PIPESTATUS[0]}; set -e
  missing=0; for ((L = a; L <= b; L++)); do l=$(printf %03d $L)
    [ -s "$OUT/parts/u_L$l.grib" ] && [ -s "$OUT/parts/v_L$l.grib" ] || missing=$((missing + 1)); done
  if [ "$rc" -ne 0 ]; then
    echo "# $(date -u +%H:%M:%S) chunk $a-$b: srun exit $rc, levels missing $missing" | tee -a "$OUT/run_info.txt"
    [ "$missing" -eq 0 ] || exit "$rc"          # a real failure: stop; resubmit restarts from the parts
  fi
done
nodemem "after all levels"
# merge: every level is now present, so this step skips all levels and only merges
set +e
srun -N "$NN" -n "$NP" -c 1 python3 "$W/recover/recover_uv.py" "${ARGS[@]}" --levels all 2>&1 | tee -a "$OUT/log.txt"
rc=${PIPESTATUS[0]}; set -e
[ "$rc" -eq 0 ] || echo "# merge step: srun exit $rc; merged files: $(ls "$OUT"/*recovered.grib 2>/dev/null | wc -l) of 2" >> "$OUT/run_info.txt"
echo "# job $SLURM_JOB_ID wall time $(( $(date +%s) - start )) s" >> "$OUT/run_info.txt"
ls -l "$OUT" >> "$OUT/run_info.txt"
