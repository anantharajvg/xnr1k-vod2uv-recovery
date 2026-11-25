#!/bin/bash
# Verify the real MPI + netCDF4 path on the cluster (run inside an allocation).
# usage: tests/selftest_mpi.sh "srun -n 8 -c 1" ["srun -n 1"]
#   arg1: parallel launcher; arg2: single-rank launcher for the serial reference
#   (default derived from arg1: "srun -n 1" / "mpirun -np 1"). The serial reference
#   is launched through the scheduler too: calling MPI_Init outside srun can fail
#   (e.g. Slurm auth plugin errors) on login nodes or in salloc shells.
set -e
LAUNCH=${1:-"mpirun -n 4"}
case "$LAUNCH" in
  srun*)   ONE_DEFAULT="srun -n 1" ;;
  mpirun*|mpiexec*) ONE_DEFAULT="mpirun -np 1" ;;
  *)       ONE_DEFAULT="" ;;
esac
ONE=${2:-$ONE_DEFAULT}
D=$(cd "$(dirname "$0")" && pwd); T=${TMPDIR:-/tmp}/vod2uv_selftest.$$; mkdir -p $T
echo "== 1/4 make analytic test file"
python3 $D/make_testfile.py $T/in.nc 160 150 -11.25
echo "== 2/4 parallel run: $LAUNCH"
$LAUNCH python3 $D/../vod2uv_mpi.py $T/in.nc $T/out.nc
python3 $D/check_output.py $T/in.nc $T/out.nc
echo "== 3/4 single-rank reference: ${ONE:-<direct>}"
$ONE python3 $D/../vod2uv_mpi.py --quiet $T/in.nc $T/out1.nc
echo "== 4/4 compare"
python3 - <<PY
import sys; sys.path.insert(0, "$D/..")
import numpy as np, vod2uv_mpi as V
a, b = V.Reader("$T/out.nc"), V.Reader("$T/out1.nc")
d = max(abs(a.read(n, slice(None)) - b.read(n, slice(None))).max() for n in ("u", "v"))
print("MPI vs single-rank max abs diff: %.3e -> %s" % (d, "PASS" if d == 0 else "CHECK"))
PY
rm -rf $T
