#!/bin/bash
# GRIB-path self-test on Andes (inside an allocation, venv active).
# usage: bash tests/selftest_grib.sh [ranks]      e.g. bash tests/selftest_grib.sh 4
set -euo pipefail
NP=${1:-4}
D=$(cd "$(dirname "$0")" && pwd)
T=${TMPDIR:-/tmp}/recover_selftest.$$
mkdir -p "$T"
echo "== 1/3 make small GRIB2 test files in $T"
srun -n 1 python3 "$D/grib_selftest.py" make "$T/in"
echo "== 2/3 recover with $NP ranks in 2 level groups"
G=$(( NP / 2 )); [ "$G" -lt 1 ] && G=1
srun -n "$NP" -c 1 python3 "$D/../recover_uv.py" --vo "$T/in/vo.grib" --d "$T/in/d.grib" \
     --template-u "$T/in/u.grib" --template-v "$T/in/v.grib" --levels all \
     --group-size "$G" --out-dir "$T/out"
echo "== 3/3 checks"
srun -n 1 python3 "$D/grib_selftest.py" check "$T/in" "$T/out"
echo "(test files kept in $T; remove when done)"
