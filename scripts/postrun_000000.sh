#!/bin/bash
# Post-run checks of time step 000000 (all 137 levels). Run on the interactive node, venv active:
#   bash scripts/postrun_000000.sh            # ~1 h; reads ~350 GB; writes only under run_000000/checks and results/
# Steps (each logged to run_000000/checks/postrun_<UTC>.log, never overwritten):
#   1. chunks the batch script carried on past (non-zero srun exit with all parts present)
#   2. M8 + cos(lat) check on every level not yet checked (check_parts.sh, 4 in parallel)
#   3. residual budget over all levels, and the pre-registered test for hypothesis (g):
#      Spearman rho(delta, rms wind speed) > 0.5 supports (g); |rho| < 0.2 counts against it
#   4. integrity: md5 of each merged file equals md5 of its 137 parts concatenated in level order
#   5. reproducibility: first attempt (other venv) vs clean run, levels 1-41, byte comparison
#   6. provenance manifest (scripts/provenance.sh)
set -uo pipefail
main() {
W=${W:-/lustre/orion/cli900/world-shared/projects/XNR1K/audit/zonalmean/vod2uv.recovery}
R=$W/run_000000; C=$R/checks; mkdir -p "$C"
LOG=$C/postrun_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
echo "== postrun_000000.sh $(date -u +%FT%TZ) on $(hostname); script md5 $(md5sum < "$0" | cut -d' ' -f1); log $LOG"
export SLURM_OVERLAP=1 SLURM_CPU_BIND=none OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=^ucx
cd "$W" || exit 1

echo "== 1. chunks with a non-zero srun exit (tolerated only if all parts were present)"
grep -n "srun exit" "$R/run_info.txt" || echo "none"

echo "== 2. per-level checks ($(date -u +%H:%M:%S))"
bash scripts/check_parts.sh "$R" 4 | tail -3
echo "levels checked: $(grep -c '^L' "$C/summary.txt"); M8 not 2/2: $(grep '^L' "$C/summary.txt" | grep -vc 'M8 PASS 2/2')"

echo "== 3. residual budget, all levels ($(date -u +%H:%M:%S))"
python3 analysis/cosphi_budget.py "$C"/cosphi_L*.csv --stats "$R/level_stats.csv" --out "$C/cosphi_budget.csv" | tail -4
python3 - "$C/cosphi_budget.csv" <<'PY'
import csv, math, sys
from scipy.stats import spearmanr
rows = list(csv.DictReader(open(sys.argv[1])))
d = {(int(r["level"]), r["var"]): r for r in rows}
levels = sorted({k[0] for k in d})
spd = {L: math.hypot(float(d[(L, "u")]["rms"]), float(d[(L, "v")]["rms"])) for L in levels
       if d[(L, "u")]["rms"] and d[(L, "v")]["rms"]}
for v in ("u", "v"):
    Ls = [L for L in levels if L in spd]
    rho, p = spearmanr([spd[L] for L in Ls], [float(d[(L, v)]["delta"]) for L in Ls])
    verdict = "supports (g)" if rho > 0.5 else ("counts against (g)" if abs(rho) < 0.2 else "inconclusive")
    print("Spearman rho(delta_%s, rms speed) = %+.3f (p %.1e, %d levels; speed %.1f..%.1f m/s) -> %s"
          % (v, rho, p, len(Ls), min(spd.values()), max(spd.values()), verdict))
PY

echo "== 4. merged files = concatenated parts ($(date -u +%H:%M:%S))"
for v in u v; do
  M=$(ls "$R"/${v}_*recovered.grib)
  a=$(md5sum < "$M" | cut -d' ' -f1)
  b=$(for L in $(seq -f %03g 1 137); do cat "$R/parts/${v}_L$L.grib"; done | md5sum | cut -d' ' -f1)
  echo "$v merged md5 $a, parts md5 $b, size $(stat -c %s "$M") -> $([ "$a" = "$b" ] && echo PASS || echo FAIL)"
done

echo "== 5. first attempt vs clean run, levels 1-41 ($(date -u +%H:%M:%S))"
srun -n 1 python3 analysis/compare_runs.py --a-parts run_000000_attempt1/parts --b-parts "$R/parts" --levels 1-41 | tail -8

echo "== 6. provenance ($(date -u +%H:%M:%S))"
bash scripts/provenance.sh | tail -1
echo "== done $(date -u +%FT%TZ)"
}
main "$@"; exit $?
