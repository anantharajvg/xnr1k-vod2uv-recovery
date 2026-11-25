#!/bin/bash
# Post-run checks (M8 header + sanity, cos(lat) residual) on finished per-level parts, while the batch job runs.
# Reads only. A level is checked only if the log reports it done (both u and v parts written).
# usage (interactive node, venv active):  bash scripts/check_parts.sh RUN_DIR [NPAR] [LEVELS...]
#   e.g. bash scripts/check_parts.sh $W/run_000000 4            # all finished levels not yet checked
# Output per level in RUN_DIR/checks/: check_LNNN.txt, cosphi_LNNN.txt, cosphi_LNNN.csv; summary in checks/summary.txt
set -euo pipefail
R=${1:?run dir}; NPAR=${2:-4}; shift $(( $# >= 2 ? 2 : 1 ))
W=${W:?set W}
X=/lustre/orion/cli900/world-shared/projects/XNR1K/data/h3f7-ndjf/global/O8000/ml-dgg
TU=$X/u/u_131_h3f7-ndjf_000000.ml-dgg.O8000.global.grib; TV=$X/v/v_132_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
C=$R/checks; mkdir -p "$C"
if [ $# -gt 0 ]; then levels="$*"; else
  levels=$(grep -oE 'level [0-9]+ done in' "$R/log.txt" | awk '{print $2}' | sort -n | uniq)
fi
one() {   # one level: M8/sanity, then cos(lat) summary line; timing recorded
  L=$1; l=$(printf %03d "$L"); t0=$(date +%s)
  [ -s "$C/cosphi_L$l.csv" ] && return 0                        # done before
  PU=$R/parts/u_L$l.grib; PV=$R/parts/v_L$l.grib
  [ -s "$PU" ] && [ -s "$PV" ] || { echo "L$l: parts missing" >> "$C/summary.txt"; return 0; }
  srun -n 1 --overlap python3 "$W/analysis/check_recovered.py" --rec-u "$PU" --rec-v "$PV" --tpl-u "$TU" --tpl-v "$TV" \
       --level "$L" > "$C/check_L$l.txt" 2>&1 || true
  srun -n 1 --overlap python3 "$W/analysis/cosphi_check.py" --rec-u "$PU" --rec-v "$PV" --tpl-u "$TU" --tpl-v "$TV" \
       --levels "$L" --csv "$C/cosphi_L$l.csv" > "$C/cosphi_L$l.txt" 2>&1 || true
  m8=$(grep -c "PASS$" "$C/check_L$l.txt" || true)
  echo "L$l: M8 PASS $m8/2; $(grep -E '^ *[0-9]+ (u|v) ' "$C/cosphi_L$l.txt" | tr -s ' ' | tr '\n' ';') $(( $(date +%s) - t0 )) s" \
       >> "$C/summary.txt"
}
export -f one; export R C W TU TV
echo "# $(date -u +%FT%TZ) check_parts on $(hostname): levels $(echo $levels | wc -w), $NPAR in parallel" >> "$C/summary.txt"
printf '%s\n' $levels | xargs -P "$NPAR" -I{} bash -c 'one {}'
grep "^L" "$C/summary.txt" | sort
echo "M8 not 2/2: $(grep '^L' "$C/summary.txt" | grep -vc 'M8 PASS 2/2' || true) level(s)"
