#!/usr/bin/env bash
# provenance.sh -- write a provenance manifest for the vod2uv.recovery work on Orion.
#
#   bash scripts/provenance.sh                 # code + ERA5 + results: md5; XNR1K inputs: size, mtime
#   bash scripts/provenance.sh --hash-inputs   # also md5 of the XNR1K 000000 GRIB inputs (~70 GB each, slow)
#
# Output: results/provenance_<UTC timestamp>.txt (never overwrites; read-only access to all inputs).
# Records: host, time, loaded modules, Python package versions, the MPI library the mpi4py
# extension links to, checksums of every code/doc file, and checksums or sizes of data files.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="${W:-$HERE}"
X="${X:-/lustre/orion/cli900/world-shared/projects/XNR1K/data/h3f7-ndjf/global/O8000/ml-dgg}"
STEP="${STEP:-000000}"
HASH_INPUTS=0
[ "${1:-}" = "--hash-inputs" ] && HASH_INPUTS=1
mkdir -p "$W/results"
OUT="$W/results/provenance_$(date -u +%Y%m%dT%H%M%SZ).txt"
[ -e "$OUT" ] && { echo "exists: $OUT" >&2; exit 1; }

{
echo "# vod2uv.recovery provenance manifest"
echo "written_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host: $(hostname)"
echo "working_dir: $W"
echo
echo "## modules"
(module -t list 2>&1 || echo "module command not available") | sed 's/^/  /'
echo
echo "## python"
python3 - <<'PY'
import importlib, sys, glob, os, subprocess
print("  python", sys.version.split()[0], sys.executable)
for name in ("numpy", "scipy", "mpi4py", "netCDF4", "eccodes", "matplotlib", "xarray", "zarr", "gcsfs", "cdsapi"):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", None)
        if name == "eccodes":
            v = "%s (library %s)" % (v, m.codes_get_api_version())
        print("  %-10s %s" % (name, v))
    except Exception as e:  # noqa: BLE001
        print("  %-10s not importable (%s)" % (name, type(e).__name__))
try:
    import mpi4py
    so = glob.glob(os.path.join(os.path.dirname(mpi4py.__file__), "MPI*.so"))[0]
    libs = subprocess.run(["ldd", so], capture_output=True, text=True).stdout
    for line in libs.splitlines():
        if "mpi" in line.lower() or "open-" in line:
            print("  mpi4py links:", line.strip())
except Exception as e:  # noqa: BLE001
    print("  mpi4py linkage: unknown (%s)" % type(e).__name__)
PY
echo
echo "## code and documents (md5)"
( cd "$W" && find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.txt' -o -name '*.csv' \) \
    -not -path './.venv/*' -not -path '*/__pycache__/*' -not -path './results/provenance_*' -print0 \
  | sort -z | xargs -0 md5sum ) | sed 's/^/  /'
echo
echo "## ERA5 data (md5, bytes, mtime)"
for f in "$W"/data/era5/*; do
  [ -f "$f" ] || continue
  printf '  %s  %s  %s  %s\n' "$(md5sum < "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" "$(stat -c %y "$f")" "$f"
done
echo
echo "## XNR1K inputs, step $STEP (read-only)"
for v in u v vo d; do
  for f in "$X/$v/"*"_${STEP}."*.grib; do
    [ -f "$f" ] || { echo "  missing: $X/$v/*_${STEP}.*.grib"; continue; }
    h="(not hashed; use --hash-inputs)"
    [ $HASH_INPUTS -eq 1 ] && h="$(md5sum < "$f" | cut -d' ' -f1)"
    printf '  %s  %s  %s  %s\n' "$h" "$(stat -c %s "$f")" "$(stat -c %y "$f")" "$f"
  done
done
echo
echo "## results and run outputs (md5, bytes)"
( cd "$W" && shopt -s nullglob && dirs=(results run_*) && \
  { find "${dirs[@]}" -type f -not -name 'provenance_*' -print0 2>/dev/null || true; } | sort -z \
  | while IFS= read -r -d '' f; do printf '  %s  %s  %s\n' "$(md5sum < "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" "$f"; done )
} > "$OUT.tmp"
mv -n "$OUT.tmp" "$OUT"
echo "wrote $OUT"
