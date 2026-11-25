#!/usr/bin/env python3
"""Post-run checks of a recovered XNR1K level (native O8000 GRIB), against the corrupted template.

  srun -n 1 python3 check_recovered.py --rec-u R_u.grib --rec-v R_v.grib --tpl-u U.grib --tpl-v V.grib --level 74

1. M8 (plan): header keys in the mars, parameter, time, vertical, geography namespaces (array keys whole),
   plus packingType, bitsPerValue, numberOfValues, identical between recovered and template message.
2. Sanity: 256,288,000 finite values, 16 bits, value ranges.
3. Recovered against corrupted: zonal-mean u, v per latitude band, and the ratio corrupted / recovered of
   zonal-mean u where |recovered| > 2 m/s (the findings of 7 Oct inferred 0.4-0.7 from ERA5).
Reads only; decodes 4 fields of 256 M points (about 8 GB of memory).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recover"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recover", "tests"))
import eccodes as ec          # noqa: E402
import core as C              # noqa: E402
import gribio as G            # noqa: E402
from grib_selftest import header   # noqa: E402  (same M8 key set as the self-test)


def single_message(path):
    offs = G.scan_offsets(path)
    if len(offs) != 1:
        raise SystemExit("%s: expected 1 message, found %d" % (path, len(offs)))
    off, n = offs[0]
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def template_message(path, level):
    gf = G.GribFile(path)
    return gf.raw(level - 1)


def recovered_message(path, level):
    """Single-level file (e.g. ...recovered.L074.grib) or the merged all-level file (ordered by level)."""
    offs = G.scan_offsets(path)
    if len(offs) == 1:
        return single_message(path)
    return G.GribFile(path).raw(level - 1)


def decode(msg):
    h = ec.codes_new_from_message(msg)
    try:
        return header(h), np.asarray(ec.codes_get_values(h), dtype=np.float64), int(ec.codes_get(h, "level"))
    finally:
        ec.codes_release(h)


def zonal_means(x, grid):
    return np.add.reduceat(x, grid.offsets[:-1]) / grid.nlon


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("rec-u", "rec-v", "tpl-u", "tpl-v"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--level", type=int, default=74)
    a = p.parse_args()
    ok = True
    grid = C.GaussGrid.octahedral(8000)
    lat = grid.lat_deg
    fields = {}
    for var in ("u", "v"):
        hr, r, Lr = decode(recovered_message(getattr(a, "rec_" + var), a.level))
        ht, t, Lt = decode(template_message(getattr(a, "tpl_" + var), a.level))
        diff = sorted(k for k in set(hr) | set(ht) if hr.get(k) != ht.get(k))
        good = (not diff and Lr == Lt == a.level and r.size == grid.npts and np.isfinite(r).all()
                and hr.get("bitsPerValue") == "16")
        ok &= good
        print("%s level %d: M8 header keys compared %d, differing: %s; values %d, finite %s, bitsPerValue %s -> %s"
              % (var, Lr, len(ht), diff or "none", r.size, bool(np.isfinite(r).all()), hr.get("bitsPerValue"),
                 "PASS" if good else "FAIL"))
        print("   recovered  range [%.3f, %.3f]  mean %.4f" % (r.min(), r.max(), r.mean()))
        print("   corrupted  range [%.3f, %.3f]  mean %.4f" % (t.min(), t.max(), t.mean()))
        fields[var] = (r, t)
    zr, zt = zonal_means(fields["u"][0], grid), zonal_means(fields["u"][1], grid)
    vr, vt = zonal_means(fields["v"][0], grid), zonal_means(fields["v"][1], grid)
    print("zonal means by 10-degree band: lat, [u] recovered, [u] corrupted, ratio, [v] recovered, [v] corrupted")
    for lo in range(-90, 90, 10):
        sel = (lat >= lo) & (lat < lo + 10)
        ur, ut = zr[sel].mean(), zt[sel].mean()
        ratio = ut / ur if abs(ur) > 2 else float("nan")
        print("  %4d..%4d  %8.2f  %8.2f  %6.3f  %7.3f  %7.3f" % (lo, lo + 10, ur, ut, ratio, vr[sel].mean(), vt[sel].mean()))
    sel = np.abs(zr) > 2
    rat = zt[sel] / zr[sel]
    print("ratio corrupted/recovered [u], rows with |[u]| > 2 m/s: median %.3f, 10-90%% %.3f..%.3f (%d of %d rows)"
          % (np.median(rat), np.percentile(rat, 10), np.percentile(rat, 90), sel.sum(), sel.size))
    print("check_recovered:", "PASS (M8 and sanity)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
