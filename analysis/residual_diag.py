#!/usr/bin/env python3
"""Shape of the cos(lat) residual r = corrupted - recovered*cos(lat) at one level, in units of the archive's packing step.

Distinguishes candidate sources of the non-packing part (work log 8 Oct):
  - pure round-to-nearest packing of both files: r/q_c roughly uniform on [-1/2, 1/2] plus a little, mean ~0;
  - truncating (floor) packing in the archive: mean near +-q_c/2;
  - double quantisation in the archive (e.g. step 2 q_c somewhere upstream): flat-topped, width ~2 q_c;
  - transform/input-packing noise: smooth bell, structure in latitude and in scale (zonal-mean share).
Prints mean, RMS, kurtosis, a histogram of r/q_c, RMS per 10-degree band, and the share of residual variance that
is zonal-mean. Reads one part file per variable and the template; ~8 GB memory.

  srun -n 1 python3 residual_diag.py --rec-u parts/u_L037.grib --rec-v parts/v_L037.grib --tpl-u U --tpl-v V --level 37
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recover"))
sys.path.insert(0, HERE)
import core as C                                                   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("rec-u", "rec-v", "tpl-u", "tpl-v"):
        p.add_argument("--" + k)
    p.add_argument("--level", type=int)
    p.add_argument("--selftest", action="store_true", help="synthetic cases on O64, no ecCodes")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not all((a.rec_u, a.rec_v, a.tpl_u, a.tpl_v, a.level)):
        p.error("--rec-u --rec-v --tpl-u --tpl-v --level are required")
    grid = C.GaussGrid.octahedral(8000)
    latr = np.radians(np.repeat(grid.lat_deg, grid.nlon))
    cosl = np.cos(latr)
    w = grid.area_weights()
    latd = np.degrees(latr)
    from check_recovered import recovered_message, template_message    # ecCodes needed only here
    from cosphi_check import decode
    for var in ("u", "v"):
        rec, qr = decode(recovered_message(getattr(a, "rec_" + var), a.level), a.level)
        cor, qc = decode(template_message(getattr(a, "tpl_" + var), a.level), a.level)
        r = cor - rec * cosl
        del rec, cor
        print("== %s level %d: q_c %.3e, q_r %.3e" % (var, a.level, qc, qr))
        analyse(r, qc, grid, w, latd)
        del r


def analyse(r, qc, grid, w, latd):
    """Statistics of r / qc; returns (mean, rms, kurtosis, zonal-mean share) for tests."""
    edges = np.arange(-3.0, 3.01, 0.25)
    x = r / qc
    mean = float(np.sum(w * x))
    var_ = float(np.sum(w * (x - mean) ** 2))
    kurt = float(np.sum(w * (x - mean) ** 4)) / var_ ** 2
    rms = math.sqrt(var_ + mean * mean)
    print("r/q_c area-mean %+.4f, rms %.4f, kurtosis %.2f (uniform 1.8, Gaussian 3.0)" % (mean, rms, kurt))
    h, _ = np.histogram(x, bins=edges)
    print("histogram of r/q_c (points), bins of 0.25 from -3 to 3; below %d, above %d"
          % (np.sum(x < edges[0]), np.sum(x >= edges[-1])))
    for lo, c in zip(edges[:-1], h):
        print("  %+5.2f..%+5.2f  %10d  %s" % (lo, lo + 0.25, c, "#" * int(60 * c / max(h.max(), 1))))
    zm = np.add.reduceat(r, grid.offsets[:-1]) / grid.nlon
    zfull = np.repeat(zm, grid.nlon)
    share = float(np.sum(w * zfull ** 2) / np.sum(w * r * r))
    print("share of residual variance in the zonal mean: %.4f" % share)
    print("band      rms(r)/q_c   mean(r)/q_c")
    for lo in range(-90, 90, 10):
        sel = (latd >= lo) & (latd < lo + 10) if lo < 80 else (latd >= lo)
        ws = w[sel] / w[sel].sum()
        print("  %4d..%3d  %7.3f      %+7.3f" % (lo, lo + 10, math.sqrt(np.sum(ws * x[sel] ** 2)), np.sum(ws * x[sel])))
    return mean, rms, kurt, share


def selftest():
    grid = C.GaussGrid.octahedral(64)
    latd = np.repeat(grid.lat_deg, grid.nlon)
    w = grid.area_weights()
    rng = np.random.default_rng(1)
    n = grid.npts
    ok = True
    cases = [("uniform rounding", rng.uniform(-0.5, 0.5, n), lambda m, r, k, z: abs(m) < 0.01 and abs(r - 0.2887) < 0.01 and abs(k - 1.8) < 0.05),
             ("floor (truncating)", rng.uniform(0, 1, n), lambda m, r, k, z: abs(m - 0.5) < 0.01),
             ("Gaussian", rng.normal(0, 1, n), lambda m, r, k, z: abs(k - 3) < 0.1),
             ("zonal stripe", np.repeat(np.sin(np.radians(grid.lat_deg) * 3), grid.nlon), lambda m, r, k, z: z > 0.999)]
    for name, r, test in cases:
        print("--- selftest:", name)
        got = analyse(r, 1.0, grid, w, latd)
        good = test(*got)
        ok &= good
        print("    ->", "PASS" if good else "FAIL")
    print("--- negative control: the uniform-rounding test must reject Gaussian data")
    m, r, k, z = analyse(rng.normal(0, 0.2887, n), 1.0, grid, w, latd)
    neg = not cases[0][2](m, r, k, z)
    ok &= neg
    print("    ->", "PASS (rejected)" if neg else "FAIL (accepted)")
    print("residual_diag selftest:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
