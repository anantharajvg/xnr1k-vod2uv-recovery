#!/usr/bin/env python3
"""Test the hypothesis that the corrupted XNR1K u, v are the wind images u*cos(lat), v*cos(lat).

Seen in the level 74 zonal means (7 Oct): corrupted/recovered [u] follows cos(lat) within 0.01 between 75S and 75N.
Point-by-point test here, for one level:
  1. per 5-degree band: least-squares slope of corrupted on recovered, against the band's mean cos(lat)
     (also cos^2 as an alternative hypothesis);
  2. residual r = corrupted - recovered*cos(lat): RMS and max against the packing of both messages
     (expected if the hypothesis is exact: |r| <= q_c/2 + cos * q_r/2, plus the difference between our
     transform and the one that produced the corrupted file);
  3. the same residual by latitude band.
Two independent routes then meet: our recovery from vo, d, and the corrupted field divided by cos(lat).

  srun -n 1 python3 cosphi_check.py --rec-u R_u.grib --rec-v R_v.grib --tpl-u U.grib --tpl-v V.grib --level 74
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recover"))
sys.path.insert(0, HERE)
import eccodes as ec                  # noqa: E402
import core as C                      # noqa: E402
from check_recovered import recovered_message, template_message   # noqa: E402
import m2_era5 as M                   # noqa: E402  (parse_levels)
import csv                            # noqa: E402


def decode(msg, level=None):
    h = ec.codes_new_from_message(msg)
    try:
        if level is not None and int(ec.codes_get(h, "level")) != level:
            raise SystemExit("message holds level %d, expected %d" % (ec.codes_get(h, "level"), level))
        q = 2.0 ** ec.codes_get(h, "binaryScaleFactor") / 10.0 ** ec.codes_get(h, "decimalScaleFactor")
        return np.asarray(ec.codes_get_values(h), dtype=np.float64), q
    finally:
        ec.codes_release(h)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("rec-u", "rec-v", "tpl-u", "tpl-v"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--level", type=int, default=None, help="one level, full band table")
    p.add_argument("--levels", default=None, help="e.g. 1-40 or all: one summary line per level")
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    grid = C.GaussGrid.octahedral(8000)
    latr = np.radians(np.repeat(grid.lat_deg, grid.nlon))
    cosl = np.cos(latr)
    w = grid.area_weights()
    latd = np.degrees(latr)
    if a.levels:
        summary(a, grid, cosl, w, latd)
        return
    level = a.level or 74
    for var in ("u", "v"):
        rec, qr = decode(recovered_message(getattr(a, "rec_" + var), level), level)
        cor, qc = decode(template_message(getattr(a, "tpl_" + var), level), level)
        r = cor - rec * cosl
        bound = qc / 2 + cosl * qr / 2
        print("== %s level %d: packing step corrupted %.2e, recovered %.2e" % (var, level, qc, qr))
        print("residual corrupted - recovered*cos(lat): rms %.3e, max %.3e, area-mean %+.2e; rms of corrupted %.3f"
              % (math.sqrt(np.sum(w * r * r)), abs(r).max(), np.sum(w * r), math.sqrt(np.sum(w * cor * cor))))
        print("share of points within the packing bound q_c/2 + cos*q_r/2: %.5f; within 2x: %.5f; within 10x: %.5f"
              % (np.mean(abs(r) <= bound * 1.0001), np.mean(abs(r) <= 2 * bound), np.mean(abs(r) <= 10 * bound)))
        k = int(np.argmax(abs(r)))
        print("largest residual at lat %.3f: corrupted %.4f, recovered %.4f, recovered*cos %.4f"
              % (latd[k], cor[k], rec[k], rec[k] * cosl[k]))
        print("band     slope(corr on rec)  mean cos   mean cos^2   resid rms   resid max")
        for lo in range(-90, 90, 5):
            sel = (latd >= lo) & (latd < lo + 5) if lo + 5 < 90 else (latd >= lo)
            ws, x, y = w[sel], rec[sel], cor[sel]
            slope = np.sum(ws * x * y) / np.sum(ws * x * x)
            mc = np.sum(ws * cosl[sel]) / ws.sum()
            mc2 = np.sum(ws * cosl[sel] ** 2) / ws.sum()
            rs = r[sel]
            print("%4d..%3d  %8.5f           %8.5f   %8.5f     %.2e    %.2e"
                  % (lo, lo + 5, slope, mc, mc2, math.sqrt(np.sum(ws * rs * rs) / ws.sum()), abs(rs).max()))


def summary(a, grid, cosl, w, latd):
    """One line per level and variable: residual RMS / pure-packing expectation, share within 2x the bound,
    max residual in packing steps, and the worst 5-degree-band slope/cos ratio between 80S and 80N."""
    levels = M.parse_levels(a.levels, 137)
    mid = abs(latd) < 80
    c2 = np.sum(w * cosl ** 2)
    rows = []
    print("level var  rms/packing  within2x   max/q_c   worst |slope/cos - 1| (|lat|<80)")
    for L in levels:
        for var in ("u", "v"):
            rec, qr = decode(recovered_message(getattr(a, "rec_" + var), L), L)
            cor, qc = decode(template_message(getattr(a, "tpl_" + var), L), L)
            r = cor - rec * cosl
            bound = qc / 2 + cosl * qr / 2
            expect = math.sqrt(qc * qc / 12 + c2 * qr * qr / 12)
            ratio = math.sqrt(np.sum(w * r * r)) / expect
            w2 = float(np.mean(abs(r) <= 2 * bound))
            worst = 0.0
            for lo in range(-80, 80, 5):
                sel = (latd >= lo) & (latd < lo + 5)
                ws, x, y = w[sel], rec[sel], cor[sel]
                slope = np.sum(ws * x * y) / np.sum(ws * x * x)
                mc = np.sum(ws * cosl[sel] * x * x) / np.sum(ws * x * x)     # x^2-weighted cos: what the slope estimates
                worst = max(worst, abs(slope / mc - 1))
            print("%5d %-3s  %8.3f    %8.5f  %8.2f   %.2e" % (L, var, ratio, w2, abs(r[mid]).max() / qc, worst))
            rows.append(dict(level=L, var=var, rms_over_packing=ratio, within2x=w2,
                             max_over_qc_lat80=abs(r[mid]).max() / qc, worst_slope_dev=worst, q_c=qc, q_r=qr))
            if a.csv:
                with open(a.csv + ".tmp", "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                    wr.writeheader()
                    wr.writerows(rows)
                os.replace(a.csv + ".tmp", a.csv)


if __name__ == "__main__":
    main()
