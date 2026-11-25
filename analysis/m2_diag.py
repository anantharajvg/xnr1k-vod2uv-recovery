#!/usr/bin/env python3
"""Diagnose the residual of the ARCO -> F320 synthesis against the CDS (MIR) F320 vo, d.

Question: is the residual only the 16-bit packing of the CDS message, or does MIR's grid differ
from ours before packing?  Decisive test: quantise OUR synthesised field with the CDS message's own
packing (reference value R, step q):  R + round((x - R)/q) * q.  If MIR's unpacked field equals
ours to well below q, this reproduces the CDS values exactly at almost every point (mismatches
only where a value sits within rounding distance of a half-step tie).
Also reported: residual RMS against the white packing-noise expectation q/sqrt(12), the residual
distribution in units of q/2, its latitude structure, and its spectrum by total wavenumber.

  python3 m2_diag.py --arco A.npz --cds B.grib --level 74
"""
import argparse
import math

import numpy as np

import m2_era5 as M
import core as C


def band_table(P, T, edges=None):
    edges = edges or (0, 100, 300, 500, 600, 630) if T >= 639 else (0, T // 4, T // 2, 3 * T // 4, T - 9)
    tot = P.sum()
    n = np.arange(T + 1)
    cnt = 2 * n + 1                                  # coefficients per n: white noise share
    out = []
    for a, b in zip(edges, list(edges[1:]) + [T + 1]):
        out.append("n %3d-%3d: %5.1f%% (white noise %5.1f%%)" % (a, b - 1, 100 * P[a:b].sum() / tot,
                                                                100 * cnt[a:b].sum() / cnt.sum()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arco", required=True)
    p.add_argument("--cds", required=True)
    p.add_argument("--level", type=int, default=74)
    a = p.parse_args()
    z = np.load(a.arco, allow_pickle=False)
    T = int(z["T"])
    L = a.level
    cf = M.CdsFile(a.cds)
    grid = M.check_cds_grid(cf, [L])
    w = grid.area_weights()
    lat = np.repeat(grid.lat_deg, grid.nlon)
    lon = np.concatenate([360.0 * np.arange(n) / n for n in grid.nlon])
    for name in ("vo", "d"):
        A = M.arco_to_A(z[name][L - 1], T)
        ours = M.synth([A], grid, T)[0]
        y = cf.read(name, L)
        meta = cf.meta[(name, L)]
        q = 2 * cf.half_step(name, L)
        R = meta["referenceValue"]
        r = y - ours
        print("== level %d %s: bitsPerValue %d, step q %.3e, reference %.6e, field rms %.3e, range [%.3e, %.3e]"
              % (L, name, meta["bitsPerValue"], q, R, math.sqrt(np.sum(w * y * y)), y.min(), y.max()))
        rr = math.sqrt(np.sum(w * r * r))
        print("residual rms %.3e = %.2f x packing-noise rms q/sqrt(12) = %.3e; area-mean %.2e; max |r| %.3e = %.2f q/2"
              % (rr, rr / (q / math.sqrt(12)), q / math.sqrt(12), np.sum(w * r), abs(r).max(), abs(r).max() / (q / 2)))
        h = abs(r) / (q / 2)
        print("share of points with |r| <= 1, 2, 4, 8 half steps: %s" %
              ", ".join("%.5f" % np.mean(h <= k) for k in (1.000001, 2, 4, 8)))
        # decisive test
        qz = R + np.round((ours - R) / q) * q
        same = np.isclose(qz, y, rtol=0, atol=q * 1e-3)
        print("our field quantised with the CDS packing reproduces CDS at %.5f of points (%d differ)"
              % (same.mean(), (~same).sum()))
        k = int(np.argmax(abs(r)))
        print("largest residual at lat %.2f lon %.2f: CDS %.4e ours %.4e" % (lat[k], lon[k], y[k], ours[k]))
        print("residual rms by latitude band:")
        for lo_, hi_ in ((-90, -80), (-80, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 80), (80, 90)):
            sel = (lat >= lo_) & (lat < hi_ if hi_ < 90 else lat <= hi_)
            ws = w[sel]
            print("  %4d..%3d: %.3e (%.2f x q/sqrt12)" % (lo_, hi_, math.sqrt(np.sum(ws * r[sel] ** 2) / ws.sum()),
                                                         math.sqrt(np.sum(ws * r[sel] ** 2) / ws.sum()) / (q / math.sqrt(12))))
        # where in (m, n) is the residual?  Regress the residual's spectrum on the ARCO spectrum, per n:
        # slope ~ 0 = noise only; slope ~ -1 at some n = MIR left those coefficients out.
        Rs = M.analyse(r, grid, T)
        wm = M.mweight(T)
        def slope(sel_m, n):
            a, b = A[sel_m, n], Rs[sel_m, n]
            ww = wm[sel_m, 0]
            den = np.sum(ww * abs(a) ** 2)
            return (np.sum(ww * (b * np.conj(a)).real) / den if den > 0 else np.nan,
                    math.sqrt(np.sum(ww * abs(b) ** 2) / den) if den > 0 else np.nan)
        allm = np.arange(T + 1)
        print("residual spectrum regressed on the ARCO spectrum, per total wavenumber n (slope; residual/signal amplitude):")
        bins = [(0, T - 40), (T - 39, T - 20), (T - 19, T - 10)] + [(n, n) for n in range(T - 9, T + 1)]
        for a0, b0 in bins:
            num = den = res = 0.0
            for n in range(a0, b0 + 1):
                sel = allm <= n
                aa, bb, ww = A[sel, n], Rs[sel, n], wm[sel, 0]
                num += np.sum(ww * (bb * np.conj(aa)).real); den += np.sum(ww * abs(aa) ** 2); res += np.sum(ww * abs(bb) ** 2)
            print("  n %3d-%3d: slope %+.4f, |residual|/|signal| %.4f" % (a0, b0, num / den, math.sqrt(res / den)))
        sel = allm <= T
        print("  at n = T, by m (slope / |residual|/|signal|):", ", ".join("m %d-%d %+.3f / %.4f" % ((m0, m1) + slope((allm >= m0) & (allm <= m1), T))
                                          for m0, m1 in ((0, T // 6), (T // 6 + 1, T // 2), (T // 2 + 1, 5 * T // 6), (5 * T // 6 + 1, T))))
        # decisive re-test under the hypothesis that MIR leaves out n = T (and, separately, n >= T-1)
        for drop in (1, 2):
            A2 = A.copy()
            A2[:, T + 1 - drop:] = 0
            o2 = M.synth([A2], grid, T)[0]
            qz2 = R + np.round((o2 - R) / q) * q
            same2 = np.isclose(qz2, y, rtol=0, atol=q * 1e-3)
            r2 = y - o2
            print("with n > %d removed from ARCO: CDS reproduced at %.5f of points; residual rms %.2f x q/sqrt12, max %.2f q/2"
                  % (T - drop, same2.mean(), math.sqrt(np.sum(w * r2 * r2)) / (q / math.sqrt(12)), abs(r2).max() / (q / 2)))
        P = M.power_by_n(r, grid, T)
        print("residual power by total wavenumber:")
        for line in band_table(P, T):
            print("  " + line)


if __name__ == "__main__":
    main()
