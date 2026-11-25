#!/usr/bin/env python3
"""M3: the reduced-grid input path on realistic spectra (ERA5 T639 from ARCO), no packing involved.

Plan test M3, as amended 7 Oct before it was first run (owner's decision): O320 cannot hold T639 (intrinsic
error 8e-4 to 1e-2 measured on synthetic spectra), so
  * formal:  O640 -- the octahedral grid IFS uses with T639; same family and same (cubic) row-limit rule as
             XNR1K's O8000.  Criterion unchanged: relative RMS of (u, v) <= 1e-6 on every level.
  * report:  N320 -- ERA5's classic reduced grid (pl from the ecCodes sample reduced_gg_pl_320_grib2).
Per level: synthesise the ARCO vo, d (our convention, verified by the calibration) exactly on the reduced
grid, run our grid-input pipeline (recover_serial) to F320, compare with u, v synthesised exactly on F320
from the same spectra (the M2a field, which matches ECMWF's u, v at packing level).  This checks the
reduced-grid path only; it is not independent of our code.

  python3 m3_reduced.py --arco A.npz --levels all --csv m3_all.csv
  python3 m3_reduced.py --selftest          # T63: O64 must pass, O32 (too coarse) must not
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recover"))
sys.path.insert(0, HERE)
import core as C            # noqa: E402
import serial as S          # noqa: E402
import m2_era5 as M         # noqa: E402

CRIT = 1e-6


def n320_grid():
    import eccodes as ec
    h = ec.codes_grib_new_from_samples("reduced_gg_pl_320_grib2")
    try:
        N = int(ec.codes_get(h, "N"))
        pl = np.asarray(ec.codes_get_array(h, "pl"), dtype=np.int64)
    finally:
        ec.codes_release(h)
    if N != 320 or pl.size != 640:
        raise SystemExit("ecCodes sample is not N320 (N=%d, %d rows)" % (N, pl.size))
    return C.GaussGrid.from_pl(320, pl, name="N320 (ecCodes sample)")


def compare(Z, D, grid_in, F, T, w, uref, vref):
    t = time.time()
    vo, d = M.synth([Z, D], grid_in, T)
    u, v = S.recover_serial(vo, d, grid_in, F, T)
    num = np.sum(w * ((u - uref) ** 2 + (v - vref) ** 2))
    rel = math.sqrt(num / np.sum(w * (uref ** 2 + vref ** 2)))
    mx = float(max(abs(u - uref).max(), abs(v - vref).max()))
    return rel, mx, time.time() - t


def run(levels, z, T, grids, csv_path=None):
    F = C.GaussGrid.regular((T + 1) // 2)                            # F320 for T639 (linear grid: 2N - 1 = T)
    w = F.area_weights()
    rows, ok = [], True
    for L in levels:
        Z = M.arco_to_A(z["vo"][L - 1], T)
        D = M.arco_to_A(z["d"][L - 1], T)
        uref, vref = M.synth([Z, D], F, T, uv=True)
        for gname, G, formal in grids:
            rel, mx, dt = compare(Z, D, G, F, T, w, uref, vref)
            verdict = ("PASS" if rel <= CRIT else "FAIL") if formal else "report"
            ok &= (verdict != "FAIL")
            print("L%03d M3 %-22s rel RMS (u,v) %.2e  max |diff| %.2e m/s  (criterion %s) %s  [%.0fs]"
                  % (L, gname, rel, mx, "<= %.0e" % CRIT if formal else "none", verdict, dt))
            rows.append(dict(level=L, grid=gname, formal=formal, rel=rel, maxabs=mx, verdict=verdict))
        if csv_path:
            with open(csv_path + ".tmp", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
            os.replace(csv_path + ".tmp", csv_path)
    print("M3:", "PASS" if ok else "FAIL", "(formal grids, %d levels)" % len(levels))
    return ok


def selftest():
    """T63 analogue: O64 (as O640 for T639) must pass; O32 (as O320) must fail; scaled-down N grid reported."""
    T = 63
    rng = np.random.default_rng(7)
    n = np.arange(T + 1)
    def rA():
        A = np.triu(rng.standard_normal((T + 1, T + 1)) + 1j * rng.standard_normal((T + 1, T + 1)))
        A = A * (1 + n[None, :]) ** -0.5 * 1e-5
        A[0] = A[0].real
        A[0, 0] = 0
        return A
    Z, D = rA(), rA()
    vo = np.stack([M.A_to_arco(Z)])
    d = np.stack([M.A_to_arco(D)])
    ok_good = run([1], {"vo": vo, "d": d}, T, [("O64", C.GaussGrid.octahedral(64), True)])
    ok_bad = run([1], {"vo": vo, "d": d}, T, [("O32 (too coarse)", C.GaussGrid.octahedral(32), True)])
    ok = ok_good and not ok_bad
    print("selftest:", "PASS" if ok else "FAIL", "(O64 passes: %s; O32 is refused: %s)" % (ok_good, not ok_bad))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arco")
    p.add_argument("--levels", default="74")
    p.add_argument("--csv")
    p.add_argument("--no-n320", action="store_true", help="skip the report-only N320 run")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.arco:
        p.error("--arco is required")
    z = np.load(a.arco, allow_pickle=False)
    T = int(z["T"])
    if str(z["time"])[:13] != "2018-11-01T00":
        raise SystemExit("ARCO file time is %s, expected 2018-11-01T00" % z["time"])
    grids = [("O640 (formal)", C.GaussGrid.octahedral(T + 1), True)]
    if not a.no_n320:
        grids.append(("N320 (report)", n320_grid(), False))
    G = grids[0][1]
    print("ARCO T%d; formal grid %s, row limits %d..%d; output F%d" % (T, G.name, G.nmen(T).min(), G.nmen(T).max(), (T + 1) // 2))
    levels = M.parse_levels(a.levels, z["vo"].shape[0])
    sys.exit(0 if run(levels, z, T, grids, a.csv) else 1)


if __name__ == "__main__":
    main()
