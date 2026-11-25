"""M1: analytic fields on synthetic octahedral (and regular) grids.

Fields: Rossby-Haurwitz waves (rotational) plus a divergent analogue, zonal
wavenumber R, degree R+1, plus solid-body rotation and a Y_1^0 divergent part
(Williamson et al. 1992, test 6, extended).  Exact u, v are known in closed form.
Pass: max |error| / max |V| <= 3e-7 (criterion fixed in the plan; double
precision here, so the expected error is far smaller).
"""
import os, sys, time
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(D))
import core as C, serial as S

A = 6371229.0


def fields(lat, lon, R, om=7.848e-6, K=7.848e-6, Dd=3.0e-6, ph=0.3, base=True):
    c, s = np.cos(lat), np.sin(lat)
    b = 1.0 if base else 0.0
    L = R * (lon - ph)
    u = b * A * om * c + A * K * c ** (R - 1) * (R * s ** 2 - c ** 2) * np.cos(L)
    v = -A * K * R * c ** (R - 1) * s * np.sin(L)
    vo = b * 2 * om * s - K * s * c ** R * (R + 1) * (R + 2) * np.cos(L)
    u += A * Dd * R * c ** (R - 1) * s * np.cos(L)
    v += A * Dd * (c ** (R + 1) - R * c ** (R - 1) * s ** 2) * np.sin(L) + b * A * Dd * c
    d = -Dd * (R + 1) * (R + 2) * c ** R * s * np.sin(L) - b * 2 * Dd * s
    return vo, d, u, v


def run(gin, gout, lmax, waves, label, expect_from=None):
    la_i, lo_i = S.point_coords(gin)
    la_o, lo_o = S.point_coords(gout)
    keep = expect_from if expect_from is not None else waves
    fin = [fields(la_i, lo_i, R, base=(i == 0)) for i, R in enumerate(waves)]
    fout = [fields(la_o, lo_o, R, base=(i == 0)) for i, R in enumerate(keep)]
    vo = sum(f[0] for f in fin); d = sum(f[1] for f in fin)
    ue = sum(f[2] for f in fout); ve = sum(f[3] for f in fout); voe = sum(f[0] for f in fout)
    t = time.time()
    u, v, vo_t, d_t = S.recover_serial(vo, d, gin, gout, lmax, A, want_vod=True)
    dt = time.time() - t
    sc = max(abs(ue).max(), abs(ve).max())
    eu, ev = abs(u - ue).max() / sc, abs(v - ve).max() / sc
    evo = abs(vo_t - voe).max() / abs(voe).max()
    ok = max(eu, ev) <= 3e-7
    print("%-34s lmax=%4d waves=%-10s u %.1e  v %.1e  vo(trunc) %.1e  [%5.1fs]  %s"
          % (label, lmax, waves, eu, ev, evo, dt, "PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ok = True
    ok &= run(C.GaussGrid.octahedral(48), C.GaussGrid.octahedral(48), 47, (10,), "O48 -> O48")
    ok &= run(C.GaussGrid.octahedral(48), C.GaussGrid.octahedral(48), 47, (40,), "O48 -> O48 (m near row limits)")
    ok &= run(C.GaussGrid.octahedral(160), C.GaussGrid.octahedral(160), 159, (30, 150), "O160 -> O160")
    ok &= run(C.GaussGrid.octahedral(160), C.GaussGrid.regular(160), 159, (30, 150), "O160 -> F160 (other grid)")
    ok &= run(C.GaussGrid.octahedral(160), C.GaussGrid.regular(40), 79, (30, 150), "O160 -> F40 at T79 (truncation)",
              expect_from=(30,))
    ok &= run(C.GaussGrid.regular(64), C.GaussGrid.octahedral(64), 63, (50,), "F64 -> O64")
    if len(sys.argv) > 1 and sys.argv[1] == "big":
        ok &= run(C.GaussGrid.octahedral(640), C.GaussGrid.octahedral(640), 639, (60, 600), "O640 -> O640")
    print("M1:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
