#!/usr/bin/env python3
"""M2 on ERA5: calibrate the ARCO spectral convention, then compare our u, v with ECMWF's.

Inputs (one time, 2018-11-01 00 UTC):
  --arco  npz from era5/fetch_arco.py: vo, d T639 spectral coefficients (137, 410240) float32,
          m-major, (re, im) interleaved -- layout verified on the data (m = 0 imaginary parts are 0)
  --cds   GRIB from era5/download_era5.py request A: vo, d, u, v on F320 (MIR, from the same spectra)

Steps per level
  1. Calibration (decides the convention, does NOT test the u/v computation):
     synthesise the ARCO vo (and d) on F320 with our convention, split into three parts
     (m = 0, even m > 0, odd m), and fit  CDS = a0*g0 + ae*ge + ao*go  (area-weighted least squares).
     Our convention gives a0 = ae = ao = 1. A factor 2 for m > 0, or a (-1)^m phase, shows up as
     ae, ao = 2 or -1. The fitted values must sit on a simple candidate (+-1, +-2, +-1/2, +-sqrt2, +-1/sqrt2)
     and the residual must be at the GRIB packing level poleward of 30 deg (RMS <= 1.05 x q/sqrt(12); decision of
     7 Oct: the gate checks the convention only, MIR's tropical n = T excess is reported); otherwise it stops.
  2. M2a: u, v from the calibrated ARCO spectra (our vod2uv in spectral space + synthesis on F320)
     against CDS u, v. Tests our vod2uv formula against MIR's.
  3. M2b = plan test M2: u, v from the CDS gridded vo, d through our full grid-input pipeline
     (recover_serial, the path used for XNR1K) against CDS u, v, and against M2a.
     Plan criterion (fixed before results): RMS diff <= q and max diff <= 4q on every level,
     q = packing step of the CDS u (v) message.  M2a is extra and reported only.
  M4 (plan, amended 7 Oct before first run): our Gauss-Legendre analysis of the CDS F320 vo, d against the
     calibrated ARCO coefficients; the difference (projected to the grid) must not exceed the CDS packing noise
     projected the same way: ratio <= 1.05 poleward of 30 deg. (Original 1e-4 relative RMS is below the packing.)
  Reported for every comparison: bias, RMS difference, max |difference|, RMS of the reference,
  regression slope, half packing step of the CDS field, and the share of the difference power at the
  highest total wavenumbers (truncation-type differences concentrate there).

  python3 m2_era5.py --arco A.npz --cds B.grib --levels 74 --csv m2_L074.csv
  python3 m2_era5.py --selftest          # no files, no ecCodes: checks the calibration logic
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
import core as C        # noqa: E402
import serial as S      # noqa: E402

RADIUS = 6371229.0
CANDIDATES = (1.0, -1.0, 2.0, -2.0, 0.5, -0.5, math.sqrt(2), -math.sqrt(2), 1 / math.sqrt(2), -1 / math.sqrt(2))
MB = 8


# ----------------------------------------------------------------------------- spectral helpers
def arco_to_A(x, T):
    """(T+1)(T+2) reals, m-major, (re, im) interleaved -> A[m, n] complex (T+1, T+1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size != (T + 1) * (T + 2):
        raise ValueError("expected %d values for T%d, got %d" % ((T + 1) * (T + 2), T, x.size))
    A = np.zeros((T + 1, T + 1), np.complex128)
    i = 0
    for m in range(T + 1):
        cnt = T + 1 - m
        seg = x[2 * i:2 * (i + cnt)]
        A[m, m:] = seg[0::2] + 1j * seg[1::2]
        i += cnt
    return A


def A_to_arco(A):
    T = A.shape[0] - 1
    out = []
    for m in range(T + 1):
        z = A[m, m:]
        out.append(np.stack((z.real, z.imag), 1).ravel())
    return np.concatenate(out)


def synth(As, grid, T, uv=False, mkeep=None, radius=RADIUS):
    """Synthesise on `grid` from spectral A[m, n] (list).  uv=False: each A is a scalar field.
    uv=True: As = [zeta, delta] -> returns [u, v].  mkeep: boolean (T+1,) mask of m to include."""
    mu = grid.mu_n
    s = np.sqrt((1 - mu) * (1 + mu))
    logs = np.log(s)
    mmax = int(min(T, grid.nmen(T).max()))
    nf = 2 if uv else len(As)
    G = np.zeros((nf, grid.nlat, mmax + 1), np.complex128)
    for b0 in range(0, mmax + 1, MB):
        mv = np.arange(b0, min(b0 + MB, mmax + 1))
        if mkeep is not None and not mkeep[mv].any():
            continue
        nk = T - int(mv.min()) + 1
        coef = np.zeros((len(As), len(mv), nk), np.complex128)
        for c, m in enumerate(mv):
            if mkeep is not None and not mkeep[m]:
                continue
            for f, A in enumerate(As):
                coef[f, c, :T + 1 - m] = A[m, m:]
        if uv:
            cP, cH = C.vod_to_uv_coeffs(mv, T, coef[0], coef[1], radius)
            xN, xS = C.synthesis(mv, T, mu, logs, cP, cH)
            xN /= s
            xS /= s
        else:
            xN, xS = C.synthesis(mv, T, mu, logs, coef, None)
        for f in range(nf):
            G[f][np.ix_(grid.rowN, mv)] = xN[f].T
            G[f][np.ix_(grid.rowS, mv)] = xS[f].T
    nm = grid.nmen(T)
    return [C.fourier_to_rows(G[f], grid, 0, grid.nlat, nm) for f in range(nf)]


def analyse(x, grid, T):
    """Gauss-Legendre analysis of a grid field -> A[m, n] complex (T+1, T+1), our convention."""
    mu, w = grid.mu_n, grid.w_n
    logs = np.log(np.sqrt((1 - mu) * (1 + mu)))
    nm = grid.nmen(T)
    mmax = int(min(T, nm.max()))
    F = C.rows_to_fourier(x, grid, 0, grid.nlat, nm, mmax)
    A = np.zeros((T + 1, T + 1), np.complex128)
    for b0 in range(0, mmax + 1, MB):
        mv = np.arange(b0, min(b0 + MB, mmax + 1))
        FN = F[grid.rowN][:, mv].T[None]
        FS = F[grid.rowS][:, mv].T[None]
        a = C.analysis(mv, T, mu, logs, 0.5 * w, FN, FS)[0]          # (nm, nk), n = m + k
        for c, m in enumerate(mv):
            A[m, m:] = a[c, :T + 1 - m]
    return A


def mweight(T):
    m = np.arange(T + 1)
    return np.where(m == 0, 1.0, 2.0)[:, None]                     # m > 0 stands for +m and -m


def power_by_n(x, grid, T):
    """Spectral power of a grid field by total wavenumber n = 0..T (m > 0 counted twice)."""
    return (np.abs(analyse(x, grid, T)) ** 2 * mweight(T)).sum(0)


# ----------------------------------------------------------------------------- statistics
def wstats(x, ref, w):
    d = x - ref
    rms_ref = math.sqrt(np.sum(w * ref * ref))
    slope = np.sum(w * x * ref) / np.sum(w * x * x)          # ref ~ slope * x
    return dict(bias=float(np.sum(w * d)), rmsd=math.sqrt(np.sum(w * d * d)), maxabs=float(abs(d).max()),
                rms_ref=rms_ref, rel=math.sqrt(np.sum(w * d * d)) / rms_ref, slope=float(slope))


def calibrate(y, parts, w):
    X = np.stack(parts, 1)
    XtW = X.T * w
    a = np.linalg.solve(XtW @ X, XtW @ y)
    r = y - X @ a
    snapped = [min(CANDIDATES, key=lambda c: abs(c - ai)) for ai in a]
    return a, snapped, math.sqrt(np.sum(w * r * r)) / math.sqrt(np.sum(w * y * y)), float(abs(r).max())


# ----------------------------------------------------------------------------- GRIB input
class CdsFile:
    """Index of a multi-variable GRIB file by (shortName, level); checks one message per key."""

    def __init__(self, path):
        import eccodes as ec
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recover"))
        import gribio as GI
        self.ec, self.path = ec, path
        self.offs = GI.scan_offsets(path)
        self.key = {}
        self.meta = {}
        for i, (off, n) in enumerate(self.offs):
            h = ec.codes_new_from_message(self._raw(i))
            try:
                k = (ec.codes_get(h, "shortName"), int(ec.codes_get(h, "level")))
                if k in self.key:
                    raise ValueError("%s: two messages for %s" % (path, k))
                self.key[k] = i
                self.meta[k] = {q: ec.codes_get(h, q) for q in
                                ("gridType", "N", "Ni", "Nj", "jScansPositively", "iScansNegatively",
                                 "longitudeOfFirstGridPointInDegrees", "numberOfValues", "dataDate",
                                 "dataTime", "binaryScaleFactor", "decimalScaleFactor", "bitsPerValue", "referenceValue",
                                 "typeOfLevel")}
            finally:
                ec.codes_release(h)

    def _raw(self, i):
        off, n = self.offs[i]
        with open(self.path, "rb") as f:
            f.seek(off)
            return f.read(n)

    def read(self, var, level):
        h = self.ec.codes_new_from_message(self._raw(self.key[(var, level)]))
        try:
            return np.asarray(self.ec.codes_get_values(h), dtype=np.float64)
        finally:
            self.ec.codes_release(h)

    def half_step(self, var, level):
        m = self.meta[(var, level)]
        return 0.5 * 2.0 ** m["binaryScaleFactor"] / 10.0 ** m["decimalScaleFactor"]


def check_cds_grid(cf, levels):
    vars_ = sorted({k[0] for k in cf.key})
    print("CDS file: %d messages, variables %s" % (len(cf.offs), vars_))
    for v in ("vo", "d", "u", "v"):
        miss = [L for L in levels if (v, L) not in cf.key]
        if miss:
            raise SystemExit("CDS file lacks %s on levels %s" % (v, miss[:10]))
    m = cf.meta[("u", levels[0])]
    print("CDS grid: %s N=%s Ni=%s Nj=%s jScansPositively=%s lon0=%s date %s %04d, %s"
          % (m["gridType"], m["N"], m["Ni"], m["Nj"], m["jScansPositively"],
             m["longitudeOfFirstGridPointInDegrees"], m["dataDate"], m["dataTime"], m["typeOfLevel"]))
    ok = (m["gridType"] == "regular_gg" and m["Ni"] == 4 * m["N"] and m["Nj"] == 2 * m["N"]
          and m["jScansPositively"] == 0 and m["iScansNegatively"] == 0
          and abs(m["longitudeOfFirstGridPointInDegrees"]) < 1e-9 and m["dataDate"] == 20181101 and m["dataTime"] == 0)
    if not ok:
        raise SystemExit("unexpected CDS grid or date (expected regular_gg F<N>, north->south, lon0 = 0, 20181101 00)")
    for k, mm in cf.meta.items():
        if (mm["gridType"], mm["N"], mm["numberOfValues"]) != (m["gridType"], m["N"], m["numberOfValues"]):
            raise SystemExit("message %s has a different grid" % (k,))
    return C.GaussGrid.regular(int(m["N"]))


# ----------------------------------------------------------------------------- per level
def do_level(L, Avo, Ad, cds, grid, T, w, radius, nband=10):
    rows = []
    t0 = time.time()
    m = np.arange(T + 1)
    masks = (m == 0, (m > 0) & (m % 2 == 0), m % 2 == 1)
    # 1. calibration, separately for vo and d
    fac = {}
    for name, A in (("vo", Avo), ("d", Ad)):
        parts = [synth([A], grid, T, mkeep=mk)[0] for mk in masks]
        y = cds[name]
        a, sn, rel, mx = calibrate(y, parts, w)
        hs = cds["half_" + name]
        # Gate (owner's decision of 7 Oct, after the 5-level diagnostic): the gate checks the convention only --
        # factors within 1e-3 of a candidate, and residual RMS <= 1.05 x packing noise q/sqrt(12) poleward of 30 deg.
        # MIR's F320 carries an extra, level-dependent component at n = T in the tropics (cause unknown); it is
        # reported (global and tropical ratios), not gated. Earlier rules (max <= 2.5 q/2; global <= 1.5) were dropped.
        r = y - np.stack(parts, 1) @ a
        noise = 2 * hs / math.sqrt(12)
        lat = np.repeat(grid.lat_deg, grid.nlon)
        ext = abs(lat) >= 30
        g_glob = math.sqrt(np.sum(w * r * r)) / noise
        g_ext = math.sqrt(np.sum(w[ext] * r[ext] ** 2) / w[ext].sum()) / noise
        okf = all(abs(ai - si) <= 1e-3 * abs(si) for ai, si in zip(a, sn))
        g_trop = math.sqrt(np.sum(w[~ext] * r[~ext] ** 2) / w[~ext].sum()) / noise
        okc = okf and g_ext <= 1.05
        print("L%03d calib %-2s: a(m=0, even m>0, odd m) = %s -> %s; RMS/packing noise |lat|>=30 %.2f (gate <= 1.05) "
              "%s; reported: global %.2f, tropics %.2f, max %.2f q/2"
              % (L, name, np.array2string(a, precision=7), [round(s, 4) for s in sn], g_ext, "OK" if okc else "FAIL",
                 g_glob, g_trop, mx / hs))
        rows.append(dict(level=L, test="calib_" + name, a0=a[0], ae=a[1], ao=a[2], rel=rel, maxabs=mx, half_step=hs,
                         noise_ratio_global=g_glob, noise_ratio_extratropics=g_ext, noise_ratio_tropics=g_trop))
        if not okc:
            raise SystemExit("calibration failed at level %d for %s: convention not one of the candidates, or "
                             "residual above packing level. Stop; do not compare u, v." % (L, name))
        fac[name] = sn
    if fac["vo"] != fac["d"]:
        raise SystemExit("vo and d calibrate to different conventions: %s vs %s" % (fac["vo"], fac["d"]))
    sc = np.where(m == 0, fac["vo"][0], np.where(m % 2 == 0, fac["vo"][1], fac["vo"][2]))
    Zc, Dc = Avo * sc[:, None], Ad * sc[:, None]
    # M4, amended 7 Oct before it was first run (the 1e-4 relative criterion is below the 16-bit packing of the CDS
    # grid itself): our analysis of the CDS F320 vo, d minus the ARCO coefficients, synthesised back to the grid,
    # must not exceed what the CDS packing alone produces through the same analysis-synthesis projection: RMS ratio
    # to projected simulated packing noise (uniform, step q) <= 1.05 poleward of 30 deg.
    # Tropical and global ratios and the coefficient relative RMS are reported.
    lat = np.repeat(grid.lat_deg, grid.nlon)
    ext = abs(lat) >= 30
    allpts = np.ones(grid.npts, bool)
    rng = np.random.default_rng(20181101)

    def wrms(x, sel):
        return math.sqrt(np.sum(w[sel] * x[sel] ** 2) / w[sel].sum())
    for name, Aref in (("vo", Zc), ("d", Dc)):
        Ag = analyse(cds[name], grid, T)
        wm = mweight(T)
        rel = math.sqrt((np.abs(Ag - Aref) ** 2 * wm).sum() / (np.abs(Aref) ** 2 * wm).sum())
        dproj = synth([Ag - Aref], grid, T)[0]
        q = 2 * cds["half_" + name]
        sim = synth([analyse(rng.uniform(-q / 2, q / 2, grid.npts), grid, T)], grid, T)[0]
        ratios = [wrms(dproj, sel) / wrms(sim, sel) for sel in (ext, ~ext, allpts)]
        ok4 = ratios[0] <= 1.05
        print("L%03d M4 %-2s (amended): projected difference / projected packing noise: |lat|>=30 %.3f (<= 1.05) %s; "
              "tropics %.2f, global %.2f; coefficient relative RMS %.2e (original 1e-4 criterion not applied)"
              % (L, name, ratios[0], "PASS" if ok4 else "FAIL", ratios[1], ratios[2], rel))
        rows.append(dict(level=L, test="M4_" + name, rel=rel, ratio_extratropics=ratios[0], ratio_tropics=ratios[1],
                         ratio_global=ratios[2], verdict="PASS" if ok4 else "FAIL"))
    lon_pts = np.concatenate([360.0 * np.arange(n) / n for n in grid.nlon])
    # 2. M2a: spectral ARCO -> our u, v
    ua, va = synth([Zc, Dc], grid, T, uv=True, radius=radius)
    # 3. M2b: CDS gridded vo, d -> our grid pipeline
    ub, vb = S.recover_serial(cds["vo"], cds["d"], grid, grid, T, radius)
    for tag, (x, y) in (("M2a_spec", (ua, va)), ("M2b_grid", (ub, vb))):
        for var, ours in (("u", x), ("v", y)):
            st = wstats(ours, cds[var], w)
            P = power_by_n(ours - cds[var], grid, T)
            hi = P[T + 1 - nband:].sum() / max(P.sum(), 1e-300)
            st.update(level=L, test="%s_%s" % (tag, var), half_step=cds["half_" + var], frac_hi_n=hi)
            q = 2 * cds["half_" + var]                      # plan: q = packing step of the reference message
            if tag == "M2b_grid":                            # plan M2: CDS F320 vo, d through our tool
                okm2 = st["rmsd"] <= q and st["maxabs"] <= 4 * q
                st["verdict"] = "PASS" if okm2 else "FAIL"
                print("L%03d M2 (plan) %s: RMS diff %.2e <= q = %.2e and max diff %.2e <= 4q = %.2e -> %s"
                      % (L, var, st["rmsd"], q, st["maxabs"], 4 * q, st["verdict"]))
            else:                                            # M2a: formal test, added post hoc on 7 Oct (owner)
                oka = st["rmsd"] <= q and st["maxabs"] <= 4 * q
                st["verdict"] = "PASS" if oka else "FAIL"
                print("L%03d M2a (post hoc) %s: RMS diff %.2e <= q = %.2e and max diff %.2e <= 4q = %.2e -> %s"
                      % (L, var, st["rmsd"], q, st["maxabs"], 4 * q, st["verdict"]))
            dd = ours - cds[var]
            k = int(np.argmax(abs(dd)))
            st.update(maxlat=float(lat[k]), maxlon=float(lon_pts[k]),
                      rms_extratropics=wrms(dd, ext), rms_tropics=wrms(dd, ~ext),
                      max_extratropics=float(abs(dd[ext]).max()), max_tropics=float(abs(dd[~ext]).max()))
            print("L%03d %s %s: max |diff| at lat %.2f lon %.2f; |lat|>=30: rms %.2e max %.2e (%.2f q); "
                  "|lat|<30: rms %.2e max %.2e (%.2f q)"
                  % (L, tag, var, st["maxlat"], st["maxlon"], st["rms_extratropics"], st["max_extratropics"],
                     st["max_extratropics"] / q, st["rms_tropics"], st["max_tropics"], st["max_tropics"] / q))
            rows.append(st)
            print("L%03d %s %s vs CDS: bias %+.2e rmsd %.2e max %.2e (rms ref %.2f, rel %.1e) slope %.7f "
                  "half step %.1e; diff power in n>%d: %.0f%%"
                  % (L, tag, var, st["bias"], st["rmsd"], st["maxabs"], st["rms_ref"], st["rel"], st["slope"],
                     cds["half_" + var], T - nband, 100 * hi))
    for var, x, y in (("u", ua, ub), ("v", va, vb)):
        st = wstats(y, x, w)
        st.update(level=L, test="M2b_minus_M2a_" + var)
        rows.append(st)
        print("L%03d M2b - M2a %s: rmsd %.2e max %.2e (input packing of CDS vo, d)" % (L, var, st["rmsd"], st["maxabs"]))
    print("L%03d done in %.0fs" % (L, time.time() - t0))
    return rows


# ----------------------------------------------------------------------------- self-test
def selftest():
    """Synthetic: 'CDS' fields made with a deliberately different convention (m>0 doubled,
    odd m sign-flipped). The calibration must find it and M2a/M2b must then agree to rounding."""
    T, N = 63, 32
    grid = C.GaussGrid.regular(N)
    rng = np.random.default_rng(1)
    def rand_A(scale):
        A = (rng.standard_normal((T + 1, T + 1)) + 1j * rng.standard_normal((T + 1, T + 1))) * scale
        A = np.triu(A)                      # n >= m  (A[m, n])
        A[0] = A[0].real
        A[0, 0] = 0
        n = np.arange(T + 1)
        return A / (1 + n[None, :]) ** 1.5
    Z, Dv = rand_A(1e-5), rand_A(2e-6)
    assert np.array_equal(arco_to_A(A_to_arco(Z), T), Z), "layout round trip"
    m = np.arange(T + 1)
    twist = np.where(m == 0, 1.0, np.where(m % 2 == 0, 2.0, -2.0))
    Zt, Dt = Z * twist[:, None], Dv * twist[:, None]
    cds = {}
    cds["vo"], cds["d"] = synth([Zt, Dt], grid, T)
    # 'ECMWF' u, v made the independent way: from the gridded vo, d (our grid pipeline)
    cds["u"], cds["v"] = S.recover_serial(cds["vo"], cds["d"], grid, grid, T)
    for k in ("vo", "d", "u", "v"):
        cds["half_" + k] = 1e-12 * abs(cds[k]).max()
    w = grid.area_weights()
    rows = do_level(1, arco_to_A(A_to_arco(Z), T), arco_to_A(A_to_arco(Dv), T), cds, grid, T, w, RADIUS)
    calib = [r for r in rows if r["test"].startswith("calib")]
    ok = all([round(r["a0"], 6), round(r["ae"], 6), round(r["ao"], 6)] == [1.0, 2.0, -2.0] for r in calib)
    worst = max(r["rel"] for r in rows if r["test"].startswith("M2"))
    ok &= worst < 1e-10
    print("selftest: calibration found (1, 2, -2): %s; worst relative u/v difference %.1e -> %s"
          % (all([round(r["a0"], 6), round(r["ae"], 6), round(r["ao"], 6)] == [1.0, 2.0, -2.0] for r in calib),
             worst, "PASS" if ok else "FAIL"))
    return ok


# ----------------------------------------------------------------------------- main
def parse_levels(s, nlev):
    if s == "all":
        return list(range(1, nlev + 1))
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arco")
    p.add_argument("--cds")
    p.add_argument("--levels", default="74")
    p.add_argument("--radius", type=float, default=RADIUS)
    p.add_argument("--csv")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.arco and a.cds):
        p.error("--arco and --cds are required")
    z = np.load(a.arco, allow_pickle=False)
    T = int(z["T"])
    if str(z["time"])[:13] != "2018-11-01T00":
        raise SystemExit("ARCO file time is %s, expected 2018-11-01T00" % z["time"])
    levels = parse_levels(a.levels, z["vo"].shape[0])
    cf = CdsFile(a.cds)
    grid = check_cds_grid(cf, levels)
    if grid.N * 2 - 1 < T:
        raise SystemExit("F%d cannot hold T%d exactly" % (grid.N, T))
    print("ARCO T%d, %d levels; comparing on %s; radius %.0f m" % (T, z["vo"].shape[0], grid.name, a.radius))
    w = grid.area_weights()
    rows = []
    for L in levels:
        cds = {v: cf.read(v, L) for v in ("vo", "d", "u", "v")}
        for v in ("vo", "d", "u", "v"):
            cds["half_" + v] = cf.half_step(v, L)
        rows += do_level(L, arco_to_A(z["vo"][L - 1], T), arco_to_A(z["d"][L - 1], T), cds, grid, T, w, a.radius)
        if a.csv:
            keys = sorted({k for r in rows for k in r})
            with open(a.csv + ".tmp", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=keys)
                wr.writeheader()
                wr.writerows(rows)
            os.replace(a.csv + ".tmp", a.csv)


if __name__ == "__main__":
    main()
