#!/usr/bin/env python3
"""
compare.py -- per-level statistics between two u, v datasets on the same regular
Gaussian grid (e.g. XNR1K truncated to T639 on F320 vs ERA5 from NCAR ds633.6).

Each field is given as PATH:VAR. Leading dimensions are (time, level) or (level);
pick the time with --a-time / --b-time (index). Grids must match (latitude to 1e-6 deg,
longitude to 1e-6 deg) - nothing is interpolated.

Statistics per level and variable, area-weighted (Gaussian weights when the
latitudes are Gaussian, else cos(latitude)):
  bias = mean(A-B), rmsd, rms_a, rms_b, corr (centred, weighted), nrmsd = rmsd/rms_b,
  and the RMS of the vector-wind difference.
Zonal means of A and B are saved for plots.py (zonal).

Optional --b-sp PATH:VAR plus --b-ab PATH (file holding a_model, b_model or hyam, hybm)
gives each level a global-mean pressure p = a + b * mean(sp) for the profile plots.

Example (NCAR files, level 74 only):
  python3 compare.py --a-u recovered_F320_T639.nc:u --a-v recovered_F320_T639.nc:v \
     --b-u $E/e5.oper.an.ml.0_5_0_2_2_u.regn320uv.2018110100_2018110105.nc:U --b-time 0 \
     --b-v $E/e5.oper.an.ml.0_5_0_2_3_v.regn320uv.2018110100_2018110105.nc:V \
     --b-sp $E/e5.oper.an.ml.128_134_sp.regn320sc.2018110100_2018110105.nc:SP \
     --b-ab $E/e5.oper.an.ml.0_5_0_2_2_u.regn320uv.2018110100_2018110105.nc \
     --levels 74 --out cmp_L074
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from vod2uv_mpi import Reader, gauss_legendre_north   # noqa: E402

LEVEL_NAMES = ("lev", "level", "hybrid", "model_level", "levelist")
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "longitude")


class Field:
    """A PATH:VAR field with (time?, level, lat, lon) layout."""

    def __init__(self, spec, time_index=None):
        path, var = spec.rsplit(":", 1)
        self.path, self.var = path, var
        self.r = Reader(path)
        dims = self.r.dims(var)
        self.dims = dims
        names = self.r.varnames()
        self.latname = next(n for n in LAT_NAMES if n in dims)
        self.lonname = next(n for n in LON_NAMES if n in dims)
        self.levname = next((n for n in dims if n in LEVEL_NAMES), None)
        self.lat = self.r.read(self.latname, slice(None)).astype(float)
        self.lon = self.r.read(self.lonname, slice(None)).astype(float)
        self.levels = (self.r.read(self.levname, slice(None)).astype(int)
                       if self.levname and self.levname in names else None)
        lead = dims[:-2]
        self.tdim = next((d for d in lead if d != self.levname), None)
        if self.tdim is not None and time_index is None:
            n = self.r.shape(var)[dims.index(self.tdim)]
            if n != 1:
                raise ValueError("%s has %d times; give the time index" % (spec, n))
            time_index = 0
        self.ti = time_index
        if self.tdim is not None and self.tdim in names:
            t = self.r.read(self.tdim, slice(None))
            u = self.r.attrs(self.tdim).get("units", "")
            self.time_label = "%s = %s %s" % (self.tdim, t[self.ti], u)
        else:
            self.time_label = "no time dimension"

    def level(self, L):
        sl = []
        for d in self.dims[:-2]:
            if d == self.tdim:
                sl.append(self.ti)
            elif d == self.levname:
                if self.levels is None:
                    sl.append(L - 1)
                else:
                    hit = np.where(self.levels == L)[0]
                    if not hit.size:
                        raise ValueError("%s: level %d not present" % (self.path, L))
                    sl.append(int(hit[0]))
            else:
                sl.append(0)
        x = self.r.read(self.var, tuple(sl) + (slice(None), slice(None))).astype(np.float64)
        return x


def lat_weights(lat):
    nlat = len(lat)
    mu, w = gauss_legendre_north(nlat) if nlat % 2 == 0 else (None, None)
    if mu is not None:
        latg = np.degrees(np.arcsin(np.concatenate((mu, -mu[::-1]))))
        for cand in (lat, lat[::-1]):
            if np.max(np.abs(cand - latg)) < 1e-4:
                ww = np.concatenate((w, w[::-1])) / 2.0
                return (ww if cand is lat else ww[::-1]), "gaussian"
    c = np.cos(np.radians(lat))
    return c / c.sum(), "cos(lat)"


def wstats(a, b, w2):
    d = a - b
    ma, mb = np.sum(w2 * a), np.sum(w2 * b)
    va, vb = np.sum(w2 * (a - ma) ** 2), np.sum(w2 * (b - mb) ** 2)
    cov = np.sum(w2 * (a - ma) * (b - mb))
    rmsb = np.sqrt(np.sum(w2 * b * b))
    rmsd = np.sqrt(np.sum(w2 * d * d))
    return dict(bias=float(np.sum(w2 * d)), rmsd=float(rmsd), rms_a=float(np.sqrt(np.sum(w2 * a * a))),
                rms_b=float(rmsb), corr=float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else float("nan"),
                nrmsd=float(rmsd / rmsb) if rmsb > 0 else float("nan"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in ("a", "b"):
        p.add_argument("--%s-u" % s, required=True); p.add_argument("--%s-v" % s, required=True)
        p.add_argument("--%s-time" % s, type=int, default=None)
        p.add_argument("--%s-label" % s, default="XNR1K" if s == "a" else "ERA5")
    p.add_argument("--b-sp", default=None); p.add_argument("--b-ab", default=None)
    p.add_argument("--levels", default="all")
    p.add_argument("--out", required=True, help="output prefix")
    a = p.parse_args(argv)

    F = {k: Field(getattr(a, k.replace("-", "_")), getattr(a, k[0] + "_time"))
         for k in ("a_u", "a_v", "b_u", "b_v")}
    ref = F["b_u"]
    for k, f in F.items():
        if f.lat.shape != ref.lat.shape or np.max(np.abs(f.lat - ref.lat)) > 1e-6:
            sys.exit("latitudes of %s differ from %s" % (f.path, ref.path))
        if f.lon.shape != ref.lon.shape or np.max(np.abs(((f.lon - ref.lon + 180) % 360) - 180)) > 1e-6:
            sys.exit("longitudes of %s differ from %s" % (f.path, ref.path))
    w, wkind = lat_weights(ref.lat)
    w2 = (w / len(ref.lon))[:, None] * np.ones((1, len(ref.lon)))
    avail = sorted(set(F["a_u"].levels.tolist()) & set(F["b_u"].levels.tolist())) \
        if F["a_u"].levels is not None and F["b_u"].levels is not None else None
    if a.levels == "all":
        if avail is None:
            sys.exit("no level coordinate; give --levels")
        levels = avail
    else:
        from recover_uv import parse_levels
        levels = parse_levels(a.levels, 10 ** 6)
    pres = None
    if a.b_sp and a.b_ab:
        fsp = Field(a.b_sp, a.b_time)
        sl = tuple(fsp.ti if d == fsp.tdim else 0 for d in fsp.dims[:-2])
        sp = fsp.r.read(fsp.var, sl + (slice(None), slice(None))).astype(float)
        rab = Reader(a.b_ab)
        names = rab.varnames()
        an = next(n for n in ("a_model", "hyam") if n in names)
        bn = next(n for n in ("b_model", "hybm") if n in names)
        A, B = rab.read(an, slice(None)).astype(float), rab.read(bn, slice(None)).astype(float)
        psm = float(np.sum(w2 * sp))
        pres = {L: float(A[L - 1] + B[L - 1] * psm) for L in range(1, len(A) + 1)}
        print("global-mean surface pressure %.1f hPa (from %s)" % (psm / 100, a.b_sp))
    print("A: %s (%s)\nB: %s (%s)\nweights: %s" % (F["a_u"].path, F["a_u"].time_label, F["b_u"].path,
                                                    F["b_u"].time_label, wkind))
    rows = []
    zon = {}
    for L in levels:
        x = {k: F[k].level(L) for k in F}
        su, sv = wstats(x["a_u"], x["b_u"], w2), wstats(x["a_v"], x["b_v"], w2)
        vec = float(np.sqrt(np.sum(w2 * ((x["a_u"] - x["b_u"]) ** 2 + (x["a_v"] - x["b_v"]) ** 2))))
        for var, s in (("u", su), ("v", sv)):
            rows.append(dict(level=L, p_hPa=(pres[L] / 100 if pres else float("nan")), var=var, **s, vec_rmsd=vec))
        zon[L] = np.stack([x[k].mean(axis=1) for k in ("a_u", "b_u", "a_v", "b_v")])
        print("level %3d  u: bias %+6.2f rmsd %5.2f corr %.4f | v: bias %+6.2f rmsd %5.2f corr %.4f | vector rmsd %5.2f m/s"
              % (L, su["bias"], su["rmsd"], su["corr"], sv["bias"], sv["rmsd"], sv["corr"], vec))
    with open(a.out + "_stats.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    np.savez(a.out + "_zonal.npz", lat=ref.lat, levels=np.array(levels),
             zonal=np.stack([zon[L] for L in levels]), labels=np.array([a.a_label, a.b_label]))
    with open(a.out + "_meta.json", "w") as f:
        json.dump(dict(a=F["a_u"].path, b=F["b_u"].path, a_time=F["a_u"].time_label,
                       b_time=F["b_u"].time_label, weights=wkind, labels=[a.a_label, a.b_label]), f, indent=1)
    print("wrote %s_stats.csv, %s_zonal.npz" % (a.out, a.out))


if __name__ == "__main__":
    main()
