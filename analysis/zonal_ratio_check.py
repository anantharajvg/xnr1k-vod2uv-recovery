#!/usr/bin/env python3
"""
Zonal-mean u: XNR1K (corrupted or recovered) against ERA5, per model level.

Reproduces the 7 Oct 2026 check (docs/findings-2026-10-07.md):
  A = IFS zonal-mean u file written by CDO, u(time, lev, lat, lon=1), e.g. u_000000-021420.tzmean.nc
      (that file is a 15-day mean of the CORRUPTED u_131_h3f7-ndjf_*.grib, see its history attribute)
  B = ERA5 zonal-mean u, u(time, lev, lat, lon=1), e.g. gcs/era5_steps/era5gcs_u_zm_20181101_0000.nc
      (ARCO 0.25-deg regridded product, single time)
For each level: regression slope of A on B (through the origin) and correlation over |lat| < LATMAX,
after interpolating A to B's latitudes. Prints the time metadata of both files: if the time windows
differ (e.g. 15-day mean vs one analysis time) the slope is only indicative.

usage: python3 zonal_ratio_check.py A.nc B.nc [--levels 20,30,...] [--latmax 85] [--point -60]
Needs numpy and scipy (netCDF classic files).
"""
import argparse
import sys

import numpy as np
from scipy.io import netcdf_file


def load(path):
    f = netcdf_file(path, "r", mmap=False)
    u = f.variables["u"]
    if u.dimensions[-1] not in ("lon", "longitude") or u.shape[-1] != 1:
        sys.exit("%s: expected zonal-mean u with lon=1, got dims %s shape %s" % (path, u.dimensions, u.shape))
    x = u[0, :, :, 0].astype(float)
    lat = f.variables["lat"][:].astype(float)
    t = f.variables["time"]
    tinfo = "time=%s %s" % (t[:].tolist(), getattr(t, "units", b"").decode() if isinstance(getattr(t, "units", ""), bytes) else getattr(t, "units", ""))
    if "time_bnds" in f.variables:
        tinfo += ", time_bnds=%s h" % f.variables["time_bnds"][:].tolist()
    hist = f._attributes.get("history", b"")
    hist = hist.decode() if isinstance(hist, bytes) else str(hist)
    f.close()
    return x, lat, tinfo, hist


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--levels", default="20,30,40,50,60,74,90,100,114,130")
    p.add_argument("--latmax", type=float, default=85.0)
    p.add_argument("--point", type=float, default=-60.0)
    a = p.parse_args()
    ua, la, ta, ha = load(a.a)
    ub, lb, tb, hb = load(a.b)
    if ua.shape[0] != ub.shape[0]:
        sys.exit("level counts differ: %d vs %d" % (ua.shape[0], ub.shape[0]))
    print("A: %s\n   %s\n   history starts: %s" % (a.a, ta, ha[:160].replace("\n", " | ")))
    print("B: %s\n   %s\n   history starts: %s" % (a.b, tb, hb[:160].replace("\n", " | ")))
    oa, ob = np.argsort(la), np.argsort(lb)
    print("\nlevel  slope(A on B)  corr   max A   max B   A, B at %g deg" % a.point)
    for L in [int(x) for x in a.levels.split(",")]:
        k = L - 1
        ai = np.interp(lb, la[oa], ua[k][oa])
        m = np.abs(lb) < a.latmax
        s = np.dot(ai[m], ub[k][m]) / np.dot(ub[k][m], ub[k][m])
        c = np.corrcoef(ai[m], ub[k][m])[0, 1]
        pa = np.interp(a.point, la[oa], ua[k][oa]); pb = np.interp(a.point, lb[ob], ub[k][ob])
        print("%5d  %12.3f  %6.3f  %6.1f  %6.1f   %6.1f, %6.1f" % (L, s, c, ua[k].max(), ub[k].max(), pa, pb))


if __name__ == "__main__":
    main()
