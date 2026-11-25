#!/usr/bin/env python3
"""
Fetch ERA5 model-level vorticity and divergence (T639 spherical-harmonic
coefficients) for one time from Google's public ARCO-ERA5 store, and save them
for the M2/M4 tests.  Run on an Andes LOGIN node (compute nodes cannot reach
storage.googleapis.com).

  pip install xarray zarr gcsfs        # once, in the vod2uv venv
  python3 fetch_arco.py --time 2018-11-01T00 --out era5_arco_sh_vo_d_2018110100.npz

Source: gs://gcp-public-data-arco-era5/co/model-level-wind.zarr-v2
(ARCO-ERA5 README: dims time, hybrid, values; values = 410,240 = (T+1)(T+2) reals
for T639, i.e. 205,120 complex coefficients in GRIB order: m = 0..T, n = m..T, (re, im)).

The script prints the dataset inventory and refuses to save if the selected time,
the 137 levels or the coefficient count are not what we expect.  It does NOT
convert normalisation or sign conventions: that is calibrated and tested in M2/M4.
"""
import argparse
import json
import sys

import numpy as np

URL = "gs://gcp-public-data-arco-era5/co/model-level-wind.zarr-v2"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=URL)
    p.add_argument("--time", default="2018-11-01T00")
    p.add_argument("--vars", nargs="+", default=["vo", "d"])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import xarray as xr
    ds = xr.open_zarr(a.url, chunks=None, storage_options=dict(token="anon"))
    print(ds)
    for v in a.vars:
        if v not in ds:
            sys.exit("variable %s not in store; variables are %s" % (v, list(ds.data_vars)))
    sel = ds[a.vars].sel(time=np.datetime64(a.time))
    t = str(sel["time"].values)
    if not t.startswith(a.time.replace("T00", "T00:00")) and not t.startswith(a.time):
        sys.exit("selected time %s does not match %s" % (t, a.time))
    out = {}
    for v in a.vars:
        x = np.asarray(sel[v].values)
        print("%s: shape %s dtype %s  min %.4e max %.4e" % (v, x.shape, x.dtype, x.min(), x.max()))
        if x.ndim != 2 or x.shape[0] != 137:
            sys.exit("%s: expected (137 levels, values), got %s" % (v, x.shape))
        nval = x.shape[1]
        T = int(round((-3 + np.sqrt(1 + 4 * nval)) / 2))
        if (T + 1) * (T + 2) != nval:
            sys.exit("%s: %d values is not (T+1)(T+2) for any T" % (v, nval))
        out[v] = x
        out[v + "_attrs"] = json.dumps({k: str(val) for k, val in sel[v].attrs.items()})
    lev = np.asarray(sel["hybrid"].values) if "hybrid" in sel.coords else np.arange(1, 138)
    np.savez(a.out, levels=lev, time=t, T=T, source=a.url, **out)
    print("saved %s: T%d, levels %s..%s, vars %s" % (a.out, T, lev[0], lev[-1], a.vars))


if __name__ == "__main__":
    main()
