#!/usr/bin/env python3
"""
Download the ERA5 model-level reference data for the XNR1K comparison
(analysis valid 2018-11-01 00 UTC, all 137 L137 model levels).

Requires: cdsapi >= 0.7 and a CDS account with ~/.cdsapirc :
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>
and acceptance of the ERA5 licence on the CDS web page of "reanalysis-era5-complete".

What is retrieved and why
-------------------------
ERA5 model-level vo, d (and lnsp) are archived as T639 spherical harmonics.
u, v on model levels are NOT archived: ECMWF's MIR derives them from spectral
vo/d at retrieval time. That makes ERA5 a test of our vod2uv against an
independent implementation, as well as the reference for XNR1K.

  A  era5_ml_F320_vo_d_u_v.grib : vo, d, u, v on the regular (full) Gaussian
                                   grid F320 (640 x 1280), levels 1-137.
                                   -> methodology test 1 (regular-grid path)
                                   -> XNR1K vs ERA5 comparison at T639
  B  era5_ml_N320_vo_d.grib     : vo, d on ERA5's native reduced Gaussian grid
                                   N320. -> methodology test 2 (reduced-grid path)
  C  era5_ml_sh_vo_d.grib       : vo, d as native T639 spherical harmonics.
                                   -> spectral reference: checks the
                                   grid->spectral (analysis) half on its own
  D  era5_ml_sh_lnsp.grib       : lnsp (level 1) spectral, and
     era5_ml_F320_lnsp.grib     : lnsp on F320, for level -> pressure.

"truncation": "none" stops MIR from truncating the spectral fields before
gridding. For F320/N320 the automatic truncation would also be T639, so this
only makes the choice explicit and reproducible.

Model-level ERA5 sits on tape in MARS; requests can queue for hours.
Requests that have already been downloaded are skipped (rerun safely).
"""
import argparse
import os
import sys

DATASET = "reanalysis-era5-complete"
LEVELS = "/".join(str(k) for k in range(1, 138))
BASE = {
    "class": "ea",
    "expver": "1",
    "stream": "oper",
    "type": "an",
    "date": "2018-11-01",
    "time": "00:00:00",
    "levtype": "ml",
}
REQUESTS = {
    # name                          extra keys                                         expected #messages
    "era5_ml_F320_vo_d_u_v.grib": ({"levelist": LEVELS, "param": "138/155/131/132",
                                    "grid": "F320", "truncation": "none"},           4 * 137),
    "era5_ml_N320_vo_d.grib":     ({"levelist": LEVELS, "param": "138/155",
                                    "grid": "N320", "truncation": "none"},           2 * 137),
    "era5_ml_sh_vo_d.grib":       ({"levelist": LEVELS, "param": "138/155"},         2 * 137),
    "era5_ml_sh_lnsp.grib":       ({"levelist": "1", "param": "152"},                1),
    "era5_ml_F320_lnsp.grib":     ({"levelist": "1", "param": "152",
                                    "grid": "F320", "truncation": "none"},           1),
}


def count_messages(path):
    try:
        import eccodes
    except ImportError:
        return None
    n = 0
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            n += 1
            eccodes.codes_release(h)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=".")
    p.add_argument("--only", nargs="*", choices=list(REQUESTS), help="subset of requests")
    p.add_argument("--dry-run", action="store_true", help="print the requests and exit")
    a = p.parse_args()
    names = a.only or list(REQUESTS)
    os.makedirs(a.outdir, exist_ok=True)

    if a.dry_run:
        for nm in names:
            print(nm, {**BASE, **REQUESTS[nm][0]})
        return 0

    import cdsapi
    c = cdsapi.Client()
    status = 0
    for nm in names:
        extra, expected = REQUESTS[nm]
        target = os.path.join(a.outdir, nm)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            print("[skip] %s exists" % target)
        else:
            req = {**BASE, **extra, "data_format": "grib"}
            print("[request] %s\n  %s" % (target, req), flush=True)
            c.retrieve(DATASET, req, target + ".part")
            os.replace(target + ".part", target)
        n = count_messages(target)
        if n is None:
            print("  (install eccodes to verify message counts; expected %d)" % expected)
        elif n != expected:
            print("  ERROR: %s has %d GRIB messages, expected %d" % (nm, n, expected))
            status = 1
        else:
            print("  OK: %d messages" % n)
    return status


if __name__ == "__main__":
    sys.exit(main())
