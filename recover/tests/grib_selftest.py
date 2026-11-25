#!/usr/bin/env python3
"""GRIB path self-test (run on Andes; needs ecCodes).  Two steps:

  python3 grib_selftest.py make  DIR   -> small GRIB2 vo, d (analytic fields) and u, v templates
  <launcher> python3 ../recover_uv.py ...   (see selftest_grib.sh)
  python3 grib_selftest.py check DIR   -> decisive checks

Checks
  A  output u, v vs our serial pipeline applied to the DECODED 16-bit vo, d:
     difference may only come from packing the output (|diff| <= 0.5 * step, + float noise)
  B  output u, v vs the analytic fields: reported (includes input packing error)
  C  header keys (mars, parameter, time, vertical, geography namespaces incl. array keys, packingType,
     bitsPerValue, pl) identical between output and template messages (M8)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import eccodes as ec                         # noqa: E402
import core as C, serial as S, gribio as G    # noqa: E402
from test_m1_analytic import fields         # noqa: E402

LEVELS = (1, 2, 3)
WAVES = {1: (5,), 2: (20,), 3: (3, 30)}
SAMPLE = "reduced_gg_pl_80_grib2"
NAMESPACES = ("mars", "parameter", "time", "vertical", "geography")


def octahedral_or_sample():
    """Try to turn the N80 sample into O80; fall back to the sample's own reduced grid."""
    h = ec.codes_grib_new_from_samples(SAMPLE)
    try:
        o = C.GaussGrid.octahedral(80)
        ec.codes_set_array(h, "pl", o.nlon.astype(np.int64))
        ec.codes_set_values(h, np.zeros(o.npts))
        if ec.codes_get(h, "numberOfValues") == o.npts:
            return h, "octahedral O80"
    except Exception as e:   # noqa: BLE001
        print("  (could not make an O80 message: %s; using the sample's reduced N80 grid)" % e)
    ec.codes_release(h)
    return ec.codes_grib_new_from_samples(SAMPLE), "classic reduced N80"


def make(d):
    os.makedirs(d, exist_ok=True)
    h0, kind = octahedral_or_sample()
    pl = np.asarray(ec.codes_get_array(h0, "pl"), dtype=np.int64)
    grid = C.GaussGrid.from_pl(int(ec.codes_get(h0, "N")), pl)
    la, lo = S.point_coords(grid)
    print("grid:", kind, grid.npts, "points")
    files = {v: open(os.path.join(d, "%s.grib" % v), "wb") for v in ("vo", "d", "u", "v")}
    params = {"vo": 138, "d": 155, "u": 131, "v": 132}
    exact = {}
    for L in LEVELS:
        f = [fields(la, lo, R, base=(i == 0), ph=0.2 * L) for i, R in enumerate(WAVES[L])]
        vals = {"vo": sum(x[0] for x in f), "d": sum(x[1] for x in f), }
        exact[L] = (sum(x[2] for x in f), sum(x[3] for x in f))
        # templates stand in for the corrupted u, v: non-constant, so packed at 16 bits like the real ones
        vals["u"], vals["v"] = 0.5 * exact[L][0], 0.5 * exact[L][1]
        for var in files:
            h = ec.codes_clone(h0)
            ec.codes_set(h, "typeOfLevel", "hybrid")
            ec.codes_set(h, "level", L)
            ec.codes_set(h, "paramId", params[var])
            ec.codes_set(h, "packingType", "grid_simple")
            ec.codes_set(h, "bitsPerValue", 16)
            ec.codes_set(h, "dataDate", 20181101)
            ec.codes_set(h, "dataTime", 0)
            ec.codes_set_values(h, vals[var])
            files[var].write(ec.codes_get_message(h))
            ec.codes_release(h)
    for f in files.values():
        f.close()
    ec.codes_release(h0)
    np.savez(os.path.join(d, "exact.npz"), **{"u%d" % L: exact[L][0] for L in LEVELS},
             **{"v%d" % L: exact[L][1] for L in LEVELS}, pl=pl)
    print("wrote", sorted(os.listdir(d)))


def bits(msg):
    h = ec.codes_new_from_message(msg)
    try:
        return ec.codes_get(h, "bitsPerValue")
    finally:
        ec.codes_release(h)


def header(h):
    out = {}
    for ns in NAMESPACES:
        it = ec.codes_keys_iterator_new(h, ns)
        while ec.codes_keys_iterator_next(it):
            k = ec.codes_keys_iterator_get_name(it)
            try:
                if ec.codes_get_size(h, k) > 1:         # array keys (pl, ...): compare whole arrays
                    out[ns + ":" + k] = tuple(ec.codes_get_array(h, k))
                else:
                    out[ns + ":" + k] = ec.codes_get_string(h, k)
            except Exception as e:   # noqa: BLE001
                out[ns + ":" + k] = "<unreadable: %s>" % type(e).__name__
        ec.codes_keys_iterator_delete(it)
    for k in ("packingType", "bitsPerValue", "numberOfValues"):
        out[k] = ec.codes_get_string(h, k)
    out["pl"] = tuple(ec.codes_get_array(h, "pl"))
    return out


def check(d, outdir):
    ok = True
    ex = np.load(os.path.join(d, "exact.npz"))
    pl = ex["pl"]
    N = len(pl) // 2
    grid = C.GaussGrid.from_pl(N, pl)
    lmax = N - 1 if np.array_equal(pl, C.GaussGrid.octahedral(N).nlon) else 2 * N - 1
    src = G.GribSource(os.path.join(d, "vo.grib"), os.path.join(d, "d.grib"))
    names = {v: [f for f in os.listdir(outdir) if f.startswith(v + ".recovered")] for v in ("u", "v")}
    for v in ("u", "v"):
        if len(names[v]) != 1:
            print("FAIL: expected one recovered %s file in %s, found %s" % (v, outdir, names[v]))
            return False
    rec = {v: G.GribFile(os.path.join(outdir, names[v][0])) for v in ("u", "v")}
    tpl = {v: G.GribFile(os.path.join(d, "%s.grib" % v)) for v in ("u", "v")}
    for v in ("u", "v"):
        if len(rec[v].index) != len(LEVELS):
            print("FAIL: %s has %d messages" % (names[v][0], len(rec[v].index))); ok = False
    for L in LEVELS:
        _, vo = src.read("vo", L)
        _, dd = src.read("d", L)
        su, sv = S.recover_serial(vo, dd, grid, grid, lmax)
        for v, sref, exv in (("u", su, ex["u%d" % L]), ("v", sv, ex["v%d" % L])):
            m, x = rec[v].read(L, v)
            hq = ec.codes_new_from_message(rec[v].raw(L - 1))
            E, Dd = ec.codes_get(hq, "binaryScaleFactor"), ec.codes_get(hq, "decimalScaleFactor")
            ec.codes_release(hq)
            bo, bt = (bits(f.raw(L - 1)) for f in (rec[v], tpl[v]))
            print("level %d %s: bitsPerValue template %d, output %d" % (L, v, bt, bo)); ok &= bo == bt == 16
            step = 2.0 ** E / 10.0 ** Dd          # actual quantisation step of the written message
            dA = abs(x - sref).max()
            dB = abs(x - exv).max() / abs(exv).max()
            okA = dA <= 0.5 * step * 1.001 + 1e-9 * abs(sref).max()
            print("level %d %s: A |out - serial(decoded input)| = %.2e (half packing step = %.2e) %s;  B rel. error vs analytic = %.1e"
                  % (L, v, dA, 0.5 * step, "PASS" if okA else "FAIL", dB))
            ok &= okA
            ht = ec.codes_new_from_message(tpl[v].raw(L - 1)); hr = ec.codes_new_from_message(rec[v].raw(L - 1))
            a, b = header(ht), header(hr)
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            ec.codes_release(ht); ec.codes_release(hr)
            print("   C header keys compared: %d, differing: %s" % (len(a), diff or "none"))
            ok &= not diff
    print("GRIB self-test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2])
    else:
        sys.exit(0 if check(sys.argv[2], sys.argv[3]) else 1)
