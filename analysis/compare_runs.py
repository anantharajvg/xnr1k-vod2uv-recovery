#!/usr/bin/env python3
"""Compare recovered u, v between two runs, level by level (reproducibility / environment check).

Accepts merged files, single-level files or parts directories.  Per level and variable it reports whether the GRIB
messages are byte-identical, and if not, the maximum difference of the decoded values in units of the packing step.

  python3 compare_runs.py --a run_L074/u_...recovered.L074.grib --b run_L074_nfs/u_...recovered.L074.grib --levels 74
  python3 compare_runs.py --a-parts run_000000_attempt1/parts --b-parts run_000000/parts --levels 1-41
Exit status 0 only if every compared level is byte-identical (use --tolerance-steps to accept packing-level changes).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recover"))
sys.path.insert(0, HERE)
import eccodes as ec                          # noqa: E402
import gribio as G                            # noqa: E402
import m2_era5 as M                           # noqa: E402  (parse_levels)
from check_recovered import recovered_message  # noqa: E402


def message(path_or_parts, var, level, parts):
    if parts:
        p = os.path.join(path_or_parts, "%s_L%03d.grib" % (var, level))
        if not os.path.exists(p):
            return None
        return recovered_message(p, level)
    return recovered_message(path_or_parts, level)


def decode(msg):
    h = ec.codes_new_from_message(msg)
    try:
        q = 2.0 ** ec.codes_get(h, "binaryScaleFactor") / 10.0 ** ec.codes_get(h, "decimalScaleFactor")
        return np.asarray(ec.codes_get_values(h)), q, int(ec.codes_get(h, "level"))
    finally:
        ec.codes_release(h)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a"); p.add_argument("--b")
    p.add_argument("--a-v"); p.add_argument("--b-v")
    p.add_argument("--a-parts"); p.add_argument("--b-parts")
    p.add_argument("--levels", required=True)
    p.add_argument("--tolerance-steps", type=float, default=0.0)
    a = p.parse_args()
    parts = bool(a.a_parts)
    levels = M.parse_levels(a.levels, 137)
    srcs = {"u": (a.a_parts or a.a, a.b_parts or a.b), "v": (a.a_parts or a.a_v, a.b_parts or a.b_v)}
    ok, n = True, 0
    for L in levels:
        for var in ("u", "v"):
            sa, sb = srcs[var]
            if sa is None or sb is None:
                continue
            ma, mb = message(sa, var, L, parts), message(sb, var, L, parts)
            if ma is None or mb is None:
                print("L%03d %s: missing in %s" % (L, var, "A" if ma is None else "B"))
                continue
            n += 1
            if ma == mb:
                print("L%03d %s: byte-identical" % (L, var))
                continue
            xa, qa, la = decode(ma)
            xb, qb, lb = decode(mb)
            d = float(abs(xa - xb).max())
            steps = d / max(qa, qb)
            good = la == lb == L and steps <= a.tolerance_steps
            ok &= good
            print("L%03d %s: bytes differ; max |diff| %.3e = %.2f packing steps (q %.2e / %.2e) -> %s"
                  % (L, var, d, steps, qa, qb, "within tolerance" if good else "DIFFERENT"))
    print("compared %d messages: %s" % (n, "PASS" if ok and n else "FAIL"))
    sys.exit(0 if ok and n else 1)


if __name__ == "__main__":
    main()
