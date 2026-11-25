#!/usr/bin/env python3
"""Packing step q = 2^E / 10^D of every message in GRIB2 grid_simple files, read from the section-5 header bytes only.

Reads a few hundred kB per 512 MB message (sections 0-5), so all 137 levels of the four XNR1K files take seconds.
Optionally predicts the wind error that 16-bit packing of the archived vo and d puts into the recovered u, v, and
compares it with the measured non-packing residual delta from cosphi_budget.py (hypothesis 8 Oct: delta comes from
the input packing, the original u, v having been computed from unpacked spectral vo, d).

Prediction for white, uniform quantisation noise (variance q^2/12, independent per point) in vo and d, projected to
T and inverted (psi = -a^2 zeta / n(n+1), chi likewise):
   <u^2 + v^2> = (s_vo^2 + s_d^2) a^2 / Np * S,   S = sum_{n=1..T} (2n+1)/(n(n+1)),   per component half of it,
and the cos-weighted comparison quantity (u cos) has <cos^2> = 2/3 of that. Approximations: equal-area points,
no aliasing beyond T; so the prediction is good to a factor of about 1.5, not better.

  python3 packing_steps.py --vo VO.grib --d D.grib --u U.grib --v V.grib [--budget cosphi_budget.csv] [--check-level 74]
"""
import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recover"))
import gribio as G          # noqa: E402

A = 6371229.0
NP = 256288000
T = 7999


def sm16(b):
    """GRIB2 scale factors are sign-and-magnitude, not two's complement."""
    x = int.from_bytes(b, "big")
    return -(x & 0x7FFF) if x & 0x8000 else x


def headers(path):
    out = []
    with open(path, "rb") as f:
        for off, n in G.scan_offsets(path):
            f.seek(off + 16)
            pos, level, rec = off + 16, None, None
            while pos < off + n - 4:
                f.seek(pos)
                h = f.read(5)
                ln, num = int.from_bytes(h[:4], "big"), h[4]
                if num == 4:
                    s = h + f.read(min(ln, 64) - 5)
                    if int.from_bytes(s[7:9], "big") == 0:          # template 4.0: octets 23-28 first fixed surface
                        sf, val = s[23], int.from_bytes(s[24:28], "big")
                        level = val / 10 ** sf if sf != 255 else val
                elif num == 5:
                    s = h + f.read(ln - 5)
                    tpl = int.from_bytes(s[9:11], "big")
                    if tpl != 0:
                        raise SystemExit("%s: data representation template %d, expected 0 (grid_simple)" % (path, tpl))
                    E, D, bits = sm16(s[15:17]), sm16(s[17:19]), s[19]
                    rec = dict(E=E, D=D, bits=bits, q=2.0 ** E / 10.0 ** D)
                    break
                pos += ln
            if rec is None:
                raise SystemExit("%s: no section 5 in message at byte %d" % (path, off))
            rec["level"] = level
            out.append(rec)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("vo", "d", "u", "v"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--budget", help="cosphi_budget.csv to compare with")
    p.add_argument("--check-level", type=int, help="cross-check this level against ecCodes")
    p.add_argument("--out")
    a = p.parse_args()
    H = {k: headers(getattr(a, k)) for k in ("vo", "d", "u", "v")}
    n = len(H["vo"])
    for k in H:
        if len(H[k]) != n:
            raise SystemExit("message counts differ: %s" % {k: len(H[k]) for k in H})
        for i, r in enumerate(H[k]):
            if r["level"] is not None and int(r["level"]) != i + 1:
                raise SystemExit("%s message %d holds level %s" % (k, i, r["level"]))
    if a.check_level:
        import eccodes as ec
        for k in H:
            h = ec.codes_new_from_message(G.GribFile(getattr(a, k)).raw(a.check_level - 1))
            e, d, b = (ec.codes_get(h, x) for x in ("binaryScaleFactor", "decimalScaleFactor", "bitsPerValue"))
            ec.codes_release(h)
            r = H[k][a.check_level - 1]
            ok = (e, d, b) == (r["E"], r["D"], r["bits"])
            print("check L%d %s: ecCodes E=%d D=%d bits=%d, header bytes E=%d D=%d bits=%d -> %s"
                  % (a.check_level, k, e, d, b, r["E"], r["D"], r["bits"], "OK" if ok else "MISMATCH"))
            if not ok:
                sys.exit(1)
    S = sum((2 * m + 1) / (m * (m + 1)) for m in range(1, T + 1))
    budget = {}
    if a.budget:
        with open(a.budget) as fh:
            for r in csv.DictReader(fh):
                budget[(int(r["level"]), r["var"])] = float(r["delta"])
    rows = []
    print("S = %.4f; prediction delta_pred = sqrt(2/3 * 1/2 * (s_vo^2+s_d^2) * a^2/Np * S)" % S)
    print("level  q_vo       q_d        q_u        q_v       delta_pred  delta_u    delta_v    u/pred  v/pred")
    for i in range(n):
        L = i + 1
        qv, qd = H["vo"][i]["q"], H["d"][i]["q"]
        pred = math.sqrt(2.0 / 3.0 * 0.5 * (qv * qv + qd * qd) / 12.0 * A * A / NP * S)
        du, dv = budget.get((L, "u")), budget.get((L, "v"))
        f = lambda x: "%.3e" % x if x is not None else "    -    "                         # noqa: E731
        g = lambda x: "%6.2f" % (x / pred) if x is not None else "   -  "                  # noqa: E731
        print("%5d  %.3e  %.3e  %.3e  %.3e  %.3e   %s  %s  %s  %s"
              % (L, qv, qd, H["u"][i]["q"], H["v"][i]["q"], pred, f(du), f(dv), g(du), g(dv)))
        rows.append(dict(level=L, q_vo=qv, q_d=qd, q_u=H["u"][i]["q"], q_v=H["v"][i]["q"],
                         bits_vo=H["vo"][i]["bits"], bits_d=H["d"][i]["bits"], delta_pred=pred,
                         delta_u=du if du is not None else "", delta_v=dv if dv is not None else ""))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
