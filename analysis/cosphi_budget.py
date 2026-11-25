#!/usr/bin/env python3
"""Split the cos(lat) residual of each level into packing noise and an extra, non-packing part (m/s).

Reads the per-level CSVs written by cosphi_check.py --levels (via scripts/check_parts.sh) and, per level and variable,
inverts   rms(r)^2 = sigma_p^2 + delta^2,   sigma_p^2 = q_c^2/12 + <cos^2> q_r^2/12   (<cos^2> = 2/3, area mean)
to give delta = sigma_p * sqrt(ratio^2 - 1): the RMS difference between our recovery*cos and the archived field that
packing does not explain. Test of hypothesis (b) vs (a) in the work log (8 Oct): if the excess ratio is mostly set by
the packing step, delta varies smoothly with level while the ratio jumps where q_c changes by a factor of 2.

  python3 cosphi_budget.py $W/run_000000/checks/cosphi_L*.csv [--out budget.csv]
Reads only; no ecCodes needed.
"""
import argparse
import csv
import math
import sys

C2 = 2.0 / 3.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+")
    p.add_argument("--out")
    p.add_argument("--stats", help="level_stats.csv of the run (rms of recovered u, v): adds delta/rms (hypothesis g,"
                   " single-precision arithmetic in the original, predicts a roughly constant relative error)")
    a = p.parse_args()
    rows = []
    for f in a.csv:
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        sys.exit("no rows")
    rows.sort(key=lambda r: (int(r["level"]), r["var"]))
    rmsof = {}
    if a.stats:
        with open(a.stats) as fh:
            for r in csv.DictReader(fh):
                rmsof[(int(r["level"]), r["var"])] = float(r["rms"])     # last row wins if a level was redone
    out = []
    print("level var   q_c        q_r        ratio  sigma_p    delta(m/s)  delta/q_c  q_c/prev   rms(m/s)  delta/rms")
    prev = {}
    for r in rows:
        L, v = int(r["level"]), r["var"]
        qc, qr, ratio = float(r["q_c"]), float(r["q_r"]), float(r["rms_over_packing"])
        sp = math.sqrt(qc * qc / 12 + C2 * qr * qr / 12)
        d = sp * math.sqrt(max(ratio * ratio - 1, 0.0))
        step = qc / prev[v] if v in prev else float("nan")
        prev[v] = qc
        rm = rmsof.get((L, v))
        print("%5d %-3s %.3e  %.3e  %5.2f  %.3e  %.3e   %5.2f      %4.2f    %s  %s"
              % (L, v, qc, qr, ratio, sp, d, d / qc, step, "%7.3f" % rm if rm else "   -   ",
                 "%.2e" % (d / rm) if rm else "   -"))
        out.append(dict(level=L, var=v, q_c=qc, q_r=qr, ratio=ratio, sigma_p=sp, delta=d, rms=rm or "",
                        delta_over_rms=d / rm if rm else ""))
    for v in ("u", "v"):
        ds = [o["delta"] for o in out if o["var"] == v]
        if ds:
            ds.sort()
            print("%s: delta median %.3e m/s, min %.3e, max %.3e (%d levels)" % (v, ds[len(ds) // 2], ds[0], ds[-1], len(ds)))
            rel = sorted(o["delta_over_rms"] for o in out if o["var"] == v and o["delta_over_rms"] != "")
            if rel:
                m = sum(rel) / len(rel)
                cv = math.sqrt(sum((x - m) ** 2 for x in rel) / len(rel)) / m
                print("%s: delta/rms median %.2e, min %.2e, max %.2e, coefficient of variation %.2f"
                      % (v, rel[len(rel) // 2], rel[0], rel[-1], cv))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
