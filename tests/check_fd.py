#!/usr/bin/env python3
"""Independent consistency check of vod2uv output.

Recomputes vorticity and divergence from u, v with finite differences
(no spectral code involved) and compares them with the input vo, d:
    zeta  = 1/(a cos phi) [ dv/dlam - d(u cos phi)/dphi ]
    delta = 1/(a cos phi) [ du/dlam + d(v cos phi)/dphi ]
Grid-scale features are badly represented by 2nd-order differences, so the
comparison is done after coarse-graining both fields by --coarsen points.
Expected for a correct conversion: slope ~ 1, correlation close to 1.
A sign error gives slope ~ -1; a radius / unit error gives slope != 1.

usage: python3 check_fd.py vod.nc uv.nc [--radius 6371229] [--latmax 80] [--coarsen 16]
Memory: processes latitude blocks; peak ~ a few x (block rows x nlon) doubles.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vod2uv_mpi as V

p = argparse.ArgumentParser()
p.add_argument("vodfile"); p.add_argument("uvfile")
p.add_argument("--radius", type=float, default=6371229.0)
p.add_argument("--latmax", type=float, default=80.0)
p.add_argument("--coarsen", type=int, default=16)
p.add_argument("--block", type=int, default=256, help="rows per block (rounded to a multiple of --coarsen)")
p.add_argument("--index", type=int, nargs="*", default=None, help="leading indices (time lev), default 0 0")
a = p.parse_args()

ri, ro = V.Reader(a.vodfile), V.Reader(a.uvfile)
dims = ri.dims("vo"); lead = ri.shape("vo")[:-2]
idx = tuple(a.index) if a.index else tuple(0 for _ in lead)
lat = ri.read(dims[-2], slice(None)).astype(float)
lon = ri.read(dims[-1], slice(None)).astype(float)
nlat, nlon = len(lat), len(lon)
phi = np.radians(lat); dlam = 2 * np.pi / nlon; A = a.radius
C = a.coarsen
a.block = max(C, (a.block // C) * C)

def rows(r, name, j0, j1):
    return r.read(name, idx + (slice(j0, j1), slice(None))).astype(np.float64)

def coarse(x):
    # C x C block average (low-pass) so that only well-resolved scales are compared
    m = (x.shape[0] // C) * C; n = (x.shape[1] // C) * C
    return x[:m, :n].reshape(m // C, C, n // C, C).mean(axis=(1, 3))

acc = {k: [] for k in ("zfd", "zin", "dfd", "din")}
zonal = []
ok = np.where(np.abs(lat) <= a.latmax)[0]
j_lo, j_hi = max(ok.min(), 1), min(ok.max(), nlat - 2)
for j0 in range(j_lo, j_hi + 1, a.block):
    j1 = min(j0 + a.block, j_hi + 1)
    u = rows(ro, "u", j0 - 1, j1 + 1); v = rows(ro, "v", j0 - 1, j1 + 1)
    vo = rows(ri, "vo", j0, j1); d = rows(ri, "d", j0, j1)
    ph = phi[j0 - 1:j1 + 1]; c = np.cos(ph)[:, None]
    # centred differences: longitude periodic, latitude non-uniform
    dvdl = (np.roll(v, -1, 1) - np.roll(v, 1, 1))[1:-1] / (2 * dlam)
    dudl = (np.roll(u, -1, 1) - np.roll(u, 1, 1))[1:-1] / (2 * dlam)
    dphi = (ph[2:] - ph[:-2])[:, None]
    ducp = ((u * c)[2:] - (u * c)[:-2]) / dphi
    dvcp = ((v * c)[2:] - (v * c)[:-2]) / dphi
    cc = c[1:-1]
    zfd = (dvdl - ducp) / (A * cc); dfd = (dudl + dvcp) / (A * cc)
    for k, x in (("zfd", zfd), ("zin", vo), ("dfd", dfd), ("din", d)):
        acc[k].append(coarse(x).ravel())
    zonal.append(np.stack((lat[j0:j1], u[1:-1].mean(1), v[1:-1].mean(1)), 1))

def stats(fd, ref, name):
    fd, ref = np.concatenate(fd), np.concatenate(ref)
    r = np.corrcoef(fd, ref)[0, 1]
    slope = np.dot(fd, ref) / np.dot(ref, ref)
    rel = np.sqrt(np.mean((fd - ref) ** 2) / np.mean(ref ** 2))
    print("%-10s corr=%.4f  slope(fd/input)=%.4f  rel.rms diff=%.3f  rms(input)=%.3e"
          % (name, r, slope, rel, np.sqrt(np.mean(ref ** 2))))
    return r, slope

print("index %s, |lat| <= %g, both fields block-averaged over %dx%d points" % (idx, a.latmax, C, C))
rz = stats(acc["zfd"], acc["zin"], "vorticity")
rd = stats(acc["dfd"], acc["din"], "divergence")
z = np.concatenate(zonal)
print("zonal means (lat, [u], [v]):")
for la in (60, 45, 30, 15, 0, -15, -30, -45, -60):
    i = np.argmin(np.abs(z[:, 0] - la))
    print("  %7.2f  %8.2f  %8.3f" % tuple(z[i]))
good = rz[0] > 0.95 and abs(rz[1] - 1) < 0.05
print("VERDICT (vorticity): %s" % ("CONSISTENT" if good else "CHECK"))
