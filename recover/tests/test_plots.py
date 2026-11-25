"""Smoke + correctness tests for compare.py and plots.py on synthetic data."""
import os, sys, subprocess, csv
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); R0 = os.path.dirname(D)
sys.path.insert(0, R0); sys.path.insert(0, D)
from scipy.io import netcdf_file
import core as C, serial as S, comm as CM, recover_uv as R, gribio as G
from test_m7_driver import make_source
from test_m1_analytic import fields
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pt"; os.makedirs(out, exist_ok=True)
py = [sys.executable]
def sh(*a):
    r = subprocess.run(py + list(a), capture_output=True, text=True, cwd=R0)
    print(r.stdout.strip()[-600:]); 
    if r.returncode: print(r.stderr[-2000:]); raise SystemExit("FAILED: %s" % " ".join(a))
# --- synthetic A (truth) and B = A + known bias and noise, regular F80, 10 levels, NCAR-like layout for B
g = C.GaussGrid.regular(80); la, lo = S.point_coords(g); nlev = 10
rng = np.random.default_rng(0)
U = np.zeros((nlev, g.nlat, 4 * 80)); V = np.zeros_like(U)
for k in range(nlev):
    f = fields(la, lo, 6 + k, ph=0.1 * k)
    U[k] = f[2].reshape(g.nlat, -1) * (0.5 + 0.1 * k); V[k] = f[3].reshape(g.nlat, -1) * (0.5 + 0.1 * k)
bias = np.linspace(-1, 1, nlev); noise = np.linspace(0.5, 2, nlev); noise[6] = 8.0      # level 7 is an outlier
UB = U + bias[:, None, None] + noise[:, None, None] * rng.normal(size=U.shape); VB = V + noise[:, None, None] * rng.normal(size=U.shape)
def write(path, names, arrs, time=False, ab=False):
    f = netcdf_file(path, "w", version=2)
    if time: f.createDimension("time", 2)
    f.createDimension("level", nlev); f.createDimension("latitude", g.nlat); f.createDimension("longitude", 320)
    f.createVariable("level", "i", ("level",))[:] = np.arange(1, nlev + 1)
    f.createVariable("latitude", "d", ("latitude",))[:] = g.lat_deg
    f.createVariable("longitude", "d", ("longitude",))[:] = np.arange(320) * 360 / 320
    if time:
        t = f.createVariable("time", "d", ("time",)); t[:] = [0, 1]; t.units = "hours since 2018-11-01"
    for n, x in zip(names, arrs):
        dims = (("time",) if time else ()) + ("level", "latitude", "longitude")
        v = f.createVariable(n, "f", dims); v[:] = np.stack([x, x * 0 + 999]) if time else x
    if ab:
        f.createVariable("a_model", "d", ("level",))[:] = np.linspace(1000, 20000, nlev)
        f.createVariable("b_model", "d", ("level",))[:] = np.linspace(0, 0.8, nlev)
    f.close()
write(out + "/A.nc", ["u", "v"], [U, V])
write(out + "/B_u.nc", ["U"], [UB], time=True, ab=True); write(out + "/B_v.nc", ["V"], [VB], time=True)
f = netcdf_file(out + "/B_sp.nc", "w", version=2); f.createDimension("time", 2); f.createDimension("latitude", g.nlat); f.createDimension("longitude", 320)
f.createVariable("latitude", "d", ("latitude",))[:] = g.lat_deg; f.createVariable("longitude", "d", ("longitude",))[:] = np.arange(320) * 1.125
f.createVariable("SP", "f", ("time", "latitude", "longitude"))[:] = 98500.0; f.close()
sh("compare.py", "--a-u", out + "/A.nc:u", "--a-v", out + "/A.nc:v", "--b-u", out + "/B_u.nc:U", "--b-v", out + "/B_v.nc:V",
   "--b-time", "0", "--b-sp", out + "/B_sp.nc:SP", "--b-ab", out + "/B_u.nc", "--out", out + "/cmp")
rows = [r for r in csv.DictReader(open(out + "/cmp_stats.csv")) if r["var"] == "u"]
eb = max(abs(float(r["bias"]) + bias[int(r["level"]) - 1]) for r in rows)
en = max(abs(float(r["rmsd"]) - np.hypot(bias[int(r["level"]) - 1], noise[int(r["level"]) - 1])) / noise[int(r["level"]) - 1] for r in rows)
print("compare: max |bias - (-true bias)| = %.3f m/s, max relative rmsd error = %.3f" % (eb, en))
assert eb < 0.05 and en < 0.02, "compare statistics wrong"
# --- plots
sh("plots.py", "profiles", "--csv", out + "/cmp_stats.csv", "--out", out + "/profiles.png")
sh("plots.py", "zonal", "--npz", out + "/cmp_zonal.npz", "--level", "5", "--out", out + "/zonal.png")
sh("plots.py", "map", "--nc", out + "/A.nc:u", "--level", "5", "--out", out + "/map_u.png")
sh("plots.py", "map", "--nc", out + "/A.nc:u", "--minus-nc", out + "/B_u.nc:U", "--minus-time", "0", "--level", "5", "--out", out + "/map_du.png")
sh("plots.py", "map", "--nc", out + "/A.nc:u", "--nc-v", out + "/A.nc:v", "--speed", "--level", "5",
   "--region", "-60", "60", "0", "70", "--width", "600", "--out", out + "/map_speed_region.png")
# --- reduced-grid output + spectra through the real driver
gin = C.GaussGrid.octahedral(96); src, meta, ex = make_source(gin, {1: (8, 60), 2: (30,)})
argv = ["--out-dir", out + "/run", "--levels", "1-2", "--out-grid", "O48", "--lmax", "95", "--write-vod", "--spec-out", "--quiet"]
CM.run_threads(lambda c: R.run(R.parse_args(argv), c, src, G.NpzSink(out + "/run")), 3)
sh("plots.py", "map", "--npz", out + "/run/recovered_O48_T95.npz:u", "--level", "1", "--width", "720", "--out", out + "/map_O48_u.png")
z = np.load(out + "/run/parts/spec_L001.npz")
pairs = np.stack([np.concatenate([z["zeta"].real]), z["zeta"].imag], 1).ravel()
np.savez(out + "/fake_arco.npz", levels=np.array([1]), T=int(z["lmax"]), vo=pairs[None].astype(np.float32),
         d=np.stack([z["delta"].real, z["delta"].imag], 1).ravel()[None].astype(np.float32))
sh("plots.py", "spectra", "--spec", out + "/run/parts/spec_L001.npz:XNR1K-test", "--arco", out + "/fake_arco.npz:1:ARCO-test",
   "--level", "1", "--out", out + "/spectra.png")
# KE from spectrum must equal 0.5 * area-mean(u^2 + v^2) of the recovered field (Parseval check)
import plots as P
er, ed = P.ke_spectrum(z["zeta"], z["delta"], int(z["lmax"]))
uv = np.load(out + "/run/parts/L001.npz"); go = C.GaussGrid.octahedral(48); w = go.area_weights()
ke = 0.5 * np.dot(w, uv["u"].astype(float) ** 2 + uv["v"].astype(float) ** 2)
print("Parseval: sum of spectrum %.6e vs grid KE %.6e (rel diff %.1e)" % (er.sum() + ed.sum(), ke, abs(er.sum() + ed.sum() - ke) / ke))
assert abs(er.sum() + ed.sum() - ke) / ke < 1e-4
sh("plots.py", "rowlimits", "--N", "8000", "--out", out + "/rowlimits.png")
print("plots tests: PASS; figures in", out)
