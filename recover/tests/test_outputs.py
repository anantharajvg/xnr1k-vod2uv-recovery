"""Non-native output (F grid, truncated, with vo/d), spectral output, and restart skip."""
import os, sys, tempfile, io, contextlib
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(D)); sys.path.insert(0, D)
import core as C, serial as S, comm as CM, recover_uv as R, gribio as G
from test_m7_driver import make_source
ok = True
gin = C.GaussGrid.octahedral(96); gout = C.GaussGrid.regular(40)
src, meta, exact = make_source(gin, {1: (20, 80), 2: (30,)})
out = tempfile.mkdtemp()
argv = ["--out-dir", out, "--levels", "1-2", "--out-grid", "F40", "--lmax", "79", "--write-vod", "--spec-out", "--no-merge", "--quiet"]
CM.run_threads(lambda c: R.run(R.parse_args(argv), c, src, G.NpzSink(out)), 4)
for L in (1, 2):
    z = np.load(os.path.join(out, "parts", "L%03d.npz" % L))
    ref = S.recover_serial(src.fields["vo"][L], src.fields["d"][L], gin, gout, 79, want_vod=True)
    d = max(abs(z[n] - r.astype(np.float32)).max() for n, r in zip(("u", "v", "vo", "d"), ref))
    print("level %d F40/T79 output vs serial (float32 parts): max diff %.1e" % (L, d)); ok &= d == 0
    sp = np.load(os.path.join(out, "parts", "spec_L%03d.npz" % L))
    nexp = sum(79 - m + 1 for m in range(80))
    print("  spectral coefficients: %d (expected %d), lmax %d" % (sp["zeta"].size, nexp, sp["lmax"])); ok &= sp["zeta"].size == nexp
# spectral consistency: synthesise vo from saved zeta and compare with truncated vo part
sp = np.load(os.path.join(out, "parts", "spec_L002.npz")); z = np.load(os.path.join(out, "parts", "L002.npz"))
coef = sp["zeta"]; nm = gout.nmen(79); G2 = np.zeros((gout.nlat, 80), complex); off = 0
lg = np.log(np.sqrt((1 - gout.mu_n) * (1 + gout.mu_n)))
for m in range(80):
    c = coef[off:off + 80 - m]; off += 80 - m
    xN, xS = C.synthesis(np.array([m]), 79, gout.mu_n, lg, c[None, None, :])
    G2[gout.rowN, m] = xN[0, 0]; G2[gout.rowS, m] = xS[0, 0]
vo_s = C.fourier_to_rows(G2, gout, 0, gout.nlat, nm)
e = abs(vo_s - z["vo"]).max() / abs(z["vo"]).max(); print("vo from saved coefficients vs vo output: %.1e" % e); ok &= e < 1e-6
# restart: second run must skip both levels
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    R.run(R.parse_args([a for a in argv if a != "--quiet"]), CM.SerialComm(), src, G.NpzSink(out))
skips = buf.getvalue().count("skipped"); print("restart run skipped %d/2 levels" % skips); ok &= skips == 2
print("outputs:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
