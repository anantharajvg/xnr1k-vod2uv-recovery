import os, sys, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(D)); sys.path.insert(0, os.path.dirname(os.path.dirname(D)))
import core as C, vod2uv_mpi as V
# 1. regression: new kernel == validated old kernel on a regular grid
g = C.GaussGrid.regular(64); L = 63
rng = np.random.default_rng(3); mv = np.arange(5, 9)
FN = rng.normal(size=(2, 4, 64)) + 1j*rng.normal(size=(2, 4, 64)); FS = rng.normal(size=(2, 4, 64)) + 1j*rng.normal(size=(2, 4, 64))
s = np.sqrt((1-g.mu_n)*(1+g.mu_n))
old = V.vod2uv_mbatch(mv, L, g.mu_n, np.log(s), s, 0.5*g.w_n, FN, FS, 6371229.0, 16)
new = C.process_mbatch(mv, L, g, g, FN, FS, 6371229.0, kchunk=16)
print("regression vs validated kernel: max|diff| = %.2e (scale %.2e)" % (max(abs(old[0]-new[0]).max(), abs(old[1]-new[1]).max()), abs(old[0]).max()))
# 2. ecTrans row limits
for N, T in ((8000, 7999), (1280, 1279), (48, 47)):
    o = C.GaussGrid.octahedral(N); nm = o.nmen(T)
    j_full = int(np.argmax(nm == T))
    print("O%d T%d: nlon[0]=%d nmen[0]=%d  nmen[1]=%d  first row with nmen=T: %d (lat %.2f, nlon %d)  monotone N: %s  symmetric: %s"
          % (N, T, o.nlon[0], nm[0], nm[1], j_full, o.lat_deg[j_full], o.nlon[j_full], bool(np.all(np.diff(nm[:N]) >= 0)), bool(np.array_equal(nm, nm[::-1]))))
    assert np.all(nm <= (o.nlon - 1)//2)
f = C.GaussGrid.regular(320); print("F320 T639 nmen range:", f.nmen(639).min(), f.nmen(639).max())
print("O8000 points:", C.GaussGrid.octahedral(8000).npts)
