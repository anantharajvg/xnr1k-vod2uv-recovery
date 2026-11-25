import sys, time, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import vod2uv_mpi as V
nlat, lmax = 16000, 7999
t = time.time(); mu, w = V.gauss_legendre_north(nlat); print("GL(16000) %.1fs  2*sum(w)-2 = %.1e" % (time.time()-t, 2*w.sum()-2))
np.save('/tmp/gl16000.npy', np.stack((mu, w)))
logs = np.log(np.sqrt((1-mu)*(1+mu))); lp = V.log_pmm_table(lmax)
for m in (0, 1, 1500, 4000, 6500, 7999):
    t = time.time()
    rows = []
    for k0, P, _ in V.legendre_chunks(np.array([m]), lmax, mu, logs, lp, 256, False):
        rows.append(P[0])
    P = np.concatenate(rows)[: lmax - m + 1]
    tg = time.time() - t
    worst = 0.0
    for par in (0, 1):
        Q = P[par::2]
        if len(Q) == 0: continue
        G = (Q * w) @ Q.T
        worst = max(worst, np.abs(G - np.eye(len(Q))).max())
    print("m=%5d nk=%5d  max|Gram - I| = %.2e   (recurrence %.1fs)  any nonfinite: %s" % (m, len(P), worst, tg, (~np.isfinite(P)).any()))
