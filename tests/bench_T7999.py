import sys, time, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import vod2uv_mpi as V
mu, w = np.load('/tmp/gl16000.npy'); lmax = 7999
s = np.sqrt((1-mu)*(1+mu)); logs = np.log(s); J = len(mu)
rng = np.random.default_rng(0)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
res = []
for m0 in ((0, 4000) if B > 16 else (0, 4000, 7000)):
    mv = np.arange(m0, m0+B)
    FN = rng.normal(size=(2, B, J)) + 1j*rng.normal(size=(2, B, J)); FS = FN.copy()
    t = time.time(); UN, US = V.vod2uv_mbatch(mv, lmax, mu, logs, s, 0.5*w, FN*1e-5, FS*1e-5, 6371229.0, 64); dt = time.time()-t
    print("B=%d m0=%d nk=%d : %.1fs  (%.2f us per (m,n) pair)  finite=%s" % (B, m0, lmax-m0+1, dt, dt/(B*(lmax-m0+1))*1e6, np.isfinite(UN).all()))
    res.append(dt/(B*(lmax-m0+1)))
npairs = (lmax+1)*(lmax+2)/2
print("extrapolated serial Legendre cost for one 2D (vo,d)->(u,v) slice: %.0f core-s" % (np.mean(res)*npairs))
