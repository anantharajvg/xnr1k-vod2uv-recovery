import sys, math, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import vod2uv_mpi as V
from scipy.special import lpmv, gammaln

# 1. Gauss nodes vs numpy (independent eig-based)
for n in (16, 96, 320):
    x, w = V.gauss_legendre_north(n)
    xr, wr = np.polynomial.legendre.leggauss(n)
    xr, wr = xr[::-1][:n//2], wr[::-1][:n//2]
    print("gauss n=%d  max|dx|=%.1e  max rel dw=%.1e  sum w=%.15f" % (n, abs(x-xr).max(), (abs(w-wr)/wr).max(), 2*w.sum()))

# 2. Pbar and H vs scipy lpmv (unnormalised, with CS phase removed)
lmax = 60
x, w = V.gauss_legendre_north(128)
logs = np.log(np.sqrt((1-x)*(1+x)))
lp = V.log_pmm_table(lmax)
err = 0; errH = 0
for m in (0, 1, 7, 30, 60):
    for k0, P, H in V.legendre_chunks(np.array([m]), lmax, x, logs, lp, 16, True):
        for i in range(P.shape[1]):
            n = m + k0 + i
            if n > lmax: continue
            N = math.sqrt((2*n+1)) * math.exp(0.5*(gammaln(n-m+1)-gammaln(n+m+1)))
            ref = (-1)**m * lpmv(m, n, x) * N
            # H via derivative identity (unnormalised): (1-x^2)P' = -n x P_n + (n+m) P_{n-1}
            refm1 = (-1)**m * lpmv(m, n-1, x) if n > m else 0*x
            Nm1 = math.sqrt(2*n+1)*math.exp(0.5*(gammaln(n-m+1)-gammaln(n+m+1)))
            refH = (-n*x*(-1)**m*lpmv(m, n, x) + (n+m)*refm1) * Nm1
            sc = max(1, abs(ref).max())
            err = max(err, abs(P[0, i]-ref).max()/sc)
            errH = max(errH, abs(H[0, i]-refH).max()/max(1, abs(refH).max()))
print("Pbar vs scipy (n<=60): max rel err %.1e ; H: %.1e" % (err, errH))
