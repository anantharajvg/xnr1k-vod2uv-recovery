"""Random band-limited psi/chi; reference u,v,vo,d built with scipy.special.sph_harm_y_all
(independent normalisation, recurrence and derivative code). Everything in float64."""
import sys, numpy as np, scipy.special as sp
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import vod2uv_mpi as V
A = 6371229.0
N = 128; nlat = 2*N; nlon = 4*N+16; L = N-1
rng = np.random.default_rng(1)
mu, w = V.gauss_legendre_north(nlat)
muall = np.concatenate((mu, -mu[::-1]))         # N->S
theta = np.arccos(muall)
lon = np.arange(nlon)*2*np.pi/nlon
Y, dY = sp.sph_harm_y_all(L, L, theta, np.zeros_like(theta), diff_n=1)   # (L+1, 2L+1, nlat)
Y = Y[:, :L+1]; dYt = dY[:, :L+1, :, 0]                                   # m >= 0
n = np.arange(L+1)[:, None]; m = np.arange(L+1)[None, :]
mask = (m <= n)
amp = 1.0/(1.0+n)**1.0
p = (rng.normal(size=(L+1, L+1)) + 1j*rng.normal(size=(L+1, L+1)))*mask*amp*1e7
q = (rng.normal(size=(L+1, L+1)) + 1j*rng.normal(size=(L+1, L+1)))*mask*amp*2e6
p[0] = q[0] = 0
E = np.exp(1j*np.outer(np.arange(L+1), lon))     # (m, lon)
def grid(coef_Y):   # coef_Y: (m, lat) complex Fourier amplitudes ; field = Re sum_m coef e^{im lam}
    return np.real(coef_Y.T @ E)
sint = np.sin(theta)
lap = -(n*(n+1))/A**2
vo = grid(np.einsum('nm,nmj->mj', lap*p, Y)); d = grid(np.einsum('nm,nmj->mj', lap*q, Y))
im = 1j*np.arange(L+1)[:, None]
u = grid(np.einsum('nm,nmj->mj', p, dYt)/A + im*np.einsum('nm,nmj->mj', q, Y)/(A*sint))
v = grid(im*np.einsum('nm,nmj->mj', p, Y)/(A*sint) - np.einsum('nm,nmj->mj', q, dYt)/A)
# --- our kernel (serial, double)
lmax = L; mmax = L
F = np.stack([np.fft.rfft(f, axis=-1)[:, :mmax+1]/nlon for f in (vo, d)])   # (2, nlat, M+1)
rowN = np.arange(N); rowS = np.arange(nlat-1, N-1, -1)
s = np.sqrt((1-mu)*(1+mu)); logs = np.log(s)
Um = np.empty_like(F)
for b0 in range(0, mmax+1, 16):
    mv = np.arange(b0, min(b0+16, mmax+1))
    UN, US = V.vod2uv_mbatch(mv, lmax, mu, logs, s, 0.5*w, F[:, rowN][:, :, mv].transpose(0, 2, 1), F[:, rowS][:, :, mv].transpose(0, 2, 1), A, 24)
    Um[:, rowN[:, None], mv[None, :]] = UN.transpose(0, 2, 1); Um[:, rowS[:, None], mv[None, :]] = US.transpose(0, 2, 1)
G = np.zeros((2, nlat, nlon//2+1), complex); G[:, :, :mmax+1] = Um*nlon
uu, vv = np.fft.irfft(G, n=nlon, axis=-1)
sc = max(abs(u).max(), abs(v).max())
print("lmax=%d random spectrum: max|u-ref|/max|V| = %.2e  v: %.2e   (max|V| = %.1f)" % (L, abs(uu-u).max()/sc, abs(vv-v).max()/sc, sc))
# sanity: sign of a pure solid-body rotation
