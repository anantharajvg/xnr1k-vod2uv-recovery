"""
Core of the XNR1K u/v recovery: Gaussian grids (regular or reduced), the
per-row wavenumber limit used by ecTrans, and batched Legendre analysis /
synthesis kernels.

The associated-Legendre recurrence, Gauss nodes and the extended-exponent
scheme are imported unchanged from vod2uv_mpi.py, where they were validated
(T7999 orthonormality <= 7e-13, scipy references to 1e-13).

Conventions
-----------
* Rows are stored north -> south, as in the XNR1K GRIB (jScansPositively=0).
* Each row starts at longitude lon0 (0 for ECMWF Gaussian grids).
* Pbar_n^m normalised so that  int_{-1}^{1} Pbar^2 dmu = 2  (sphere mean of |Y|^2 = 1).
* f(lambda, mu) = sum_{m=-M}^{M} sum_n f_n^m Pbar_n^m(mu) e^{i m lambda},  f_n^{-m} = conj(f_n^m).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vod2uv_mpi import (gauss_legendre_north, log_pmm_table, legendre_chunks,  # noqa: E402
                        _rstack)

_LPMM = {}


def lpmm(lmax):
    if lmax not in _LPMM:
        _LPMM[lmax] = log_pmm_table(lmax)
    return _LPMM[lmax]


# =============================================================================
# Grids
# =============================================================================
class GaussGrid:
    """Global Gaussian grid with 2N rows (north -> south) and nlon[j] points per row."""

    def __init__(self, N, nlon, name=None, lon0=0.0, mu=None, w=None):
        self.N = int(N)
        self.nlat = 2 * self.N
        self.nlon = np.asarray(nlon, dtype=np.int64)
        if self.nlon.shape != (self.nlat,):
            raise ValueError("nlon must have 2N entries")
        if not np.array_equal(self.nlon, self.nlon[::-1]):
            raise ValueError("row lengths are not symmetric about the equator")
        self.offsets = np.concatenate(([0], np.cumsum(self.nlon)))
        self.npts = int(self.offsets[-1])
        self.lon0 = float(lon0)
        self.name = name or ("grid N%d" % N)
        if mu is None:
            mu, w = gauss_legendre_north(self.nlat)
        self.mu_n = mu            # positive nodes, j = 0 nearest the north pole
        self.w_n = w
        self.reduced = bool(np.any(self.nlon != self.nlon[0]))
        # file row -> (pair index, hemisphere); rows N..2N-1 mirror rows N-1..0
        j = np.arange(self.nlat)
        self.pair = np.where(j < self.N, j, self.nlat - 1 - j)
        self.rowN = np.arange(self.N)
        self.rowS = self.nlat - 1 - np.arange(self.N)

    # --- constructors -------------------------------------------------------
    @classmethod
    def regular(cls, N):
        return cls(N, np.full(2 * N, 4 * N), name="F%d" % N)

    @classmethod
    def octahedral(cls, N):
        north = 20 + 4 * np.arange(N)
        return cls(N, np.concatenate((north, north[::-1])), name="O%d" % N)

    @classmethod
    def from_pl(cls, N, pl, name=None):
        return cls(N, np.asarray(pl), name=name or "reduced N%d" % N)

    @classmethod
    def from_spec(cls, spec):
        """'F320', 'O8000', 'N320' needs a pl array and is not supported here."""
        kind, N = spec[0].upper(), int(spec[1:])
        if kind == "F":
            return cls.regular(N)
        if kind == "O":
            return cls.octahedral(N)
        raise ValueError("grid spec must be F<N> or O<N>: %s" % spec)

    # --- geometry -----------------------------------------------------------
    @property
    def lat_deg(self):
        lat = np.degrees(np.arcsin(self.mu_n))
        return np.concatenate((lat, -lat[::-1]))

    def cos2(self):
        """cos^2(latitude) per file row (ecTrans F%R1MU2)."""
        c = (1.0 - self.mu_n) * (1.0 + self.mu_n)
        return np.concatenate((c, c[::-1]))

    def nmen(self, T):
        """Per-row maximum zonal wavenumber, as ecTrans setup_geom_mod computes it.

        linear or full grid : min(T, (nlon-1)//2)
        quadratic grid      : running max of int((nlon-1)/(2+zsqm2)),       zsqm2 = (3*(NDGL-1-T)//NDGL)*cos^2
        cubic grid          : running max of int((nlon-1)/(2+cos^2)) - 1
        (running max taken from the pole towards the equator, in each hemisphere)
        """
        T = int(T)
        ndgl = self.nlat
        nsmaxlin = ndgl - 1
        if T >= nsmaxlin or not self.reduced:
            return np.minimum(T, (self.nlon - 1) // 2)
        cos2 = self.cos2()
        if T >= ndgl * 2 // 3 - 1:          # quadratic
            zsqm2 = (3 * (nsmaxlin - T) // ndgl) * cos2   # Fortran integer arithmetic, then * real
            dec = 0
        else:                               # cubic
            zsqm2 = cos2
            dec = 1
        raw = np.floor((self.nlon - 1).astype(np.float64) / (2.0 + zsqm2)).astype(np.int64) - dec
        out = np.empty(ndgl, dtype=np.int64)
        nh = self.N
        out[0] = min(T, raw[0])
        for j in range(1, nh):
            out[j] = min(T, max(out[j - 1], raw[j]))
        out[ndgl - 1] = min(T, raw[ndgl - 1])
        for j in range(ndgl - 2, nh - 1, -1):
            out[j] = min(T, max(out[j + 1], raw[j]))
        return out

    def area_weights(self):
        """Weight per grid point (sum = 1): Gauss weight / (2 nlon)."""
        w = np.concatenate((self.w_n, self.w_n[::-1])) / 2.0
        return np.repeat(w / self.nlon, self.nlon)


# =============================================================================
# Fourier step (per row, variable length)
# =============================================================================
def rows_to_fourier(values, grid, r0, r1, nmen, mmax):
    """values: 1-D array of the points of rows r0..r1-1 (concatenated).
    Returns F (r1-r0, mmax+1) complex with F_m = (1/nlon) sum f e^{-i m (lambda - lon0)} ... e^{-i m lon0}
    and zeros for m > nmen[row]."""
    F = np.zeros((r1 - r0, mmax + 1), np.complex128)
    base = grid.offsets[r0]
    lam0 = math.radians(grid.lon0)
    for i, r in enumerate(range(r0, r1)):
        a, b = grid.offsets[r] - base, grid.offsets[r + 1] - base
        nl = b - a
        k = int(min(nmen[r], mmax))
        c = np.fft.rfft(values[a:b])[:k + 1] / nl
        if lam0 != 0.0:
            c = c * np.exp(-1j * np.arange(k + 1) * lam0)
        F[i, :k + 1] = c
    return F


def fourier_to_rows(G, grid, r0, r1, nmen):
    """G: (r1-r0, mmax+1) complex Fourier coefficients; returns concatenated row values."""
    out = np.empty(int(grid.offsets[r1] - grid.offsets[r0]))
    base = grid.offsets[r0]
    lam0 = math.radians(grid.lon0)
    for i, r in enumerate(range(r0, r1)):
        nl = int(grid.nlon[r])
        k = int(min(nmen[r], G.shape[1] - 1))
        buf = np.zeros(nl // 2 + 1, np.complex128)
        c = G[i, :k + 1]
        if lam0 != 0.0:
            c = c * np.exp(1j * np.arange(k + 1) * lam0)
        buf[:k + 1] = c * nl
        a = grid.offsets[r] - base
        out[a:a + nl] = np.fft.irfft(buf, n=nl)
    return out


# =============================================================================
# Legendre kernels for one batch of zonal wavenumbers
# =============================================================================
def analysis(mvals, lmax, mu, logs, wh, FN, FS, kchunk=64):
    """Gauss-Legendre analysis.  FN, FS: (C, B, J) complex at rows +mu_j, -mu_j.
    wh = 0.5 * Gauss weights (J,).  Returns alm (C, B, nk) complex, n = m + k."""
    C, B, J = FN.shape
    nk = lmax - int(mvals.min()) + 1
    Gs = _rstack(np.transpose((FN + FS) * wh, (1, 2, 0)), 2)   # (B, J, 2C)
    Ga = _rstack(np.transpose((FN - FS) * wh, (1, 2, 0)), 2)
    alm = np.zeros((B, nk, 2 * C))
    for k0, P, _ in legendre_chunks(mvals, lmax, mu, logs, lpmm(lmax), kchunk, False):
        K = P.shape[1]
        e0 = k0 % 2
        o0 = 1 - e0
        alm[:, k0 + e0:k0 + K:2] = np.matmul(P[:, e0::2], Gs)
        alm[:, k0 + o0:k0 + K:2] = np.matmul(P[:, o0::2], Ga)
    out = np.empty((C, B, nk), np.complex128)
    for c in range(C):
        out[c] = alm[:, :, 2 * c] + 1j * alm[:, :, 2 * c + 1]
    n = mvals[:, None] + np.arange(nk)[None, :]
    out[:, n > lmax] = 0.0
    return out


def synthesis(mvals, lmax, mu, logs, cP, cH=None, kchunk=64):
    """Legendre synthesis of  sum_n [ cP Pbar_n^m + cH H_n^m ]  at +mu_j and -mu_j.
    cP, cH: (K, B, nk) complex (cH may be None).  Returns (xN, xS) each (K, B, J) complex."""
    K, B, nk = cP.shape
    J = len(mu)
    CP = _rstack(np.transpose(cP, (1, 0, 2)), 1)            # (B, 2K, nk)
    CH = _rstack(np.transpose(cH, (1, 0, 2)), 1) if cH is not None else None
    Sym = np.zeros((B, 2 * K, J))
    Anti = np.zeros((B, 2 * K, J))
    for k0, P, H in legendre_chunks(mvals, lmax, mu, logs, lpmm(lmax), kchunk, CH is not None):
        Kc = P.shape[1]
        e0 = k0 % 2
        o0 = 1 - e0
        ke = slice(k0 + e0, k0 + Kc, 2)
        ko = slice(k0 + o0, k0 + Kc, 2)
        Sym += np.matmul(CP[:, :, ke], P[:, e0::2])
        Anti += np.matmul(CP[:, :, ko], P[:, o0::2])
        if CH is not None:
            Sym += np.matmul(CH[:, :, ko], H[:, o0::2])
            Anti += np.matmul(CH[:, :, ke], H[:, e0::2])
    rN, rS = Sym + Anti, Sym - Anti
    xN = np.stack([rN[:, 2 * k] + 1j * rN[:, 2 * k + 1] for k in range(K)])
    xS = np.stack([rS[:, 2 * k] + 1j * rS[:, 2 * k + 1] for k in range(K)])
    return xN, xS


def vod_to_uv_coeffs(mvals, lmax, zeta, delta, radius):
    """From vorticity/divergence coefficients (B, nk) build the P and H coefficients of
    u cos(phi), v cos(phi):  u cos = (1/a) sum[-psi H + i m chi P],  v cos = (1/a) sum[i m psi P + chi H]."""
    nk = zeta.shape[1]
    n = mvals[:, None].astype(np.float64) + np.arange(nk)[None, :]
    nn1 = n * (n + 1)
    f = np.where((n > 0) & (n <= lmax), radius / np.where(nn1 > 0, nn1, 1), 0.0)
    psi = -f * zeta          # psi / a
    chi = -f * delta         # chi / a
    im = 1j * mvals[:, None].astype(np.float64)
    cP = np.stack((im * chi, im * psi))
    cH = np.stack((-psi, chi))
    return cP, cH


def process_mbatch(mvals, lmax, gin, gout, FN, FS, radius, want_vod=False, kchunk=64):
    """Full per-batch pipeline: analysis on the input grid nodes, u/v (and optionally
    truncated vo/d) synthesis on the output grid nodes.
    Returns (outN, outS, alm): out* (K, B, Jout) complex, K = 2 (u, v) or 4 (u, v, vo, d)."""
    mu_i, w_i = gin.mu_n, gin.w_n
    logs_i = np.log(np.sqrt((1 - mu_i) * (1 + mu_i)))
    alm = analysis(mvals, lmax, mu_i, logs_i, 0.5 * w_i, FN, FS, kchunk)
    cP, cH = vod_to_uv_coeffs(mvals, lmax, alm[0], alm[1], radius)
    mu_o = gout.mu_n
    s_o = np.sqrt((1 - mu_o) * (1 + mu_o))
    logs_o = np.log(s_o)
    uN, uS = synthesis(mvals, lmax, mu_o, logs_o, cP, cH, kchunk)
    uN /= s_o
    uS /= s_o
    if want_vod:
        vN, vS = synthesis(mvals, lmax, mu_o, logs_o, alm[:2], None, kchunk)
        uN = np.concatenate((uN, vN))
        uS = np.concatenate((uS, vS))
    return uN, uS, alm
