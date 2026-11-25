#!/usr/bin/env python3
"""
vod2uv_mpi.py -- MPI-distributed conversion of relative vorticity and divergence
on a (regular) Gaussian grid to u and v wind components.

Method (same mathematics as IFS / CDO dv2uv, different memory strategy):

  1. grid -> Fourier   : real FFT along each latitude row      (rows distributed)
  2. global transpose  : MPI_Alltoallv  rows  <->  zonal wavenumbers m
  3. Fourier -> spectral: Gauss-Legendre quadrature per m         (m distributed)
       zeta_n^m, delta_n^m
  4. psi_n^m = -a^2 zeta_n^m / (n(n+1)),   chi_n^m = -a^2 delta_n^m / (n(n+1))
  5. spectral -> Fourier for
       u cos(phi) = (1/a) sum [ -psi H_n^m + i m chi P_n^m ] e^{i m lambda}
       v cos(phi) = (1/a) sum [  i m psi P_n^m + chi H_n^m ] e^{i m lambda}
     with H_n^m = (1-mu^2) dP_n^m/dmu
  6. transpose back and inverse FFT                              (rows distributed)

Associated Legendre functions are generated on the fly (never stored as a
table) by the standard stable three-term recurrence in n, with an extended
exponent per (m, latitude) so that P_m^m ~ sin(theta)^m does not underflow at
high m near the poles (needed beyond m ~ 1500-2000 in IEEE double).
Equatorial symmetry halves the Legendre work.

Memory per rank is O(N_grid / N_ranks) plus small work arrays; nothing scales
with lmax^2 * nlat as a precomputed Legendre table does (the reason CDO fails).

Requirements: numpy, mpi4py (optional: runs serially without it),
              netCDF4 (preferred) or scipy (fallback, netCDF-3 only, small files).

Usage:
  mpirun -n 256 python vod2uv_mpi.py vod.rgg.nc uv.rgg.nc
  python vod2uv_mpi.py --help
"""

import argparse
import math
import sys
import time

import numpy as np

LN2 = math.log(2.0)
RESCALE_EXP = 500           # rescale mantissas when |p| > 2**RESCALE_EXP
RESCALE_THR = 2.0 ** RESCALE_EXP
RESCALE_FAC = 2.0 ** (-RESCALE_EXP)


# =============================================================================
# Communicator abstraction (mpi4py, or serial fallback)
# =============================================================================
class SerialComm:
    rank = 0
    size = 1

    def bcast(self, obj, root=0):
        return obj

    def barrier(self):
        pass

    def allreduce_max(self, x):
        return x

    def alltoallv(self, sendbuf, scounts, recvbuf, rcounts):
        recvbuf[: int(rcounts[0])] = sendbuf[: int(scounts[0])]

    def Send(self, arr, dest, tag):
        raise RuntimeError("Send on a serial communicator")

    def Recv(self, arr, source, tag):
        raise RuntimeError("Recv on a serial communicator")


class MPIComm:
    def __init__(self, comm):
        from mpi4py import MPI
        self.MPI = MPI
        self.c = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

    def bcast(self, obj, root=0):
        return self.c.bcast(obj, root=root)

    def barrier(self):
        self.c.Barrier()

    def allreduce_max(self, x):
        return self.c.allreduce(x, op=self.MPI.MAX)

    def alltoallv(self, sendbuf, scounts, recvbuf, rcounts):
        lim = 2 ** 31 - 1
        sd = np.concatenate(([0], np.cumsum(scounts)[:-1]))
        rd = np.concatenate(([0], np.cumsum(rcounts)[:-1]))
        if max(sd.max() + max(scounts), rd.max() + max(rcounts)) > lim:
            raise RuntimeError("alltoallv message larger than 2^31 elements; "
                               "use more MPI ranks")
        sc = [int(x) for x in scounts]
        rc = [int(x) for x in rcounts]
        sd = [int(x) for x in sd]
        rd = [int(x) for x in rd]
        T = self.MPI.DOUBLE_COMPLEX
        self.c.Alltoallv([sendbuf, (sc, sd), T], [recvbuf, (rc, rd), T])

    def Send(self, arr, dest, tag):
        self.c.Send(np.ascontiguousarray(arr), dest=dest, tag=tag)

    def Recv(self, arr, source, tag):
        self.c.Recv(arr, source=source, tag=tag)


def _launcher_ntasks():
    import os
    for k in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
        v = os.environ.get(k)
        if v and v.isdigit():
            return int(v)
    return 1


def get_comm():
    try:
        from mpi4py import MPI
        comm = MPIComm(MPI.COMM_WORLD)
    except ImportError:
        comm = SerialComm()
    nt = _launcher_ntasks()
    if comm.size == 1 and nt > 1:
        raise SystemExit("ERROR: launcher started %d tasks but MPI reports size 1 "
                         "(mpi4py missing or built against a different MPI). Aborting to "
                         "avoid %d independent serial runs." % (nt, nt))
    return comm


# =============================================================================
# Gauss-Legendre nodes / weights
# =============================================================================
def gauss_legendre_north(nlat, tol=1e-15, maxit=100):
    """Positive Gauss nodes mu_j (descending, j=0 nearest north pole) and
    weights (full-sphere weights sum to 2). nlat must be even."""
    if nlat % 2:
        raise ValueError("odd number of Gaussian latitudes not supported")
    nh = nlat // 2
    k = np.arange(1, nh + 1, dtype=np.float64)
    # Tricomi initial guess
    x = (1.0 - (nlat - 1) / (8.0 * nlat ** 3)) * np.cos(np.pi * (4 * k - 1) / (4 * nlat + 2))
    for it in range(maxit):
        p0 = np.ones_like(x)
        p1 = x.copy()
        for j in range(2, nlat + 1):
            p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
        dp = nlat * (p0 - x * p1) / ((1.0 - x) * (1.0 + x))
        dx = p1 / dp
        x -= dx
        if np.max(np.abs(dx)) < tol:
            break
    # final derivative at converged nodes
    p0 = np.ones_like(x)
    p1 = x.copy()
    for j in range(2, nlat + 1):
        p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
    dp = nlat * (p0 - x * p1) / ((1.0 - x) * (1.0 + x))
    w = 2.0 / ((1.0 - x) * (1.0 + x) * dp * dp)
    return x, w


# =============================================================================
# Normalised associated Legendre functions, generated on the fly
#   Pbar_n^m normalised so that  int_{-1}^{1} Pbar^2 dmu = 2   (IFS convention)
# =============================================================================
def log_pmm_table(lmax):
    """log Pbar_m^m / sin^m(theta) for m = 0..lmax."""
    m = np.arange(1, lmax + 1, dtype=np.float64)
    t = np.zeros(lmax + 1)
    t[1:] = np.cumsum(0.5 * np.log((2 * m + 1) / (2 * m)))
    return t


def legendre_chunks(mvals, lmax, mu, logs, lpmm, kchunk, want_H):
    """Yield (k0, P, H) where P[b, i, j] = Pbar_{m_b + k0 + i}^{m_b}(mu_j)
    (zero/garbage for n > lmax, caller masks), H = (1-mu^2) dPbar/dmu."""
    B = len(mvals)
    J = len(mu)
    m = mvals.astype(np.float64)[:, None]
    L = lpmm[mvals][:, None] + m * logs[None, :]
    E = np.floor(L / LN2)
    pc = np.exp(L - E * LN2)            # mantissa of Pbar_m^m in [1,2)
    E = E.astype(np.int32)
    pp = np.zeros_like(pc)
    mu2 = mu[None, :]
    nk = lmax - int(mvals.min()) + 1
    for k0 in range(0, nk, kchunk):
        K = min(kchunk, nk - k0)
        P = np.empty((B, K, J))
        H = np.empty((B, K, J)) if want_H else None
        for i in range(K):
            k = k0 + i
            n = m + k
            if k > 0:
                valid = n <= lmax
                nn = n * n - m * m
                a = np.where(valid, np.sqrt((4 * n * n - 1) / np.where(nn > 0, nn, 1)), 0.0)
                if k > 1:
                    b = np.where(valid, np.sqrt((2 * n + 1) * (n - 1 - m) * (n - 1 + m)
                                                / ((2 * n - 3) * np.where(nn > 0, nn, 1))), 0.0)
                    pn = a * mu2 * pc - b * pp
                else:
                    pn = a * mu2 * pc
                pp = pc
                pc = pn
                if (k & 3) == 0:
                    big = (np.abs(pc) > RESCALE_THR) | (np.abs(pp) > RESCALE_THR)
                    if big.any():
                        pc[big] *= RESCALE_FAC
                        pp[big] *= RESCALE_FAC
                        E[big] += RESCALE_EXP
            P[:, i, :] = np.ldexp(pc, E)
            if want_H:
                if k > 0:
                    c = np.sqrt((2 * n + 1) * (n * n - m * m) / (2 * n - 1))
                    H[:, i, :] = np.ldexp(-n * mu2 * pc + c * pp, E)
                else:
                    H[:, i, :] = np.ldexp(-n * mu2 * pc, E)
        yield k0, P, H


def _rstack(z, axis):
    """complex array -> real array with (Re, Im) interleaved along `axis`."""
    r = np.stack((z.real, z.imag), axis=axis + 1)
    shp = list(z.shape)
    shp[axis] *= 2
    return r.reshape(shp)


def vod2uv_mbatch(mvals, lmax, mu, logs, s, wh, FN, FS, radius, kchunk=64):
    """Core kernel for one batch of zonal wavenumbers.

    FN, FS : (2, B, J) complex Fourier coefficients of (vo, d) at northern
             (mu_j) and southern (-mu_j) Gaussian rows.
    wh     : 0.5 * Gauss weights (J,)
    returns UN, US : (2, B, J) complex Fourier coefficients of (u, v).
    """
    B = len(mvals)
    J = len(mu)
    m0 = int(mvals.min())
    nk = lmax - m0 + 1
    # ---------------- direct Legendre transform (analysis)
    Gs = _rstack(np.transpose((FN + FS) * wh, (1, 2, 0)), 2)   # (B, J, 4)
    Ga = _rstack(np.transpose((FN - FS) * wh, (1, 2, 0)), 2)
    alm = np.zeros((B, nk, 4))
    for k0, P, _ in legendre_chunks(mvals, lmax, mu, logs, lpmm_cache(lmax), kchunk, False):
        K = P.shape[1]
        e0 = k0 % 2                      # local index of first even k
        o0 = 1 - e0
        alm[:, k0 + e0:k0 + K:2] = np.matmul(P[:, e0::2], Gs)
        alm[:, k0 + o0:k0 + K:2] = np.matmul(P[:, o0::2], Ga)
    zeta = alm[:, :, 0] + 1j * alm[:, :, 1]
    delta = alm[:, :, 2] + 1j * alm[:, :, 3]
    # ---------------- psi/a, chi/a
    n = mvals[:, None].astype(np.float64) + np.arange(nk)[None, :]
    nn1 = n * (n + 1)
    f = np.where((n > 0) & (n <= lmax), radius / np.where(nn1 > 0, nn1, 1), 0.0)
    psi = -f * zeta
    chi = -f * delta
    im = 1j * mvals[:, None].astype(np.float64)
    CP = _rstack(np.stack((im * chi, im * psi), axis=1), 1)      # (B, 4, nk)
    CH = _rstack(np.stack((-psi, chi), axis=1), 1)
    # ---------------- inverse Legendre transform (synthesis) of u cos, v cos
    Sym = np.zeros((B, 4, J))
    Anti = np.zeros((B, 4, J))
    for k0, P, H in legendre_chunks(mvals, lmax, mu, logs, lpmm_cache(lmax), kchunk, True):
        K = P.shape[1]
        e0 = k0 % 2
        o0 = 1 - e0
        ke = slice(k0 + e0, k0 + K, 2)
        ko = slice(k0 + o0, k0 + K, 2)
        Sym += np.matmul(CP[:, :, ke], P[:, e0::2]) + np.matmul(CH[:, :, ko], H[:, o0::2])
        Anti += np.matmul(CP[:, :, ko], P[:, o0::2]) + np.matmul(CH[:, :, ke], H[:, e0::2])
    rN = (Sym + Anti) / s
    rS = (Sym - Anti) / s
    UN = np.stack((rN[:, 0] + 1j * rN[:, 1], rN[:, 2] + 1j * rN[:, 3]))
    US = np.stack((rS[:, 0] + 1j * rS[:, 1], rS[:, 2] + 1j * rS[:, 3]))
    return UN, US


_LPMM = {}


def lpmm_cache(lmax):
    if lmax not in _LPMM:
        _LPMM[lmax] = log_pmm_table(lmax)
    return _LPMM[lmax]


# =============================================================================
# Decompositions
# =============================================================================
def block_range(n, size, rank):
    q, r = divmod(n, size)
    lo = rank * q + min(rank, r)
    return lo, lo + q + (1 if rank < r else 0)


def partition_m(mmax, lmax, size):
    """Contiguous m ranges with ~equal Legendre cost (lmax - m + 1)."""
    cost = (lmax - np.arange(mmax + 1) + 1).astype(np.float64)
    cum = np.cumsum(cost)
    tot = cum[-1]
    edges = [0]
    for r in range(1, size):
        edges.append(int(np.searchsorted(cum, tot * r / size)) + 1)
    edges.append(mmax + 1)
    edges = np.maximum.accumulate(np.minimum(edges, mmax + 1))
    return [(int(edges[r]), int(edges[r + 1])) for r in range(size)]


def match_gaussian_rows(lat_file, mu_n, tol_deg):
    """Map file rows to Gaussian node index (north/south)."""
    lat_gl = np.degrees(np.arcsin(mu_n))       # descending positive
    J = len(mu_n)
    rowN = -np.ones(J, dtype=np.int64)
    rowS = -np.ones(J, dtype=np.int64)
    asc = lat_gl[::-1]
    maxdev = 0.0
    for i, la in enumerate(lat_file):
        a = abs(la)
        p = int(np.clip(np.searchsorted(asc, a), 1, J - 1))
        jj = p if abs(asc[p] - a) < abs(asc[p - 1] - a) else p - 1
        dev = abs(asc[jj] - a)
        maxdev = max(maxdev, dev)
        j = J - 1 - jj
        tgt = rowN if la > 0 else rowS
        if tgt[j] >= 0:
            raise ValueError("latitude %g maps to an already used Gaussian node" % la)
        tgt[j] = i
    if (rowN < 0).any() or (rowS < 0).any():
        raise ValueError("file latitudes are not a complete Gaussian grid")
    if maxdev > tol_deg:
        raise ValueError("file latitudes deviate from Gaussian nodes by %.3e deg "
                         "(tolerance %.1e). Not a regular Gaussian grid?" % (maxdev, tol_deg))
    return rowN, rowS, maxdev


# =============================================================================
# netCDF I/O adapters (netCDF4 preferred, scipy.io fallback)
# =============================================================================
class Reader:
    def __init__(self, path, backend="auto"):
        self.backend = None
        if backend in ("auto", "netcdf4"):
            try:
                import netCDF4
                self.ds = netCDF4.Dataset(path, "r")
                self.ds.set_auto_maskandscale(False)
                self.backend = "netcdf4"
            except ImportError:
                if backend == "netcdf4":
                    raise
        if self.backend is None:
            from scipy.io import netcdf_file
            self.ds = netcdf_file(path, "r", mmap=True, maskandscale=False)
            self.backend = "scipy"

    def varnames(self):
        return list(self.ds.variables.keys())

    def dims(self, name):
        return tuple(self.ds.variables[name].dimensions)

    def shape(self, name):
        return tuple(self.ds.variables[name].shape)

    def attrs(self, name=None):
        if self.backend == "netcdf4":
            obj = self.ds if name is None else self.ds.variables[name]
            return {a: obj.getncattr(a) for a in obj.ncattrs()}
        obj = self.ds if name is None else self.ds.variables[name]
        return dict(obj._attributes)

    def dimlen(self, d):
        return len(self.ds.dimensions[d]) if self.backend == "netcdf4" else \
            (self.ds.dimensions[d] if self.ds.dimensions[d] is not None else
             self.ds._recs)

    def is_unlimited(self, d):
        if self.backend == "netcdf4":
            return self.ds.dimensions[d].isunlimited()
        return self.ds.dimensions[d] is None

    def read(self, name, sl):
        return np.array(self.ds.variables[name][sl])

    def dtype(self, name):
        v = self.ds.variables[name]
        if self.backend == "netcdf4":
            return np.dtype(v.dtype)
        return np.dtype(v.typecode()).newbyteorder("=")

    def close(self):
        self.ds.close()


class Writer:
    def __init__(self, path, backend, fmt="NETCDF4", deflate=0):
        self.backend = backend
        self.deflate = deflate
        if backend == "netcdf4":
            import netCDF4
            self.ds = netCDF4.Dataset(path, "w", format=fmt)
        else:
            from scipy.io import netcdf_file
            self.ds = netcdf_file(path, "w", version=2)

    def create_dim(self, name, size):
        self.ds.createDimension(name, size)

    def create_var(self, name, dtype, dims, attrs, chunks=None):
        if self.backend == "netcdf4":
            kw = {}
            if chunks is not None:
                kw["chunksizes"] = chunks
            if self.deflate:
                kw.update(zlib=True, complevel=self.deflate, shuffle=True)
            v = self.ds.createVariable(name, dtype, dims, **kw)
            v.setncatts(attrs)
        else:
            v = self.ds.createVariable(name, np.dtype(dtype).char, dims)
            for a, val in attrs.items():
                setattr(v, a, val)
        return v

    def set_global(self, attrs):
        for a, val in attrs.items():
            setattr(self.ds, a, val)

    def write(self, name, sl, data):
        import warnings
        with warnings.catch_warnings():
            # netCDF4-python sets ndarray.shape internally (deprecated in NumPy 2.5); harmless
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*shape.*")
            self.ds.variables[name][sl] = data

    def close(self):
        self.ds.close()


# =============================================================================
# Driver
# =============================================================================
def transpose_ring_to_m(comm, F, mranges, nrows_all, nlat):
    """F: (2, nrow_loc, M+1) -> (2, nlat, nm_loc) (rows in file order)."""
    size, rank = comm.size, comm.rank
    nrow = F.shape[1]
    scounts = [2 * nrow * (b - a) for a, b in mranges]
    sendbuf = np.empty(sum(scounts), np.complex128)
    off = 0
    for q, (a, b) in enumerate(mranges):
        cnt = scounts[q]
        sendbuf[off:off + cnt] = F[:, :, a:b].ravel()
        off += cnt
    ma, mb = mranges[rank]
    nm = mb - ma
    rcounts = [2 * (r1 - r0) * nm for (r0, r1) in nrows_all]
    recvbuf = np.empty(sum(rcounts), np.complex128)
    comm.alltoallv(sendbuf, scounts, recvbuf, rcounts)
    del sendbuf
    Fm = np.empty((2, nlat, nm), np.complex128)
    off = 0
    for r, (r0, r1) in enumerate(nrows_all):
        cnt = rcounts[r]
        Fm[:, r0:r1, :] = recvbuf[off:off + cnt].reshape(2, r1 - r0, nm)
        off += cnt
    return Fm


def transpose_m_to_ring(comm, Um, mranges, nrows_all, mmax):
    """Um: (2, nlat, nm_loc) -> (2, nrow_loc, M+1)."""
    rank = comm.rank
    nm = Um.shape[2]
    scounts = [2 * (r1 - r0) * nm for (r0, r1) in nrows_all]
    sendbuf = np.empty(sum(scounts), np.complex128)
    off = 0
    for r, (r0, r1) in enumerate(nrows_all):
        cnt = scounts[r]
        sendbuf[off:off + cnt] = Um[:, r0:r1, :].ravel()
        off += cnt
    r0, r1 = nrows_all[rank]
    nrow = r1 - r0
    rcounts = [2 * nrow * (b - a) for a, b in mranges]
    recvbuf = np.empty(sum(rcounts), np.complex128)
    comm.alltoallv(sendbuf, scounts, recvbuf, rcounts)
    del sendbuf
    Ur = np.empty((2, nrow, mmax + 1), np.complex128)
    off = 0
    for q, (a, b) in enumerate(mranges):
        cnt = rcounts[q]
        Ur[:, :, a:b] = recvbuf[off:off + cnt].reshape(2, nrow, b - a)
        off += cnt
    return Ur


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MPI vorticity/divergence -> u/v on a regular Gaussian grid")
    p.add_argument("infile")
    p.add_argument("outfile")
    p.add_argument("--vor", default="vo", help="vorticity variable name [vo]")
    p.add_argument("--div", default="d", help="divergence variable name [d]")
    p.add_argument("--uname", default="u")
    p.add_argument("--vname", default="v")
    p.add_argument("--truncation", default="cubic", choices=["cubic", "quadratic", "linear"],
                   help="spectral truncation implied by grid: cubic T=nlat/2-1 (TCo, default), "
                        "quadratic T=(2nlat-1)/3, linear T=nlat-1")
    p.add_argument("--lmax", type=int, default=None, help="explicit truncation (overrides --truncation)")
    p.add_argument("--radius", type=float, default=6371229.0,
                   help="Earth radius in m [6371229 = IFS; CDO uses 6371000]")
    p.add_argument("--mbatch", type=int, default=4, help="zonal wavenumbers per Legendre batch [4]")
    p.add_argument("--kchunk", type=int, default=64, help="degrees per Legendre chunk [64]")
    p.add_argument("--lat-tol", type=float, default=1e-6, help="max |file lat - Gaussian lat| in deg")
    p.add_argument("--format", default="NETCDF4", help="netCDF4 output format [NETCDF4]")
    p.add_argument("--deflate", type=int, default=0, help="zlib level for output (netCDF4 only)")
    p.add_argument("--backend", default="auto", choices=["auto", "netcdf4", "scipy"])
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def run(args, comm):
    rank, size = comm.rank, comm.size
    t_start = time.time()

    def log(msg):
        if rank == 0 and not args.quiet:
            print("[vod2uv %8.1fs] %s" % (time.time() - t_start, msg), flush=True)

    rd = Reader(args.infile, args.backend)
    dims = rd.dims(args.vor)
    if rd.dims(args.div) != dims:
        raise ValueError("vorticity and divergence have different dimensions")
    latname, lonname = dims[-2], dims[-1]
    lat = rd.read(latname, slice(None)).astype(np.float64)
    lon = rd.read(lonname, slice(None)).astype(np.float64)
    nlat, nlon = len(lat), len(lon)
    lead_shape = rd.shape(args.vor)[:-2]
    nslices = int(np.prod(lead_shape)) if lead_shape else 1

    dlon = 360.0 / nlon
    if np.max(np.abs(np.diff(lon) - dlon)) > 1e-6 * dlon:
        raise ValueError("longitudes are not uniform increasing with spacing 360/nlon")
    lam0 = math.radians(lon[0])

    if args.lmax is not None:
        lmax = args.lmax
    elif args.truncation == "cubic":
        lmax = nlat // 2 - 1
    elif args.truncation == "quadratic":
        lmax = (2 * nlat - 1) // 3
    else:
        lmax = nlat - 1
    mmax = min(lmax, (nlon - 1) // 2)
    if size > mmax + 1:
        raise ValueError("more ranks (%d) than zonal wavenumbers (%d)" % (size, mmax + 1))
    if size > nlat:
        raise ValueError("more ranks than latitude rows")
    log("grid nlat=%d nlon=%d  lmax=%d mmax=%d  slices=%d  ranks=%d  backend=%s"
        % (nlat, nlon, lmax, mmax, nslices, size, rd.backend))

    # Gaussian nodes (computed once, broadcast)
    if rank == 0:
        mu, w = gauss_legendre_north(nlat)
    else:
        mu = w = None
    mu, w = comm.bcast((mu, w))
    rowN, rowS, maxdev = match_gaussian_rows(lat, mu, args.lat_tol)
    log("Gaussian nodes OK (max |dlat| = %.2e deg)" % maxdev)
    s = np.sqrt((1.0 - mu) * (1.0 + mu))
    logs = np.log(s)
    wh = 0.5 * w

    nrows_all = [block_range(nlat, size, r) for r in range(size)]
    mranges = partition_m(mmax, lmax, size)
    r0, r1 = nrows_all[rank]
    ma, mb = mranges[rank]
    mloc = np.arange(ma, mb)
    nfreq = nlon // 2 + 1
    mm = np.arange(mmax + 1)
    phase_f = np.exp(-1j * mm * lam0) / nlon
    phase_b = np.exp(1j * mm * lam0) * nlon

    # ---------------- output file (rank 0)
    wr = None
    if rank == 0:
        wr = Writer(args.outfile, rd.backend, args.format, args.deflate)
        used = set()
        for vn in rd.varnames():
            if vn in (args.vor, args.div):
                continue
            used.update(rd.dims(vn))
        used.update(dims)
        for d in rd.ds.dimensions:
            if d in used:
                wr.create_dim(d, None if rd.is_unlimited(d) else rd.dimlen(d))
        for vn in rd.varnames():
            if vn in (args.vor, args.div):
                continue
            wr.create_var(vn, rd.dtype(vn), rd.dims(vn), rd.attrs(vn))
            wr.write(vn, slice(None), rd.read(vn, slice(None)))
        a_vo = rd.attrs(args.vor)
        common = {k: v for k, v in a_vo.items() if k.startswith("CDI_grid")}
        chunks = tuple([1] * len(lead_shape) + [min(nlat, 64), nlon])
        wr.create_var(args.uname, "f4", dims,
                      dict(standard_name="eastward_wind", long_name="U component of wind",
                           units="m s**-1", param="2.2.0", **common), chunks)
        wr.create_var(args.vname, "f4", dims,
                      dict(standard_name="northward_wind", long_name="V component of wind",
                           units="m s**-1", param="3.2.0", **common), chunks)
        ga = rd.attrs()
        hist = "vod2uv_mpi.py: lmax=%d radius=%.1f ranks=%d" % (lmax, args.radius, size)
        ga["history"] = hist + ("\n" + ga["history"] if "history" in ga else "")
        wr.set_global(ga)

    t_leg = 0.0
    t_comm = 0.0
    for islice in range(nslices):
        idx = np.unravel_index(islice, lead_shape) if lead_shape else ()
        sl = tuple(int(i) for i in idx) + (slice(r0, r1), slice(None))
        # 1. read + FFT (rows)
        F = np.empty((2, r1 - r0, mmax + 1), np.complex128)
        for c, vn in enumerate((args.vor, args.div)):
            g = rd.read(vn, sl).astype(np.float64)
            F[c] = np.fft.rfft(g, axis=-1)[:, :mmax + 1] * phase_f
            del g
        # 2. transpose
        tc = time.time()
        Fm = transpose_ring_to_m(comm, F, mranges, nrows_all, nlat)
        del F
        t_comm += time.time() - tc
        # 3-5. Legendre in m-space
        tl = time.time()
        Um = np.empty_like(Fm)
        for b0 in range(0, len(mloc), args.mbatch):
            b1 = min(b0 + args.mbatch, len(mloc))
            mv = mloc[b0:b1]
            FN = Fm[:, rowN, b0:b1].transpose(0, 2, 1)
            FS = Fm[:, rowS, b0:b1].transpose(0, 2, 1)
            UN, US = vod2uv_mbatch(mv, lmax, mu, logs, s, wh, FN, FS, args.radius, args.kchunk)
            Um[:, rowN, b0:b1] = UN.transpose(0, 2, 1)
            Um[:, rowS, b0:b1] = US.transpose(0, 2, 1)
        del Fm
        t_leg += time.time() - tl
        # 6. transpose back + inverse FFT
        tc = time.time()
        Ur = transpose_m_to_ring(comm, Um, mranges, nrows_all, mmax)
        del Um
        t_comm += time.time() - tc
        out = np.empty((2, r1 - r0, nlon), np.float32)
        G = np.zeros((r1 - r0, nfreq), np.complex128)
        for c in range(2):
            G[:, :mmax + 1] = Ur[c] * phase_b
            out[c] = np.fft.irfft(G, n=nlon, axis=-1)
        del Ur, G
        # 7. gather to writer
        if rank == 0:
            for c, vn in enumerate((args.uname, args.vname)):
                wr.write(vn, sl, out[c])
            for r in range(1, size):
                q0, q1 = nrows_all[r]
                buf = np.empty((2, q1 - q0, nlon), np.float32)
                comm.Recv(buf, r, 100 + islice % 1000)
                slr = sl[:-2] + (slice(q0, q1), slice(None))
                for c, vn in enumerate((args.uname, args.vname)):
                    wr.write(vn, slr, buf[c])
        else:
            comm.Send(out, 0, 100 + islice % 1000)
        del out
        log("slice %d/%d done" % (islice + 1, nslices))

    t_leg = comm.allreduce_max(t_leg)
    t_comm = comm.allreduce_max(t_comm)
    if rank == 0:
        wr.close()
    rd.close()
    log("finished: Legendre(max rank) %.1fs, transposes(max rank) %.1fs" % (t_leg, t_comm))


def main(argv=None):
    args = parse_args(argv)
    comm = get_comm()
    try:
        run(args, comm)
    except Exception as e:
        sys.stderr.write("[rank %d] ERROR: %s\n" % (comm.rank, e))
        sys.stderr.flush()
        if isinstance(comm, MPIComm):
            comm.c.Abort(1)
        raise


if __name__ == "__main__":
    main()
