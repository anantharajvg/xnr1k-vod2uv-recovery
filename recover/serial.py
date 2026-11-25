"""Single-process reference pipeline (same kernels as the MPI driver).
Used by the tests and as the reference that the MPI driver must reproduce bitwise (M7)."""
import numpy as np

import core as C


def recover_serial(vo, d, gin, gout, lmax, radius=6371229.0, want_vod=False, mbatch=4, kchunk=64):
    nm_in = gin.nmen(lmax)
    nm_out = gout.nmen(lmax)
    mmax = int(min(lmax, nm_in.max()))
    F = np.stack([C.rows_to_fourier(x, gin, 0, gin.nlat, nm_in, mmax) for x in (vo, d)])
    K = 4 if want_vod else 2
    G = np.zeros((K, gout.nlat, mmax + 1), np.complex128)
    for b0 in range(0, mmax + 1, mbatch):
        mv = np.arange(b0, min(b0 + mbatch, mmax + 1))
        FN = F[:, gin.rowN][:, :, mv].transpose(0, 2, 1)
        FS = F[:, gin.rowS][:, :, mv].transpose(0, 2, 1)
        oN, oS, _ = C.process_mbatch(mv, lmax, gin, gout, FN, FS, radius, want_vod, kchunk)
        G[:, gout.rowN[:, None], mv[None, :]] = oN.transpose(0, 2, 1)
        G[:, gout.rowS[:, None], mv[None, :]] = oS.transpose(0, 2, 1)
    return [C.fourier_to_rows(G[k], gout, 0, gout.nlat, nm_out) for k in range(K)]


def point_coords(grid):
    """Latitude (rad) and longitude (rad) of every point, rows north -> south."""
    lat = np.radians(np.repeat(grid.lat_deg, grid.nlon))
    lon = np.concatenate([np.radians(grid.lon0) + 2 * np.pi * np.arange(n) / n for n in grid.nlon])
    return lat, lon
