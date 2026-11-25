"""Write a CDO-like regular Gaussian netCDF-3 file with analytic vo/d and the exact u/v."""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import vod2uv_mpi as V
from scipy.io import netcdf_file
A = 6371229.0
def fields(N, R, lon0=0.0, nlev=2, ntime=2, rng=None):
    nlat = 2*N; nlon = 4*N + 16
    mu, w = V.gauss_legendre_north(nlat)
    lat = np.degrees(np.arcsin(np.concatenate((mu, -mu[::-1]))))   # N -> S, like CDO
    lon = lon0 + np.arange(nlon)*360.0/nlon
    phi = np.radians(lat)[:, None]; lam = np.radians(lon)[None, :]
    c, s = np.cos(phi), np.sin(phi)
    out = []
    for it in range(ntime):
        for k in range(nlev):
            om, K, D = 7.848e-6*(1+0.1*k), 7.848e-6*(1+0.3*it), 3.0e-6*(1-0.2*k)
            ph = 0.3*it + 0.7*k                       # phase shift in lambda
            # rotational: Rossby-Haurwitz wave (Williamson et al. 1992, test 6)
            u = A*om*c + A*K*c**(R-1)*(R*s**2 - c**2)*np.cos(R*(lam-ph))
            v = -A*K*R*c**(R-1)*s*np.sin(R*(lam-ph))
            vo = 2*om*s - K*s*c**R*(R+1)*(R+2)*np.cos(R*(lam-ph))
            # divergent: chi = A^2 D cos^R sin sin(R lam) ; plus chi2 = A^2 D sin(phi) (Y_1^0)
            u += A*D*R*c**(R-1)*s*np.cos(R*(lam-ph))
            v += A*D*(c**(R+1) - R*c**(R-1)*s**2)*np.sin(R*(lam-ph)) + A*D*c
            d = -D*(R+1)*(R+2)*c**R*s*np.sin(R*(lam-ph)) - 2*D*s
            out.append((vo, d, u, v))
    return lat, lon, out
def write(path, N, R, lon0=0.0, nlev=2, ntime=2):
    lat, lon, flds = fields(N, R, lon0, nlev, ntime)
    nlat, nlon = len(lat), len(lon)
    f = netcdf_file(path, 'w', version=2)
    f.createDimension('time', None); f.createDimension('lon', nlon); f.createDimension('lat', nlat); f.createDimension('lev', nlev)
    t = f.createVariable('time', 'd', ('time',)); t.units = 'hours since 2018-11-1 00:00:00'; t.standard_name='time'
    x = f.createVariable('lon', 'd', ('lon',)); x[:] = lon; x.units = 'degrees_east'
    y = f.createVariable('lat', 'd', ('lat',)); y[:] = lat; y.units = 'degrees_north'
    z = f.createVariable('lev', 'd', ('lev',)); z[:] = np.arange(1, nlev+1)
    vo = f.createVariable('vo', 'f', ('time', 'lev', 'lat', 'lon')); vo.units = 's**-1'; vo.CDI_grid_type='gaussian'
    dv = f.createVariable('d', 'f', ('time', 'lev', 'lat', 'lon')); dv.units = 's**-1'
    U = np.zeros((ntime, nlev, nlat, nlon)); Vv = np.zeros_like(U)
    i = 0
    for it in range(ntime):
        t[it] = 6.0*it
        for k in range(nlev):
            a, b, c, e = flds[i]; i += 1
            vo[it, k] = a.astype(np.float32); dv[it, k] = b.astype(np.float32)
            U[it, k] = c; Vv[it, k] = e
    f.close()
    np.savez(path + '.exact.npz', u=U, v=Vv)
if __name__ == '__main__':
    write(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
