import sys, numpy as np
_D = __import__('os').path.dirname(__import__('os').path.abspath(__file__)); sys.path.insert(0, _D); sys.path.insert(0, __import__('os').path.dirname(_D))
import vod2uv_mpi as V, fakempi, make_testfile as M
from scipy.io import netcdf_file
d = '/tmp/v2u'; import os; os.makedirs(d, exist_ok=True)
def check(N, R, lon0, nranks, extra=()):
    fin = '%s/in_N%d.nc' % (d, N); fout = '%s/out_N%d_np%d.nc' % (d, N, nranks)
    M.write(fin, N, R, lon0)
    args = V.parse_args([fin, fout, '--backend', 'scipy', '--quiet', *extra])
    if nranks == 1: V.run(args, V.SerialComm())
    else: fakempi.run_parallel(lambda c: V.run(args, c), nranks)
    ex = np.load(fin + '.exact.npz')
    with netcdf_file(fout, 'r', mmap=False) as f:
        u = f.variables['u'][:].astype(float); v = f.variables['v'][:].astype(float)
    sc = max(abs(ex['u']).max(), abs(ex['v']).max())
    eu = abs(u-ex['u']).max()/sc; ev = abs(v-ex['v']).max()/sc
    print("N=%4d R=%3d lon0=%5.1f ranks=%d : max|u-u_exact|/max|V| = %.2e, v: %.2e  (max|V|=%.1f m/s)" % (N, R, lon0, nranks, eu, ev, sc))
    return u, v
if __name__ == '__main__' and len(sys.argv) == 1:
    u1, v1 = check(48, 40, 0.0, 1)
    u3, v3 = check(48, 40, 0.0, 3)
    u7, v7 = check(48, 40, 0.0, 7, ('--mbatch', '5', '--kchunk', '7'))
    print("rank-count invariance: max|u(np1)-u(np3)| = %.2e, np7 = %.2e" % (abs(u1-u3).max(), abs(u1-u7).max()))
    check(48, 40, -11.25, 4)     # shifted longitude origin
    check(160, 150, 0.0, 4)      # higher m
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'big':
    import time; t = time.time()
    check(1280, 1000, 0.0, 2)
    print("N1280 end-to-end (4 slices, 2 simulated ranks): %.0fs" % (time.time()-t))
