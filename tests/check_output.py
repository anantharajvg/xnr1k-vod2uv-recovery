"""Compare vod2uv output against the analytic u/v saved by make_testfile.py."""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
import vod2uv_mpi as V
fin, fout = sys.argv[1], sys.argv[2]
ex = np.load(fin + '.exact.npz')
r = V.Reader(fout)
u = r.read('u', slice(None)).astype(float); v = r.read('v', slice(None)).astype(float)
sc = max(abs(ex['u']).max(), abs(ex['v']).max())
eu, ev = abs(u-ex['u']).max()/sc, abs(v-ex['v']).max()/sc
print("backend=%s  max rel error u=%.2e v=%.2e  -> %s" % (r.backend, eu, ev, "PASS" if max(eu, ev) < 3e-7 else "FAIL"))
sys.exit(0 if max(eu, ev) < 3e-7 else 1)
