"""M7 + driver tests (no GRIB library needed): the MPI driver, run with thread-based
ranks in several world/group layouts, must reproduce the serial reference bitwise,
and the serial reference must match the analytic answer (M1 level of accuracy)."""
import os, shutil, sys, tempfile
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(D)); sys.path.insert(0, D)
import core as C, serial as S, comm as CM, recover_uv as R, gribio as G
from test_m1_analytic import fields

A = 6371229.0


class MemoryTemplateSink:
    """Stands in for GribTemplateSink: records what would be encoded."""
    def __init__(self, meta):
        self.meta, self.store = meta, {}
    def done(self, var, level): return (var, level) in self.store
    def template_meta(self, var, level):
        m = dict(self.meta); m.update(level=level, shortName=var); return m
    def write(self, var, level, values): self.store[(var, level)] = values.copy()
    def merge(self, var, levels, out_dir): return "(memory)"


def make_source(grid, waves_per_level):
    la, lo = S.point_coords(grid)
    flds = {"vo": {}, "d": {}}; exact = {}
    for L, waves in waves_per_level.items():
        f = [fields(la, lo, Rw, base=(i == 0), ph=0.1 * L) for i, Rw in enumerate(waves)]
        flds["vo"][L] = sum(x[0] for x in f); flds["d"][L] = sum(x[1] for x in f)
        exact[L] = (sum(x[2] for x in f), sum(x[3] for x in f))
    meta = dict(N=grid.N, pl=grid.nlon, gridType="reduced_gg", jScansPositively=0, iScansNegatively=0,
                bitmapPresent=0, longitudeOfFirstGridPointInDegrees=0.0, numberOfValues=grid.npts)
    return G.MemorySource(flds, meta), meta, exact


def run_driver(world_size, group, src, meta, argv):
    sinks = []
    def fn(comm):
        sink = MemoryTemplateSink(meta)
        sinks.append((comm.rank, sink))
        R.run(R.parse_args(argv + (["--group-size", str(group)] if group else [])), comm, src, sink)
    if world_size == 1:
        fn(CM.SerialComm())
    else:
        CM.run_threads(fn, world_size)
    merged = {}
    for _, s in sinks:
        merged.update(s.store)
    return merged


if __name__ == "__main__":
    ok = True
    grid = C.GaussGrid.octahedral(80)
    waves = {1: (12,), 2: (40,), 3: (70,), 4: (5, 60), 5: (25,)}
    src, meta, exact = make_source(grid, waves)
    out = tempfile.mkdtemp()
    argv = ["--out-dir", out, "--levels", "1-5", "--quiet"]
    # serial reference and its accuracy
    ref = {}
    for L in waves:
        u, v = S.recover_serial(src.fields["vo"][L], src.fields["d"][L], grid, grid, grid.N - 1, A)
        ref[("u", L)], ref[("v", L)] = u, v
        ue, ve = exact[L]; sc = max(abs(ue).max(), abs(ve).max())
        e = max(abs(u - ue).max(), abs(v - ve).max()) / sc
        print("level %d waves %-8s serial vs analytic: %.1e" % (L, waves[L], e)); ok &= e <= 3e-7
    for ws, gsz in ((1, None), (3, None), (4, 2), (6, 3), (6, 1), (5, 5)):
        got = run_driver(ws, gsz, src, meta, argv)
        missing = set(ref) - set(got)
        dmax = max(abs(got[k] - ref[k]).max() for k in ref if k in got)
        same = not missing and all(np.array_equal(got[k], ref[k]) for k in ref)
        print("world %d, group %s: levels written %d/%d, max|diff vs serial| = %.1e, bitwise identical: %s"
              % (ws, gsz or ws, len(got) // 2, len(waves), dmax, same))
        ok &= not missing and same          # plan criterion M7: bitwise identical
    # the batch partition must cover 0..mmax exactly, align to batches, and give every rank work
    for (mm, ll, sz, B) in ((79, 79, 7, 4), (7999, 7999, 32, 4), (7999, 7999, 2000, 4), (639, 639, 3, 5)):
        pr = R.partition_m_batches(mm, ll, sz, B)
        cover = [m for a, b in pr for m in range(a, b)] == list(range(mm + 1))
        aligned = all(a % B == 0 for a, _ in pr) and all(b > a for a, b in pr)
        print("partition mmax=%d ranks=%d batch=%d: covers=%s aligned=%s max/min cost=%.2f"
              % (mm, sz, B, cover, aligned, (lambda c: max(c) / min(c))([sum(ll - m + 1 for m in range(a, b)) for a, b in pr])))
        ok &= cover and aligned
    try:
        R.partition_m_batches(7, 7, 3, 4); print("FAIL: oversubscribed partition accepted"); ok = False
    except ValueError:
        print("oversubscribed partition refused (expected)")
    # restart: parts exist -> skipped
    shutil.rmtree(out)
    print("M7:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
