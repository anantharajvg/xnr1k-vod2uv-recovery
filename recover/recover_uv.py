#!/usr/bin/env python3
"""
recover_uv.py -- rebuild model-level u, v from vo, d on a Gaussian grid
(native O8000 reduced octahedral for XNR1K), MPI-parallel with level groups.

  world ranks are split into groups of --group-size ranks; each group processes
  whole levels (levels are dealt round-robin to groups). Inside a group:
    root decodes the vo and d GRIB messages -> scatters latitude-row blocks
    -> per-row FFT (row length nlon_j, wavenumbers m <= NMEN_j, ecTrans rule)
    -> all-to-all transpose to zonal wavenumber blocks
    -> Gauss-Legendre analysis to T(lmax), psi/chi, synthesis of u, v on the output grid
    -> transpose back -> per-row inverse FFT -> gather to root -> write.

Output
  --out-grid native : GRIB2, each message cloned from the matching message of the
                      corrupted u/v files (--template-u/--template-v); only values change.
  --out-grid F<N>|O<N> : per-level .npz parts merged into one netCDF (regular grid) or
                      .npz (reduced grid), optionally with truncated vo, d (--write-vod).
  --spec-out        : zeta, delta spectral coefficients per level (m-major, n = m..lmax).

Restartable: a level whose part files exist is skipped unless --overwrite.

Example (Andes, one 32-rank group per node, 4 nodes, level 74 only):
  srun -N 4 -n 128 -c 1 python3 recover_uv.py \
     --vo $X/vo/vo_138_h3f7-ndjf_000000.ml-dgg.O8000.global.grib \
     --d  $X/d/d_155_h3f7-ndjf_000000.ml-dgg.O8000.global.grib \
     --template-u $X/u/u_131_h3f7-ndjf_000000.ml-dgg.O8000.global.grib \
     --template-v $X/v/v_132_h3f7-ndjf_000000.ml-dgg.O8000.global.grib \
     --levels 74 --group-size 32 --out-dir $W/run_L074
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import core as C          # noqa: E402
import comm as CM         # noqa: E402
from vod2uv_mpi import block_range   # noqa: E402


def mem_report():
    """Root's resident memory, node MemAvailable and /dev/shm use, in GB (Linux; '?' where unavailable)."""
    def gb(x):
        return "%.1f" % (x / 2 ** 30) if x is not None else "?"
    rss = avail = shm = None
    try:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
        st = os.statvfs("/dev/shm")
        shm = (st.f_blocks - st.f_bfree) * st.f_frsize
    except Exception:   # noqa: BLE001
        pass
    return "mem: root rss %s GB, node available %s GB, /dev/shm used %s GB" % (gb(rss), gb(avail), gb(shm))


def partition_m_batches(mmax, lmax, size, mbatch):
    """Contiguous m ranges, one per rank, made of WHOLE global batches
    [0, mbatch), [mbatch, 2*mbatch), ... with about equal Legendre cost.

    Every zonal wavenumber is then processed in the same batch, with the same array
    shapes, whatever the number of ranks, so results are bitwise reproducible across
    layouts (BLAS kernels can round differently for different shapes)."""
    nb = (mmax + 1 + mbatch - 1) // mbatch
    if size > nb:
        raise ValueError("group size %d exceeds the number of m batches %d" % (size, nb))
    m = np.arange(mmax + 1)
    cost = np.bincount(m // mbatch, weights=(lmax - m + 1).astype(np.float64), minlength=nb)
    cum = np.cumsum(cost)
    edges = [0]
    for r in range(1, size):
        e = int(np.searchsorted(cum, cum[-1] * r / size)) + 1
        edges.append(min(max(e, edges[-1] + 1), nb - (size - r)))
    edges.append(nb)
    return [(edges[r] * mbatch, min(edges[r + 1] * mbatch, mmax + 1)) for r in range(size)]

PACK_STEPS = 65535.0      # 16-bit simple packing


def parse_levels(s, nlev):
    if s in ("all", None):
        return list(range(1, nlev + 1))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    if out[0] < 1 or out[-1] > nlev:
        raise ValueError("levels outside 1..%d" % nlev)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vo"); p.add_argument("--d")
    p.add_argument("--template-u"); p.add_argument("--template-v")
    p.add_argument("--levels", default="all", help="'all', '74', '1-10,74' [all]")
    p.add_argument("--lmax", type=int, default=None,
                   help="truncation; default N-1 for octahedral (cubic) input, 2N-1 otherwise")
    p.add_argument("--out-grid", default="native", help="native | F<N> | O<N> [native]")
    p.add_argument("--write-vod", action="store_true", help="also write truncated vo, d (non-native output)")
    p.add_argument("--spec-out", action="store_true", help="write zeta, delta coefficients per level")
    p.add_argument("--radius", type=float, default=6371229.0)
    p.add_argument("--group-size", type=int, default=None, help="ranks per level group [all ranks]")
    p.add_argument("--mbatch", type=int, default=4)
    p.add_argument("--kchunk", type=int, default=64)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-merge", action="store_true", help="leave per-level parts, skip the final merge")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- transposes
def to_mspace(grp, F, mranges, rows_all, nlat):
    """F: (C, nrow_loc, M+1) -> (C, nlat, nm_loc)."""
    Cc, nrow = F.shape[0], F.shape[1]
    sc = [Cc * nrow * (b - a) for a, b in mranges]
    sb = np.concatenate([F[:, :, a:b].ravel() for a, b in mranges]) if sum(sc) else np.zeros(0, np.complex128)
    ma, mb = mranges[grp.rank]
    nm = mb - ma
    rc = [Cc * (r1 - r0) * nm for r0, r1 in rows_all]
    rb = np.empty(sum(rc), np.complex128)
    grp.alltoallv(sb, sc, rb, rc)
    Fm = np.empty((Cc, nlat, nm), np.complex128)
    off = 0
    for (r0, r1), n in zip(rows_all, rc):
        Fm[:, r0:r1, :] = rb[off:off + n].reshape(Cc, r1 - r0, nm)
        off += n
    return Fm


def to_rows(grp, Um, mranges, rows_all, mmax):
    """Um: (K, nlat_out, nm_loc) -> (K, nrow_loc_out, M+1)."""
    K, nm = Um.shape[0], Um.shape[2]
    sc = [K * (r1 - r0) * nm for r0, r1 in rows_all]
    sb = np.concatenate([Um[:, r0:r1, :].ravel() for r0, r1 in rows_all])
    q0, q1 = rows_all[grp.rank]
    rc = [K * (q1 - q0) * (b - a) for a, b in mranges]
    rb = np.empty(sum(rc), np.complex128)
    grp.alltoallv(sb, sc, rb, rc)
    Ur = np.zeros((K, q1 - q0, mmax + 1), np.complex128)
    off = 0
    for (a, b), n in zip(mranges, rc):
        Ur[:, :, a:b] = rb[off:off + n].reshape(K, q1 - q0, b - a)
        off += n
    return Ur


# ----------------------------------------------------------------------------- checks
def check_input_meta(mv, md, grid_ref=None):
    errs = []
    for m, nm in ((mv, "vo"), (md, "d")):
        if m.get("gridType") not in ("reduced_gg", "regular_gg", None):
            errs.append("%s: gridType %s" % (nm, m.get("gridType")))
        if m.get("jScansPositively", 0) != 0:
            errs.append("%s: rows not north to south" % nm)
        if m.get("iScansNegatively", 0) != 0:
            errs.append("%s: longitudes decrease along rows" % nm)
        if m.get("bitmapPresent", 0) not in (0, None):
            errs.append("%s: bitmap present (missing values)" % nm)
        if abs(float(m.get("longitudeOfFirstGridPointInDegrees", 0.0) or 0.0)) > 1e-9:
            errs.append("%s: first longitude is not 0" % nm)
    for k in ("N", "level", "dataDate", "dataTime", "step", "numberOfValues"):
        if mv.get(k) != md.get(k):
            errs.append("vo and d differ in %s: %s vs %s" % (k, mv.get(k), md.get(k)))
    if "pl" in mv and "pl" in md and not np.array_equal(mv["pl"], md["pl"]):
        errs.append("vo and d have different pl arrays")
    if grid_ref is not None and "pl" in mv and not np.array_equal(mv["pl"], grid_ref.nlon):
        errs.append("level %s grid differs from the first level's" % mv.get("level"))
    return errs


def check_template(mt, mv, var):
    errs = []
    for k in ("N", "level", "dataDate", "dataTime", "step", "numberOfValues"):
        if mt.get(k) != mv.get(k):
            errs.append("template %s differs from input in %s: %s vs %s" % (var, k, mt.get(k), mv.get(k)))
    if mt.get("shortName") != var:
        errs.append("template shortName %s, expected %s" % (mt.get("shortName"), var))
    if "pl" in mt and "pl" in mv and not np.array_equal(mt["pl"], mv["pl"]):
        errs.append("template %s grid differs from input" % var)
    return errs


# ----------------------------------------------------------------------------- driver
def run(args, world, source=None, sink=None):
    t0 = time.time()
    rank = world.rank
    gs = args.group_size or world.size
    if world.size % gs:
        raise ValueError("world size %d is not a multiple of --group-size %d" % (world.size, gs))
    ngroups = world.size // gs
    color = rank // gs
    grp = world.split(color)
    groot = grp.rank == 0

    def log(msg, allgroups=False):
        if not args.quiet and (rank == 0 or (allgroups and groot)):
            print("[recover %7.1fs g%02d] %s" % (time.time() - t0, color, msg), flush=True)

    if source is None:
        from gribio import GribSource
        source = GribSource(args.vo, args.d)
    native = args.out_grid == "native"
    if sink is None:
        if native:
            from gribio import GribTemplateSink
            sink = GribTemplateSink(args.template_u, args.template_v, args.out_dir)
        else:
            from gribio import NpzSink
            sink = NpzSink(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- grid from the first level's header (world root), broadcast
    info = None
    if rank == 0:
        nlev = source.nlevels()
        levels = parse_levels(args.levels, nlev)
        mv, _ = source.read("vo", levels[0])
        info = dict(nlev=nlev, levels=levels, N=int(mv["N"]), pl=mv.get("pl"), meta=mv)
    info = world.bcast(info)
    N, levels = info["N"], info["levels"]
    gin = C.GaussGrid.from_pl(N, info["pl"]) if info["pl"] is not None else C.GaussGrid.regular(N)
    octa = C.GaussGrid.octahedral(N)
    is_octa = np.array_equal(gin.nlon, octa.nlon)
    lmax = args.lmax if args.lmax is not None else (N - 1 if is_octa else 2 * N - 1)
    gout = gin if native else C.GaussGrid.from_spec(args.out_grid)
    nm_in, nm_out = gin.nmen(lmax), gout.nmen(lmax)
    mmax = int(min(lmax, nm_in.max()))
    want_vod = bool(args.write_vod) and not native
    K = 4 if want_vod else 2
    log("input %s (%s, %d points), output %s, lmax=%d, mmax=%d, levels %s, %d group(s) of %d"
        % (gin.name, "octahedral" if is_octa else "reduced/regular", gin.npts, gout.name, lmax, mmax,
           "%d..%d (%d)" % (levels[0], levels[-1], len(levels)), ngroups, gs))

    rows_in = [block_range(gin.nlat, gs, r) for r in range(gs)]
    rows_out = [block_range(gout.nlat, gs, r) for r in range(gs)]
    mranges = partition_m_batches(mmax, lmax, gs, args.mbatch)
    if any(b <= a for a, b in mranges):
        raise ValueError("group size %d too large for mmax %d" % (gs, mmax))
    r0, r1 = rows_in[grp.rank]
    q0, q1 = rows_out[grp.rank]
    ma, mb = mranges[grp.rank]
    mloc = np.arange(ma, mb)
    cnt_in = [int(gin.offsets[b] - gin.offsets[a]) for a, b in rows_in]
    cnt_out = [int(gout.offsets[b] - gout.offsets[a]) for a, b in rows_out]
    wts = gout.area_weights() if groot else None

    my_levels = levels[color::ngroups]
    stats_rows = []
    for L in my_levels:
        skip = grp.bcast(bool(sink.done("u", L) and sink.done("v", L)) and not args.overwrite if groot else None)
        if skip:
            log("level %d: parts exist, skipped" % L, True)
            continue
        tl = time.time()
        # ---- read + scatter
        F = np.empty((2, r1 - r0, mmax + 1), np.complex128)
        metas = {}
        for c, var in enumerate(("vo", "d")):
            vals = None
            if groot:
                metas[var], vals = source.read(var, L)
                if vals.size != gin.npts:
                    raise ValueError("level %d %s: %d values, grid has %d" % (L, var, vals.size, gin.npts))
            loc = np.empty(cnt_in[grp.rank])
            grp.scatterv(vals, cnt_in, loc)
            del vals
            F[c] = C.rows_to_fourier(loc, gin, r0, r1, nm_in, mmax)
            del loc
        if groot:
            errs = check_input_meta(metas["vo"], metas["d"], gin if info["pl"] is not None else None)
            if native:
                for var in ("u", "v"):
                    errs += check_template(sink.template_meta(var, L), metas["vo"], var)
        else:
            errs = None
        errs = grp.bcast(errs)
        if errs:
            raise ValueError("level %d: %s" % (L, "; ".join(errs)))
        t_read = time.time() - tl
        # ---- transform
        Fm = to_mspace(grp, F, mranges, rows_in, gin.nlat)
        del F
        Um = np.zeros((K, gout.nlat, len(mloc)), np.complex128)
        spec = [] if args.spec_out else None
        for b0 in range(0, len(mloc), args.mbatch):
            b1 = min(b0 + args.mbatch, len(mloc))
            mv_ = mloc[b0:b1]
            FN = Fm[:, gin.rowN, b0:b1].transpose(0, 2, 1)
            FS = Fm[:, gin.rowS, b0:b1].transpose(0, 2, 1)
            oN, oS, alm = C.process_mbatch(mv_, lmax, gin, gout, FN, FS, args.radius, want_vod, args.kchunk)
            Um[:, gout.rowN, b0:b1] = oN.transpose(0, 2, 1)
            Um[:, gout.rowS, b0:b1] = oS.transpose(0, 2, 1)
            if spec is not None:
                for i, m in enumerate(mv_):
                    spec.append(alm[:, i, :lmax - m + 1])
        del Fm
        Ur = to_rows(grp, Um, mranges, rows_out, mmax)
        del Um
        # ---- inverse FFT + gather + write
        names = ("u", "v", "vo", "d")[:K]
        out = {}
        for k in range(K):
            loc = C.fourier_to_rows(Ur[k], gout, q0, q1, nm_out)
            full = np.empty(gout.npts) if groot else None
            grp.gatherv(loc, cnt_out, full)
            out[names[k]] = full
        del Ur
        if spec is not None:
            parts = grp.gather(np.concatenate(spec, axis=1) if spec else np.zeros((2, 0), np.complex128))
        if groot:
            if native:
                for var in ("u", "v"):
                    sink.write(var, L, out[var])
            else:
                sink.write_level(L, {k: v.astype(np.float32) for k, v in out.items()})
            if spec is not None:
                coef = np.concatenate(parts, axis=1)
                np.savez(os.path.join(args.out_dir, "parts", "spec_L%03d.npz" % L),
                         zeta=coef[0], delta=coef[1], lmax=lmax, mmax=mmax)
            for var, x in out.items():
                mean = float(np.dot(wts, x))
                rms = float(np.sqrt(np.dot(wts, x * x)))
                stats_rows.append(dict(level=L, var=var, min=float(x.min()), max=float(x.max()), mean=mean,
                                       rms=rms, pack_step=float((x.max() - x.min()) / PACK_STEPS)))
            log("level %d done in %.0fs (read+scatter %.0fs)  u[%.2f, %.2f]  v[%.2f, %.2f]  %s"
                % (L, time.time() - tl, t_read, out["u"].min(), out["u"].max(), out["v"].min(), out["v"].max(),
                   mem_report()), True)
        del out

    # ---- per-level statistics and final merge
    allstats = world.gather(stats_rows)
    world.barrier()
    if rank == 0:
        rows = sorted([r for s in allstats for r in s], key=lambda r: (r["level"], r["var"]))
        if rows:
            path = os.path.join(args.out_dir, "level_stats.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if new:
                    w.writeheader()
                w.writerows(rows)
        if not args.no_merge:
            if native:
                if all(sink.done("u", L) and sink.done("v", L) for L in levels):
                    for var in ("u", "v"):
                        log("merged %s" % sink.merge(var, levels, args.out_dir))
                else:
                    log("some levels missing; merge skipped (rerun to complete)")
            else:
                merge_npz(args, sink, levels, gout, lmax, K)
    log("finished")


def merge_npz(args, sink, levels, gout, lmax, K):
    names = ("u", "v", "vo", "d")[:K]
    if not all(os.path.exists(sink.part_path(L)) for L in levels):
        print("some levels missing; merge skipped")
        return
    if not gout.reduced:
        import netCDF4
        path = os.path.join(args.out_dir, "recovered_%s_T%d.nc" % (gout.name, lmax))
        ds = netCDF4.Dataset(path + ".tmp", "w", format="NETCDF4")
        ds.createDimension("lev", len(levels)); ds.createDimension("lat", gout.nlat)
        ds.createDimension("lon", int(gout.nlon[0]))
        ds.createVariable("lev", "i4", ("lev",))[:] = levels
        la = ds.createVariable("lat", "f8", ("lat",)); la[:] = gout.lat_deg; la.units = "degrees_north"
        lo = ds.createVariable("lon", "f8", ("lon",)); lo[:] = gout.lon0 + 360.0 * np.arange(gout.nlon[0]) / gout.nlon[0]
        lo.units = "degrees_east"
        units = {"u": "m s-1", "v": "m s-1", "vo": "s-1", "d": "s-1"}
        vs = {n: ds.createVariable(n, "f4", ("lev", "lat", "lon"), zlib=True, complevel=1) for n in names}
        for n in names:
            vs[n].units = units[n]
        for i, L in enumerate(levels):
            z = np.load(sink.part_path(L))
            for n in names:
                vs[n][i] = z[n].reshape(gout.nlat, -1)
        ds.history = "recover_uv.py lmax=%d radius=%.1f" % (lmax, args.radius)
        ds.close()
        os.replace(path + ".tmp", path)
    else:
        path = os.path.join(args.out_dir, "recovered_%s_T%d.npz" % (gout.name, lmax))
        arrs = {n: np.stack([np.load(sink.part_path(L))[n] for L in levels]) for n in names}
        np.savez(path, levels=np.array(levels), nlon=gout.nlon, **arrs)
    print("merged", path)


def main(argv=None):
    args = parse_args(argv)
    world = CM.get_world()
    try:
        run(args, world)
    except Exception as e:   # noqa: BLE001
        sys.stderr.write("[rank %d] ERROR: %s\n" % (world.rank, e))
        sys.stderr.flush()
        if world.size > 1:
            world.abort(1)
        raise


if __name__ == "__main__":
    main()
