#!/usr/bin/env python3
"""
plots.py -- figures for the XNR1K u/v recovery and its validation (PNG, matplotlib).

Subcommands
  map       one field, a difference of two fields, or wind speed; global or a region.
            Inputs: netCDF PATH:VAR (regular grid), GRIB (any Gaussian grid, one level),
            or a merged reduced-grid .npz from recover_uv.py.
  zonal     zonal-mean u and v of A and B at one level      (from compare.py *_zonal.npz)
  profiles  bias, RMSD and correlation against level/pressure (from compare.py *_stats.csv),
            with levels that jump from their neighbours flagged
  spectra   rotational and divergent kinetic-energy spectra  (from --spec-out files, or ARCO)
  rowlimits per-row wavenumber limit used for the O8000 grid (ecTrans rule vs (nlon-1)/2)

Maps on reduced grids are resampled to a display raster (rows: linear in latitude;
along rows: linear in longitude). That resampling is for display only and is
stated on every map; no statistic is computed from it.

Examples
  python3 plots.py map --grib $W/u_131_...recovered.grib --level 74 --var u --out u_L074.png
  python3 plots.py map --grib REC.grib --minus-grib CORRUPTED.grib --level 74 --var u --out du.png
  python3 plots.py map --nc recovered_F320_T639.nc:u --level 74 \
          --minus-nc $E/e5...u.regn320uv.2018110100_2018110105.nc:U --minus-time 0 --out du_era5.png
  python3 plots.py map --grib REC_u.grib --grib-v REC_v.grib --speed --level 74 \
          --region -80 -40 20 50 --out speed_zoom.png
  python3 plots.py zonal --npz cmp_zonal.npz --level 74 --out zonal_L074.png
  python3 plots.py profiles --csv cmp_stats.csv --out profiles.png
  python3 plots.py spectra --spec parts/spec_L074.npz:XNR1K --arco era5_arco.npz:74:ERA5 --out ke.png
  python3 plots.py rowlimits --N 8000 --out rowlimits.png
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402

# ---------------------------------------------------------------- style (reference palette, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]            # categorical slots 1-3 (all-pairs safe)
BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
RED = ["#f6d3d2", "#efa3a2", "#e8706f", "#e34948", "#b8302f", "#8a1f1e", "#5c1110"]
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", BLUE + [NEUTRAL] + RED)
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", BLUE[::-1])
A_EARTH = 6371229.0

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 1.5,
    "legend.frameon": False, "figure.dpi": 150,
})


# ---------------------------------------------------------------- field access
class Rows:
    """A field as Gaussian rows: lat (deg, per row), nlon per row, flat values, lon0 = 0."""

    def __init__(self, lat, nlon, values, label):
        self.lat, self.nlon, self.values, self.label = np.asarray(lat), np.asarray(nlon), values, label
        self.off = np.concatenate(([0], np.cumsum(self.nlon)))

    def row(self, j):
        return self.values[self.off[j]:self.off[j + 1]]


def from_nc(spec, level, time_index):
    from compare import Field
    f = Field(spec, time_index)
    x = f.level(level)
    lon0 = f.lon[0]
    if abs(((lon0 + 180) % 360) - 180) > 1e-6:
        shift = int(round(-lon0 / (360.0 / len(f.lon)))) % len(f.lon)
        x = np.roll(x, shift, axis=1)
    lat = f.lat
    if lat[0] < lat[-1]:
        lat, x = lat[::-1], x[::-1]
    return Rows(lat, np.full(len(lat), x.shape[1]), x.ravel(), "%s (%s)" % (os.path.basename(f.path), f.var))


def from_grib(path, level, var):
    import core as C
    from gribio import GribFile
    m, vals = GribFile(path).read(level, var)
    pl = m.get("pl")
    N = int(m["N"])
    g = C.GaussGrid.from_pl(N, pl) if pl is not None else C.GaussGrid.regular(N)
    return Rows(g.lat_deg, g.nlon, vals, "%s level %d" % (os.path.basename(path), level))


def from_npz(spec, level):
    import core as C
    path, var = spec.rsplit(":", 1)
    z = np.load(path)
    i = int(np.where(z["levels"] == level)[0][0])
    nlon = z["nlon"]
    g = C.GaussGrid.from_pl(len(nlon) // 2, nlon)
    return Rows(g.lat_deg, nlon, z[var][i].astype(np.float64), "%s (%s)" % (os.path.basename(path), var))


def display(rows, lat_t, lon_t):
    """Resample rows to a (len(lat_t), len(lon_t)) raster, for display only."""
    out = np.empty((len(lat_t), len(lon_t)))
    lat = rows.lat                                   # descending
    cache = {}

    def along(j):
        if j not in cache:
            r = rows.row(j)
            n = len(r)
            x = (lon_t % 360.0) / (360.0 / n)
            i0 = np.floor(x).astype(int)
            f = x - i0
            cache[j] = (1 - f) * r[i0 % n] + f * r[(i0 + 1) % n]
        return cache[j]

    for i, la in enumerate(lat_t):
        k = np.searchsorted(-lat, -la)
        if k <= 0:
            out[i] = along(0)
        elif k >= len(lat):
            out[i] = along(len(lat) - 1)
        else:
            t = (lat[k - 1] - la) / (lat[k - 1] - lat[k])
            out[i] = (1 - t) * along(k - 1) + t * along(k)
    return out


def get(args, which, var, level):
    if getattr(args, which + "nc"):
        return from_nc(getattr(args, which + "nc"), level, getattr(args, which + "time"))
    if getattr(args, which + "grib"):
        return from_grib(getattr(args, which + "grib"), level, var)
    if getattr(args, which + "npz"):
        return from_npz(getattr(args, which + "npz"), level)
    return None


# ---------------------------------------------------------------- map
def cmd_map(a):
    region = a.region or [-180, 180, -90, 90]
    lon0, lon1, la0, la1 = region
    nx = a.width
    ny = max(2, int(round(nx * (la1 - la0) / (lon1 - lon0))))
    lon_t = np.linspace(lon0, lon1, nx, endpoint=False) + 0.5 * (lon1 - lon0) / nx
    lat_t = np.linspace(la1, la0, ny, endpoint=False) - 0.5 * (la1 - la0) / ny
    A = get(a, "", a.var, a.level)
    if A is None:
        sys.exit("give --nc, --grib or --npz")
    title_var = a.var
    if a.speed:
        if not a.grib_v and not a.nc_v:
            sys.exit("--speed needs --grib-v or --nc-v for the v component")
        Vr = from_grib(a.grib_v, a.level, "v") if a.grib_v else from_nc(a.nc_v, a.level, a.time)
        field = np.hypot(display(A, lat_t, lon_t), display(Vr, lat_t, lon_t))
        title_var, cmap, signed = "wind speed", SEQUENTIAL, False
        A.label = A.label + " with " + Vr.label
    else:
        field = display(A, lat_t, lon_t)
        signed = True
        cmap = DIVERGING
    B = get(a, "minus_", a.var, a.level)
    if B is not None:
        field = field - display(B, lat_t, lon_t)
        title_var = "%s difference" % a.var
    lim = np.nanpercentile(np.abs(field), a.pct)
    vmin, vmax = (-lim, lim) if signed else (0.0, np.nanpercentile(field, a.pct))
    fig, ax = plt.subplots(figsize=(9.6, max(3.2, 9.6 * ny / nx + 1.2)))
    im = ax.imshow(field, extent=[lon0, lon1, la0, la1], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)"); ax.grid(False)
    if a.coastlines:
        add_coastlines(ax, region)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend="both" if signed else "max")
    cb.set_label(a.units); cb.outline.set_visible(False)
    t = a.title or "%s, level %d%s" % (title_var, a.level, "" if B is None else ": A minus B")
    ax.set_title(t)
    sub = "A: %s" % A.label + ("" if B is None else "\nB: %s" % B.label)
    sub += "\nColour limits: %g%% of |values|; display raster %dx%d (resampled for display only)" % (a.pct, nx, ny)
    fig.text(0.01, 0.005, sub, fontsize=7.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.07 if B is None else 0.09, 1, 1))
    fig.savefig(a.out); print("wrote", a.out)


def add_coastlines(ax, region):
    """Coastlines from cartopy's Natural Earth data, if cartopy and its data are available."""
    try:
        import cartopy.feature as cf
        geoms = cf.COASTLINE.with_scale("110m" if region[1] - region[0] > 90 else "50m").geometries()
        for g in geoms:
            for line in getattr(g, "geoms", [g]):
                x, y = line.xy
                x = ((np.asarray(x) + 180) % 360) - 180 if region[0] < 0 else np.asarray(x) % 360
                br = np.where(np.abs(np.diff(x)) > 180)[0] + 1
                for xs, ys in zip(np.split(x, br), np.split(np.asarray(y), br)):
                    ax.plot(xs, ys, color=INK, linewidth=0.4)
        ax.set_xlim(region[0], region[1]); ax.set_ylim(region[2], region[3])
    except Exception as e:   # noqa: BLE001
        print("coastlines skipped (%s)" % e)


# ---------------------------------------------------------------- zonal
def cmd_zonal(a):
    z = np.load(a.npz)
    i = int(np.where(z["levels"] == a.level)[0][0])
    lat = z["lat"]; zz = z["zonal"][i]
    la, lb = [str(s) for s in z["labels"]]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
    for ax, (ia, ib), name in ((axs[0], (0, 1), "u"), (axs[1], (2, 3), "v")):
        ax.plot(lat, zz[ia], color=SERIES[0], label=la)
        ax.plot(lat, zz[ib], color=SERIES[1], label=lb)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title("Zonal-mean %s, level %d" % (name, a.level)); ax.set_xlabel("latitude (deg)")
        ax.set_ylabel("m/s"); ax.set_xlim(-90, 90)
    axs[0].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(a.out); print("wrote", a.out)


# ---------------------------------------------------------------- profiles
def robust_flags(levels, y, z=3.0, win=5):
    """Levels whose value departs from the running median of its neighbours by > z robust SDs."""
    y = np.asarray(y, float)
    med = np.array([np.median(y[max(0, k - win // 2):k + win // 2 + 1]) for k in range(len(y))])
    r = y - med
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    if mad == 0:
        return []
    return [int(levels[k]) for k in np.where(np.abs(r) > z * mad)[0]]


def cmd_profiles(a):
    import csv
    rows = list(csv.DictReader(open(a.csv)))
    lev = sorted({int(r["level"]) for r in rows})
    def col(var, key):
        d = {int(r["level"]): float(r[key]) for r in rows if r["var"] == var}
        return np.array([d[L] for L in lev])
    p = col("u", "p_hPa")
    use_p = np.all(np.isfinite(p)) and len(lev) > 1
    y = p if use_p else np.array(lev, float)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 5.2), sharey=True)
    flagged = {}
    for k, (key, title, unit) in enumerate((("bias", "Bias (A minus B)", "m/s"), ("rmsd", "RMS difference", "m/s"),
                                            ("corr", "Correlation", ""))):
        ax = axs[k]
        for j, var in enumerate(("u", "v")):
            x = col(var, key)
            ax.plot(x, y, color=SERIES[j], marker="o" if len(lev) < 3 else None, markersize=4, label=var)
            if key == "rmsd" and len(lev) >= 5:
                fl = robust_flags(lev, x)
                flagged[var] = fl
                for L in fl:
                    ax.plot(x[lev.index(L)], y[lev.index(L)], marker="o", markersize=7, mfc="none",
                            mec=INK, mew=1.2, linestyle="none")
        if key == "bias":
            ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(title); ax.set_xlabel(unit)
    if use_p:
        axs[0].set_yscale("log"); axs[0].invert_yaxis(); axs[0].set_ylabel("global-mean pressure (hPa)")
    else:
        axs[0].invert_yaxis(); axs[0].set_ylabel("model level")
    axs[0].legend(loc="lower left")
    note = "Open circles: levels whose RMSD departs > 3 robust SDs from its 5-level running median. "
    note += "u: %s; v: %s" % (flagged.get("u", []) or "none", flagged.get("v", []) or "none")
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(a.out)
    print("wrote", a.out); print(note)


# ---------------------------------------------------------------- spectra
def ke_spectrum(zeta, delta, lmax, radius=A_EARTH):
    """E_rot(n), E_div(n) in m^2 s^-2 per degree, for coefficients in m-major order
    (m = 0..M, n = m..lmax), normalisation: sphere mean of |Y|^2 = 1."""
    Er = np.zeros(lmax + 1); Ed = np.zeros(lmax + 1)
    off = 0
    m = 0
    while off < len(zeta):
        nk = lmax - m + 1
        wgt = 1.0 if m == 0 else 2.0
        n = np.arange(m, lmax + 1)
        Er[n] += wgt * np.abs(zeta[off:off + nk]) ** 2
        Ed[n] += wgt * np.abs(delta[off:off + nk]) ** 2
        off += nk; m += 1
    n = np.arange(lmax + 1)
    f = np.where(n > 0, radius ** 2 / (2.0 * np.maximum(n * (n + 1), 1)), 0.0)
    return Er * f, Ed * f


def cmd_spectra(a):
    curves = []
    for s in a.spec or []:
        path, label = (s.rsplit(":", 1) + [None])[:2] if ":" in s else (s, None)
        z = np.load(path)
        er, ed = ke_spectrum(z["zeta"], z["delta"], int(z["lmax"]))
        curves.append((label or os.path.basename(path), er, ed))
    for s in a.arco or []:
        path, level, label = s.split(":")
        z = np.load(path)
        T = int(z["T"])
        i = int(np.where(np.asarray(z["levels"]).astype(int) == int(level))[0][0])
        c = {v: z[v][i].astype(np.float64).reshape(-1, 2) for v in ("vo", "d")}
        zeta, delta = (c[v][:, 0] + 1j * c[v][:, 1] for v in ("vo", "d"))
        er, ed = ke_spectrum(zeta * a.arco_scale, delta * a.arco_scale, T)
        curves.append((label + " (ARCO; normalisation assumed, see M4)", er, ed))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    nmax = max(len(c[1]) - 1 for c in curves)
    top = max(max(c[1][1:].max(), c[2][1:].max()) for c in curves)
    for ax, k, title in ((axs[0], 1, "Rotational kinetic energy"), (axs[1], 2, "Divergent kinetic energy")):
        for j, c in enumerate(curves):
            n = np.arange(len(c[k]))
            ax.loglog(n[1:], c[k][1:], color=SERIES[j % 3], label=c[0])
        n_a = max(2, nmax // 20)                       # reference slopes anchored on the first curve
        n0 = np.array([n_a, nmax], float)
        ref = curves[0][k][n_a]
        for slope, ls, lab in ((-3.0, "--", "n^-3"), (-5.0 / 3.0, ":", "n^-5/3")):
            yv = ref * (n0 / n_a) ** slope
            ax.loglog(n0, yv, color=INK2, linestyle=ls, linewidth=0.9)
            ax.text(n0[1], yv[1], " " + lab, color=INK2, fontsize=8, va="center")
        ax.set_xlim(1, nmax * 1.6)
        ax.set_ylim(top * 1e-12, top * 3)
        ax.set_title(title + (", level %s" % a.level if a.level else "")); ax.set_xlabel("total wavenumber n")
    axs[0].set_ylabel("m^2 s^-2 per wavenumber"); axs[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(a.out); print("wrote", a.out)


# ---------------------------------------------------------------- row limits
def cmd_rowlimits(a):
    import core as C
    g = C.GaussGrid.octahedral(a.N)
    T = a.T if a.T is not None else a.N - 1
    nm = g.nmen(T)
    lat = g.lat_deg[:g.N]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(lat, nm[:g.N], color=SERIES[0], label="ecTrans cubic rule (used)")
    ax.plot(lat, np.minimum(T, (g.nlon[:g.N] - 1) // 2), color=SERIES[1], label="(nlon-1)/2, capped at T")
    ax.set_xlabel("latitude (deg, northern hemisphere)"); ax.set_ylabel("maximum zonal wavenumber m")
    ax.set_title("Per-row wavenumber limit, O%d at T%d" % (a.N, T)); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(a.out); print("wrote", a.out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("map")
    for pre in ("", "minus_"):
        m.add_argument("--%snc" % pre.replace("_", "-")); m.add_argument("--%sgrib" % pre.replace("_", "-"))
        m.add_argument("--%snpz" % pre.replace("_", "-"))
        m.add_argument("--%stime" % pre.replace("_", "-"), type=int, default=None)
    m.add_argument("--grib-v"); m.add_argument("--nc-v"); m.add_argument("--speed", action="store_true")
    m.add_argument("--var", default="u"); m.add_argument("--level", type=int, required=True)
    m.add_argument("--region", type=float, nargs=4, metavar=("LON0", "LON1", "LAT0", "LAT1"))
    m.add_argument("--width", type=int, default=1440, help="display raster columns [1440]")
    m.add_argument("--pct", type=float, default=99.5, help="colour limit percentile [99.5]")
    m.add_argument("--coastlines", action="store_true", help="draw coastlines (needs cartopy + Natural Earth data)")
    m.add_argument("--units", default="m/s"); m.add_argument("--title"); m.add_argument("--out", required=True)
    z = sp.add_parser("zonal"); z.add_argument("--npz", required=True); z.add_argument("--level", type=int, required=True)
    z.add_argument("--out", required=True)
    pr = sp.add_parser("profiles"); pr.add_argument("--csv", required=True); pr.add_argument("--out", required=True)
    s = sp.add_parser("spectra"); s.add_argument("--spec", nargs="*", help="PATH[:LABEL] from --spec-out")
    s.add_argument("--arco", nargs="*", help="PATH:LEVEL:LABEL from fetch_arco.py")
    s.add_argument("--arco-scale", type=float, default=1.0); s.add_argument("--level")
    s.add_argument("--out", required=True)
    r = sp.add_parser("rowlimits"); r.add_argument("--N", type=int, default=8000); r.add_argument("--T", type=int)
    r.add_argument("--out", required=True)
    a = p.parse_args(argv)
    {"map": cmd_map, "zonal": cmd_zonal, "profiles": cmd_profiles, "spectra": cmd_spectra,
     "rowlimits": cmd_rowlimits}[a.cmd](a)


if __name__ == "__main__":
    main()
