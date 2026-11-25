"""GRIB input/output for the recovery, plus an in-memory stand-in used by tests.

GRIB files here hold one 256,288,000-point field per message (~512 MB packed,
~2 GB decoded). Messages are located by reading the 16-byte GRIB indicator
section and seeking, so finding level k never reads the k-1 messages before it.
Only the group root calls read/write.
"""
import os

import numpy as np

KEYS = ("shortName", "level", "typeOfLevel", "gridType", "N", "numberOfValues", "edition",
        "jScansPositively", "iScansNegatively", "bitmapPresent", "packingType", "bitsPerValue",
        "longitudeOfFirstGridPointInDegrees", "dataDate", "dataTime", "step", "paramId")


def scan_offsets(path):
    """[(offset, length)] of every GRIB message in the file."""
    out = []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        pos = 0
        while pos < size:
            f.seek(pos)
            h = f.read(16)
            if len(h) < 16:
                break
            if h[:4] != b"GRIB":
                raise ValueError("%s: no GRIB indicator at byte %d" % (path, pos))
            ed = h[7]
            if ed == 2:
                n = int.from_bytes(h[8:16], "big")
            elif ed == 1:
                n = int.from_bytes(h[4:7], "big")
            else:
                raise ValueError("%s: unknown GRIB edition %d at byte %d" % (path, ed, pos))
            out.append((pos, n))
            pos += n
    return out


def _ec():
    import eccodes
    return eccodes


def meta_of(h):
    ec = _ec()
    m = {}
    for k in KEYS:
        if ec.codes_is_defined(h, k):
            try:
                m[k] = ec.codes_get(h, k)
            except Exception:   # noqa: BLE001
                m[k] = None
    if ec.codes_is_defined(h, "pl"):
        m["pl"] = np.asarray(ec.codes_get_array(h, "pl"), dtype=np.int64)
    return m


class GribFile:
    def __init__(self, path):
        self.path = path
        self.index = scan_offsets(path)

    def raw(self, i):
        off, n = self.index[i]
        with open(self.path, "rb") as f:
            f.seek(off)
            b = f.read(n)
        if len(b) != n or b[-4:] != b"7777":
            raise ValueError("%s: message %d truncated or missing end section" % (self.path, i))
        return b

    def _handle(self, level, short_name):
        """Decode message level-1 and check it is the requested level/variable.
        XNR1K files hold levels 1..137 in order; anything else is an error, not a search."""
        ec = _ec()
        i = level - 1
        if not 0 <= i < len(self.index):
            raise ValueError("%s: no message for level %d (%d messages)" % (self.path, level, len(self.index)))
        h = ec.codes_new_from_message(self.raw(i))
        m = meta_of(h)
        if m.get("level") != level or (short_name is not None and m.get("shortName") != short_name):
            ec.codes_release(h)
            raise ValueError("%s: message %d is %s level %s, expected %s level %d (file not ordered by level?)"
                             % (self.path, i, m.get("shortName"), m.get("level"), short_name, level))
        return h, m

    def meta(self, level, short_name=None):
        h, m = self._handle(level, short_name)
        _ec().codes_release(h)
        return m

    def read(self, level, short_name=None):
        ec = _ec()
        h, m = self._handle(level, short_name)
        try:
            vals = np.asarray(ec.codes_get_values(h), dtype=np.float64)
        finally:
            ec.codes_release(h)
        return m, vals


class GribSource:
    """vo and d from separate GRIB files (XNR1K layout)."""

    def __init__(self, vo_path, d_path):
        self.files = {"vo": GribFile(vo_path), "d": GribFile(d_path)}

    def nlevels(self):
        return len(self.files["vo"].index)

    def read(self, var, level):
        return self.files[var].read(level, var)


class GribTemplateSink:
    """Writes recovered u, v per level by cloning the corresponding message of the
    corrupted u, v files, so every header key stays the same; only values change."""

    def __init__(self, u_template, v_template, out_dir):
        self.tpl = {"u": GribFile(u_template), "v": GribFile(v_template)}
        self.paths = {"u": u_template, "v": v_template}
        self.parts = os.path.join(out_dir, "parts")
        os.makedirs(self.parts, exist_ok=True)

    def part_path(self, var, level):
        return os.path.join(self.parts, "%s_L%03d.grib" % (var, level))

    def done(self, var, level):
        return os.path.exists(self.part_path(var, level))

    def template_meta(self, var, level):
        return self.tpl[var].meta(level, var)

    def write(self, var, level, values):
        ec = _ec()
        h, _ = self.tpl[var]._handle(level, var)
        try:
            if ec.codes_get(h, "numberOfValues") != values.size:
                raise ValueError("template %s level %d has %d values, recovered field has %d"
                                 % (var, level, ec.codes_get(h, "numberOfValues"), values.size))
            bpv = int(ec.codes_get(h, "bitsPerValue"))
            if bpv <= 0:
                raise ValueError("template %s level %d has bitsPerValue %d (constant field?); "
                                 "refusing to guess the packing precision" % (var, level, bpv))
            ec.codes_set_values(h, values)
            if int(ec.codes_get(h, "bitsPerValue")) != bpv:
                ec.codes_set(h, "bitsPerValue", bpv)          # repacks at the template precision
            got = int(ec.codes_get(h, "bitsPerValue"))
            if got != bpv:
                raise ValueError("%s level %d: written bitsPerValue %d, template %d" % (var, level, got, bpv))
            msg = ec.codes_get_message(h)
        finally:
            ec.codes_release(h)
        p = self.part_path(var, level)
        with open(p + ".tmp", "wb") as fo:
            fo.write(msg)
        os.replace(p + ".tmp", p)

    def final_name(self, var, levels, nlev_template):
        base = os.path.basename(self.paths[var])
        stem = base[:-5] if base.endswith(".grib") else base
        tag = "" if list(levels) == list(range(1, nlev_template + 1)) else \
            ".L" + ("%03d" % levels[0] if len(levels) == 1 else "%03d-%03d" % (levels[0], levels[-1]))
        return stem + ".recovered" + tag + ".grib"

    def merge(self, var, levels, out_dir):
        name = os.path.join(out_dir, self.final_name(var, levels, len(self.tpl[var].index)))
        with open(name + ".tmp", "wb") as fo:
            for L in levels:
                with open(self.part_path(var, L), "rb") as fi:
                    while True:
                        b = fi.read(64 << 20)
                        if not b:
                            break
                        fo.write(b)
        os.replace(name + ".tmp", name)
        return name


# ----------------------------------------------------------------------------- tests / small grids
class MemorySource:
    """Test stand-in: {'vo': {level: values}, 'd': {...}} plus a metadata dict per level."""

    def __init__(self, fields, meta):
        self.fields, self.meta = fields, meta

    def nlevels(self):
        return len(self.fields["vo"])

    def read(self, var, level):
        m = dict(self.meta)
        m["level"] = level
        m["shortName"] = var
        return m, np.array(self.fields[var][level], dtype=np.float64)


class NpzSink:
    """Per-level .npz parts (any output grid); merged to one netCDF by merge_netcdf."""

    def __init__(self, out_dir):
        self.parts = os.path.join(out_dir, "parts")
        os.makedirs(self.parts, exist_ok=True)

    def part_path(self, level):
        return os.path.join(self.parts, "L%03d.npz" % level)

    def done(self, var, level):
        return os.path.exists(self.part_path(level))

    def write_level(self, level, arrays):
        p = self.part_path(level)
        np.savez(p + ".tmp.npz", **arrays)
        os.replace(p + ".tmp.npz", p)
