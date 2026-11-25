# What each file does

All files are published as they were when the results were produced; see [PROVENANCE.md](PROVENANCE.md).
Every Python script prints its usage with `--help` (or in its header comment).

## How the pieces fit

```
                        vod2uv_mpi.py  (Gauss nodes, Legendre recurrence, m-partition)
                              │ imported by
                              ▼
  vo, d GRIB ──► recover/recover_uv.py ──► u, v GRIB (header cloned from the archived u, v)
                  ├── core.py    grids, ecTrans row limits, analysis/synthesis kernels
                  ├── comm.py    MPI (mpi4py) / serial / thread communicators, level groups
                  ├── gribio.py  indexed GRIB reading, template-cloned writing (ecCodes)
                  └── serial.py  single-process reference the MPI driver must match bitwise
  checks: analysis/check_recovered.py (M8), analysis/cosphi_check.py (cos φ test), analysis/compare_runs.py
```

## Top level

| File | What it does |
|---|---|
| `vod2uv_mpi.py` (699 lines) | The first tool: converts ζ, δ on a **regular** Gaussian grid in netCDF to u, v (`srun -n 256 python3 vod2uv_mpi.py vod.nc uv.nc`). Real FFT per latitude row, MPI all-to-all transpose to zonal wavenumbers m, Gauss–Legendre analysis per m, ψ = −a²ζ/(n(n+1)), χ = −a²δ/(n(n+1)), synthesis of u cos φ and v cos φ, transpose back, inverse FFT. Associated Legendre functions come from the three-term recurrence with an extended exponent per (m, latitude), so they do not underflow at T7999 near the poles; no Legendre table is stored, so memory does not grow with T³. It was written because CDO's `dv2uv` stores that table (about 3 TB at T7999, estimated). Superseded for XNR1K by `recover/`, which reads the native reduced O8000 GRIB files, but `recover/core.py` imports its Gauss-node, Legendre and partition functions, so it is required. Notes: [vod2uv_mpi.md](vod2uv_mpi.md) |
| `requirements.txt` | Python dependencies with installation notes (see [INSTALL.md](INSTALL.md)) |

## `recover/`: the recovery used for XNR1K

| File | What it does |
|---|---|
| `recover_uv.py` | MPI driver. World ranks are split into groups (`--group-size`); groups take whole levels round-robin. In a group, the root decodes the vo and d GRIB messages and scatters latitude-row blocks; each rank does the row FFTs (row length nlon_j, wavenumbers m ≤ NMEN_j by the ecTrans rule), an all-to-all transpose to blocks of m, Gauss–Legendre analysis to T`lmax`, ψ and χ, synthesis of u, v on the output grid, the transpose back and inverse FFTs; the root gathers and writes. Output `--out-grid native` writes GRIB2 messages cloned from the corrupted u/v files (`--template-u/-v`), changing only the values; `F<N>`/`O<N>` writes per-level `.npz` parts merged to netCDF or `.npz`; `--spec-out` saves the ζ, δ coefficients. Per-level parts are written to a temporary name and renamed, so a rerun skips finished levels (`--overwrite` to redo) and the final step merges the parts unless `--no-merge`. Logs time and memory per level |
| `core.py` | Gaussian grids (regular, octahedral, any `pl`); the per-row wavenumber limit used by ecTrans; row ↔ Fourier conversion; batched Legendre analysis/synthesis (`process_mbatch`). Conventions: rows north → south as in the XNR1K GRIB; P̄ normalised so that ∫P̄²dμ = 2; Earth radius 6,371,229 m |
| `comm.py` | Communicator layer: mpi4py, a serial stand-in and a thread-based stand-in for tests. Only bcast, barrier, allreduce_max, alltoallv, scatterv/gatherv, gather and split are used |
| `gribio.py` | GRIB I/O with ecCodes. Finds messages by reading each 16-byte indicator section and seeking, so level k is reached without reading the earlier messages (each message is about 512 MB packed, 2 GB decoded). Writes output by cloning the matching template message and setting only the values (16-bit simple packing). Also an in-memory source/sink for tests |
| `serial.py` | Single-process reference pipeline with the same kernels; the MPI driver must reproduce it bitwise (test M7) |
| `compare.py` | Per-level statistics of A against B on the same regular grid: bias, RMSD, correlation, vector-wind RMSD; zonal means for plots. Nothing is interpolated: grids must match |
| `plots.py` | Figures: `map` (field, difference or wind speed; global or regional; GRIB, netCDF or reduced-grid `.npz`), `zonal`, `profiles`, `spectra` (rotational and divergent kinetic energy), `rowlimits`. Reduced grids are resampled to a display raster for maps only, labelled as such, and never used for statistics |
| `README.md` | Development notes on `recover/` and the tests passed at the time (kept unedited) |

### `recover/tests/`

| File | What it tests | Needs |
|---|---|---|
| `test_unit.py` | New kernel against the validated `vod2uv_mpi` kernel; ecTrans row limits on O8000, O1280, O48; O8000 point count = 256,288,000 | numpy |
| `test_m1_analytic.py` | **M1**: analytic fields (Rossby–Haurwitz and divergent waves) through the full pipeline on O48–O160 and to other grids and truncations; pass if the relative error ≤ 3e-7 | numpy |
| `test_m7_driver.py` | **M7**: the driver with 1–6 thread-ranks and several group sizes must be bitwise identical to `serial.py`; m-partition coverage and balance | numpy |
| `test_outputs.py` | Regular-grid output, spectral-coefficient output, restart skipping | numpy |
| `test_plots.py OUTDIR` | Comparison statistics on fields with known bias and noise; flagged outlier level; Parseval check of the KE spectrum; writes test figures | numpy, scipy, matplotlib |
| `grib_selftest.py`, `selftest_grib.sh [ranks]` | GRIB + MPI path on small analytic GRIB2 files: (A) output equals the serial pipeline applied to the decoded 16-bit inputs to within half a packing step; (B) error against the analytic fields, reported; (C) header keys identical to the template (M8) | mpi4py, ecCodes, Slurm `srun` |

## `tests/`: tests of `vod2uv_mpi.py`

| File | What it does |
|---|---|
| `test_core.py` | Gauss nodes and weights; P̄ and H = (1−μ²)dP/dμ against `scipy.special.lpmv` for n ≤ 60 |
| `test_random_scipy.py` | Random full spectrum at T127 against a reference built with `scipy.special.sph_harm_y_all` |
| `test_pipeline.py` | End-to-end on analytic netCDF files with 1, 3, 4 and 7 simulated ranks (`fakempi.py`); writes to `/tmp/v2u` |
| `test_ortho_T7999.py` | Orthonormality of the Legendre functions at nlat = 16,000, T7999, for several m (slow; writes `/tmp/gl16000.npy`) |
| `bench_T7999.py` | Timing of the T7999 kernel (reads `/tmp/gl16000.npy` from the previous test) |
| `make_testfile.py`, `check_output.py` | Write a CDO-like netCDF file with analytic vo, d and exact u, v; compare a tool output with it |
| `check_fd.py vod.nc uv.nc` | Independent check without spectral code: recomputes ζ, δ from u, v by finite differences and compares them with the input after coarse-graining (expects slope ≈ 1) |
| `fakempi.py` | Thread-based stand-in for an MPI communicator (tests only) |
| `selftest_mpi.sh "srun -n 8 -c 1"` | Real MPI and netCDF4 path on a cluster: parallel run against a single-rank run, which must be identical |

## `analysis/`: validation and diagnosis

Test names (M1–M8) are those of the validation plan; the criteria were fixed before the tests were run. The
paper's Table II lists them.

| File | What it does |
|---|---|
| `m2_era5.py` | ERA5 tests for one time (2018-11-01 00 UTC). (1) Calibrates the ARCO spectral convention against the CDS gridded vo, d (fit of m = 0, even m, odd m; must land on simple factors at packing level poleward of 30°). (2) **M2a**: u, v from the ARCO T639 spectra against ECMWF's u, v (CDS). (3) **M2**: u, v from the CDS *gridded* vo, d through our grid-input pipeline against CDS u, v (criterion: RMS ≤ q and max ≤ 4q per level, q the packing step). **M4**: our analysis of the gridded vo, d against the ARCO coefficients, relative to the projected packing noise |
| `m2_diag.py` | Diagnoses an M2/M2a residual: re-quantises our field with the CDS message's own packing to see whether the residual is only packing |
| `m3_reduced.py` | **M3**: synthesise the ARCO vo, d exactly on a reduced grid (O640 formal; N320 reported), run our pipeline, compare with the exact F320 u, v (criterion: relative RMS ≤ 1e-6). `--selftest` checks that a too-coarse grid fails |
| `check_recovered.py` | **M8** and sanity for one recovered level: header keys identical to the template; 256,288,000 finite 16-bit values; zonal means and the archived/recovered ratio |
| `cosphi_check.py` | The cos φ test for one level: per 5° band, slope of archived on recovered against the band's cos φ (and cos² φ as the alternative); residual r = archived − recovered·cos φ against the packing bound q_c/2 + cos φ·q_r/2; residual by band. `--csv` for the budget |
| `cosphi_budget.py` | Over all levels: the part δ of the residual that packing does not explain, from rms(r)² = σ_p² + δ² |
| `residual_diag.py` | Shape of the residual for one level: mean, RMS, kurtosis, histogram in units of q_c, latitude structure, zonal-mean share (distinguishes rounding, truncation, double quantisation, transform noise) |
| `packing_steps.py` | Reads the packing step of every message from sections 0–5 only (seconds for 137 levels); predicts the wind error from 16-bit packing of the archived vo, d |
| `compare_runs.py` | Two runs, merged files or parts directories: byte-identical per message, or the maximum difference in packing steps. Exit 0 only if all compared levels are identical |
| `zonal_ratio_check.py` | The first check (7 Oct 2026): zonal-mean u of XNR1K against ERA5 per level, regression slope and correlation. Indicative only when the two files cover different times |

## `era5/`

| File | What it does |
|---|---|
| `download_era5.py` | CDS (MARS) requests for 2018-11-01 00 UTC, model levels 1–137: (A) vo, d, u, v on the regular F320 grid, `truncation: none` (the file M2, M2a and M4 use); (B) vo, d on the reduced N320 grid; (C) vo, d as T639 spherical harmonics; (D) lnsp, spectral and on F320, for level-to-pressure. Each file is checked for its expected message count. `--only` selects requests; `--dry-run` prints them |
| `fetch_arco.py` | ERA5 T639 spectral vo, d from `gs://gcp-public-data-arco-era5/co/model-level-wind.zarr-v2` for one time, saved as `.npz`; refuses to save if the time, the 137 levels or the coefficient count are unexpected |

## `scripts/` (Slurm, as used on OLCF Andes)

The OLCF paths and project account inside them are the ones that were used; set `W` (working directory),
`X` (input directory) and `VENV` for another system, and change `#SBATCH -A`.

| File | What it does |
|---|---|
| `run_L074.sh [ranks]` | First XNR1K recovery: level 74 of step 000000 in one 32-rank group, inside an allocation. Records code checksums, Python and ecCodes paths and input file stats in `run_info.txt`; refuses to overwrite |
| `run_000000.sbatch` | All 137 levels of step 000000 on 4 nodes (one 32-rank group per node), version 4. Runs levels in short chunks, each its own `srun` step, logs node memory between chunks, refuses a Python environment on Lustre, uses Open MPI without UCX by default, tolerates a non-zero `srun` exit only if every level of the chunk has both parts, then merges. Restartable |
| `status_000000.sh [JOBID]` | Read-only status of a run: scheduler state, levels done, parts, merged files, errors, memory |
| `check_parts.sh RUN_DIR [NPAR]` | `check_recovered.py` and `cosphi_check.py` on every finished level, in parallel; summary in `RUN_DIR/checks/summary.txt` |
| `postrun_000000.sh` | All post-run checks: chunks with non-zero exits; per-level checks; residual budget and the pre-registered Spearman test of the single-precision hypothesis; merged file = concatenated parts (md5); first attempt against clean run (levels 1–41); provenance |
| `provenance.sh [--hash-inputs]` | Manifest: host, modules, Python packages, the MPI library mpi4py links to, checksums of code and results, sizes or checksums of inputs; never overwrites |

## Not included

The project's development records (work log, handoff notes, findings), Globus sync and usage-accounting scripts,
an experimental single-precision emulation (`--emulate-fp32`, staged but never used for any published result),
figure-drawing scripts for presentations, and all data.
