# recover/ — XNR1K u, v recovery from native O8000 vo, d

Software phase of the approved plan (revision 3).

| File | Purpose |
|---|---|
| `core.py` | Gaussian grids (regular, octahedral, any `pl`), ecTrans per-row wavenumber limit, Legendre analysis/synthesis kernels |
| `serial.py` | Single-process reference pipeline (the MPI driver must match it bitwise) |
| `comm.py` | mpi4py wrapper with level groups (`split`), serial and thread stand-ins |
| `gribio.py` | GRIB2 input (seek-indexed messages, single decode), output cloned from the corrupted u/v messages |
| `recover_uv.py` | MPI driver |
| `compare.py` | Per-level statistics of A against B on the same regular grid (bias, RMSD, correlation, vector RMSD), zonal means, optional pressure per level |
| `plots.py` | Figures: `map` (field, difference, wind speed; global or region; GRIB, netCDF or reduced-grid npz), `zonal`, `profiles` (with flagged levels), `spectra` (rotational/divergent KE), `rowlimits` |
| `tests/` | M1, M7 and output tests (run anywhere); GRIB self-test (Andes, needs ecCodes) |
| `../era5/fetch_arco.py` | ERA5 vo, d spectral coefficients from ARCO (login node only) |

## Tests already passed here (no MPI library, no ecCodes in my workspace)

| Test | Result |
|---|---|
| Unit: new kernel vs validated vod2uv kernel | relative difference 3e-16 |
| Unit: O8000 point count from the octahedral formula | 256,288,000 (matches the GRIB) |
| M1: analytic fields, O48 to O640, including other output grids and truncation | max error 2e-14 to 6e-11 relative (2e-8 where m sits beyond the grid's row limits); pass threshold 3e-7 |
| M7: driver with 1, 3, 4, 5, 6 thread-ranks and group sizes 1 to 5 | bitwise identical to serial |
| compare + plots: known bias and noise recovered (bias within 0.023 m/s, RMSD within 0.7%); outlier level flagged; KE spectrum sums to the grid KE (Parseval, 1e-9) | pass |
| Outputs: F40/T79 with vo, d and spectral coefficients; restart skip | identical to serial; coefficients consistent to 3e-8 (float32 storage); both levels skipped on rerun |

On Andes, 7 Oct (details: `../docs/worklog-2026-10-07.md`): all of the above reproduced; M7 bitwise in all layouts
after the m-partition fix; GRIB + MPI self-test (O80, 4 ranks, 2 groups) passed after the output-precision fix:
16-bit output, check A within half a packing step, header keys identical to the template.

## On Andes

```bash
source ~/venvs/vod2uv/bin/activate          # numpy, mpi4py, netCDF4, eccodes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd <bundle>/recover
python3 tests/test_unit.py && python3 tests/test_m1_analytic.py && python3 tests/test_m7_driver.py && python3 tests/test_outputs.py && python3 tests/test_plots.py $TMPDIR/plots_test
bash tests/selftest_grib.sh 4               # inside an allocation; prints PASS/FAIL for checks A, B, C
```

On a login node (ARCO is reachable only there):
```bash
pip install xarray zarr gcsfs
python3 ../era5/fetch_arco.py --time 2018-11-01T00 --out $W/era5/era5_arco_sh_vo_d_2018110100.npz
```

Level 74 recovery is the step after Gate A (M2, M3 pass); the command is in the header of `recover_uv.py`.
Do not run it before then.

## Figures

Run on a compute node for O8000 GRIB input (each level decodes to about 2 GB):
```bash
srun -n 1 python3 plots.py map --grib $W/run_L074/u_131_h3f7-ndjf_000000.ml-dgg.O8000.global.recovered.L074.grib \
     --level 74 --var u --out u_L074.png
srun -n 1 python3 plots.py map --grib REC_u.grib --minus-grib $X/u/u_131_h3f7-ndjf_000000.ml-dgg.O8000.global.grib \
     --level 74 --var u --out u_recovered_minus_corrupted_L074.png
python3 plots.py profiles --csv cmp_stats.csv --out profiles.png
python3 plots.py rowlimits --N 8000 --out rowlimits_O8000.png
```
Maps of reduced grids are resampled to a display raster; this is labelled on every map and is never used for statistics.
Coastlines need cartopy and its Natural Earth data (`--coastlines`); without them the map is drawn without coastlines.
