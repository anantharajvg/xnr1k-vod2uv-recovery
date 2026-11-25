# xnr1k-vod2uv-recovery

Software used to diagnose and recover the corrupted model-level winds (u, v) of the ECMWF IFS
Experimental Nature Run at 1 km (XNR1K), archived at the Oak Ridge Leadership Computing Facility (OLCF).
The winds are rebuilt from the archived relative vorticity (ζ) and divergence (δ) through a spherical-harmonic
transform, without the original post-processing software.

This repository accompanies the paper

> V. G. Anantharaj and I. M. Dulá Razzolini, "Human–Agent Forensics and Recovery of an Extreme-Scale Climate
> Dataset," IEEE Workshop on Curation and Contextualization of Extreme Datasets (CCED 2026), at IEEE BigData 2026,
> Phoenix, AZ, December 2026 (accepted; to appear in the IEEE BigData 2026 proceedings).

A separate ORNL technical report with the full theory and validation is in preparation.

## What was wrong, and what this code does

The archived XNR1K model-level u and v are the true winds multiplied by cos(latitude): the final division by
cos φ was missing when the files were written. This was found from the data alone and verified point by point
(see [docs/REPRODUCE.md](docs/REPRODUCE.md), step 5).

The code here recomputes u and v from the archived ζ and δ on the native O8000 grid (TCo7999, 256,288,000
points per field, 137 model levels):

1. analysis of ζ, δ to spherical harmonics up to T7999 (row-wise FFT with the ecTrans per-row wavenumber limit,
   then Gauss–Legendre quadrature);
2. streamfunction and velocity potential, ψ = −a²ζ/(n(n+1)), χ = −a²δ/(n(n+1)), with a = 6,371,229 m;
3. synthesis of u cos φ and v cos φ, division by cos φ, inverse FFT on the O8000 rows;
4. output written into copies of the archived (corrupted) GRIB2 messages, so every header key is unchanged and
   only the values differ (16-bit simple packing, as in the archive).

The work is MPI-parallel: ranks form groups, each group processes whole model levels, and inside a group the
data are transposed between latitude rows and zonal wavenumbers.

## How the code was written and reviewed

All of the code in this repository was written by an AI agent, Claude (Anthropic), working in its own
sandbox, under the direction of V. G. Anantharaj, who approved every test criterion before it was run and ran every
job on OLCF systems. The paper describes this workflow.

- Confidence in the code rests mainly on tests against independent references: analytic solutions, ECMWF's own
  ERA5 winds, and the archive itself divided by cos φ. They are listed in [docs/REPRODUCE.md](docs/REPRODUCE.md),
  with the failures as well as the passes.
- Human review: after the paper was submitted, V. G. Anantharaj reviewed the high-level logic and structure of
  the transform module, `vod2uv_mpi.py`; this was not a line-by-line review. The rest of the code (`recover/`,
  `analysis/`, `era5/`, `scripts/`) has not been reviewed by a person. A colleague has been asked to review the code.
- The files are published exactly as they were when the results were produced (checksums in
  [docs/PROVENANCE.md](docs/PROVENANCE.md)). Comments inside the code and the notes in `recover/README.md` and
  `docs/vod2uv_mpi.md` were written during development and refer to the development setting ("my workspace",
  "Andes"); they are kept unedited.

## Repository layout

| Path | Contents |
|---|---|
| `vod2uv_mpi.py` | The first tool (regular Gaussian grid, netCDF input). It holds the validated Legendre kernels that `recover/` imports, so it is required. See [docs/vod2uv_mpi.md](docs/vod2uv_mpi.md) |
| `recover/` | The recovery used for XNR1K: native-grid driver `recover_uv.py`, transform kernels `core.py`, MPI layer `comm.py`, GRIB I/O `gribio.py`, serial reference `serial.py`, comparison and plotting tools |
| `recover/tests/` | Tests of `recover/`: unit, analytic (M1), parallel layouts (M7), outputs, plots; GRIB + MPI self-test |
| `tests/` | Tests of `vod2uv_mpi.py`: Gauss weights, Legendre functions, random spectra, analytic pipeline, MPI self-test |
| `analysis/` | Validation and diagnosis: ERA5 tests (M2, M2a, M4, M3), header and sanity check (M8), the cos φ test, residual budget and diagnostics, run comparison |
| `era5/` | Downloads of the ERA5 reference data (Copernicus CDS; Google ARCO-ERA5) |
| `scripts/` | The batch and post-run scripts used on OLCF Andes (Slurm) |
| `docs/` | [INSTALL.md](docs/INSTALL.md), [CODE.md](docs/CODE.md) (every file), [REPRODUCE.md](docs/REPRODUCE.md), [PROVENANCE.md](docs/PROVENANCE.md); `docs/provenance/`: the run-time manifests and `pip freeze` from OLCF |

## Quick start (no MPI, no data)

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
cd recover
python3 tests/test_unit.py && python3 tests/test_m1_analytic.py && python3 tests/test_m7_driver.py \
  && python3 tests/test_outputs.py && python3 tests/test_plots.py /tmp/plots_test
cd ../tests
python3 test_core.py && python3 test_random_scipy.py && python3 test_pipeline.py
```

Each test prints its numbers and `PASS`; the expected values are in [docs/REPRODUCE.md](docs/REPRODUCE.md), step 1.
For the full environment (MPI, ecCodes, netCDF4) see [docs/INSTALL.md](docs/INSTALL.md).

## Data

No data are included. The XNR1K files are on OLCF's Orion file system and need OLCF project access; the scripts
contain the OLCF paths that were used. The ERA5 reference data are public (Copernicus Climate Data Store; Google
ARCO-ERA5), and `era5/` downloads them.

## Status and known limitations

- One of 961 output times (step 000000) has been recovered, all 137 levels. The other 960 have not been run.
- After the cos φ factor is removed, the recovered winds agree with the archive to within the archive's 16-bit
  rounding plus a small residual (0.15–2.29 mm/s RMS across levels; medians 2.9e-5 (u) and 6.5e-5 (v) of the
  wind's RMS). Its cause is not
  established. The leading hypothesis, single-precision arithmetic in the original computation, is
  **not confirmed**, and our own pre-registered test weakened it.
- One pre-registered ERA5 test (M2, gridded input) failed at model level 135 (u); we attribute it to the ERA5
  reference fields, not to this code. It is recorded as a failure.
- The GRIB and MPI paths were run only on OLCF Andes (Open MPI 4.0.4), which has since been retired. Three runs
  there died with bus errors inside the MPI transport; the workaround (Open MPI without UCX) is built into
  `scripts/run_000000.sbatch`. The cause was not established.
- Byte-identical results were verified on Andes: level 74 in three runs (two Python environments, two MPI
  transports), and 79 messages of levels 1–41 between two independent runs of step 000000. On other machines (other
  BLAS, FFT or MPI libraries), results may differ at the level of a packing step; this has not been tested.

## Licence

See [LICENSE](LICENSE).

## Acknowledgments

This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory,
which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.
