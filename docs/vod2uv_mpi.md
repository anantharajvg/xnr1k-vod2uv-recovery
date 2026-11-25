# vod2uv_mpi

MPI-distributed vorticity/divergence → u/v on a regular Gaussian grid (designed for N8000 / TCo7999: 16000 × 32016).

## Why CDO fails
CDO precomputes and keeps the associated Legendre polynomials (and their derivatives) for every (n, m, latitude) in memory. The developers quote ~13 GB at T1279 (CDO ≥ 2.0). That cost scales like T³, so T7999 needs roughly 13 GB × (8000/1280)³ ≈ 3 TB. The input data is only 4 GB.

## Algorithm
1. Each rank reads a contiguous block of latitude rows and does an rFFT along longitude.
2. `MPI_Alltoallv`: rows → contiguous zonal-wavenumber (m) ranges, split so each rank gets about the same Legendre cost.
3. For each m: Gauss–Legendre analysis gives ζₙᵐ and δₙᵐ. Then ψ = −a²ζ/n(n+1) and χ = −a²δ/n(n+1), and the synthesis computes u·cosφ and v·cosφ (IFS formulation).
4. Transpose back, inverse FFT, and rank 0 writes the output as a stream.

Legendre functions are built on the fly with the 3-term recurrence. Each (m, latitude) carries its own extended exponent, so P_mᵐ ∝ sinᵐθ does not underflow at T7999 near the poles. Equatorial symmetry halves the work. No table is stored.

## Requirements
`numpy`, `mpi4py`, `netCDF4` (a serial build is fine: every rank reads its own rows, and only rank 0 writes). `scipy` is a fallback that handles small netCDF-3 files only.

## Run
```bash
export OMP_NUM_THREADS=1            # one rank per core; the kernel is single-threaded numpy
srun -n 256 python3 vod2uv_mpi.py vod.rgg.nc uv.rgg.nc            # Earth radius 6371229 m (IFS)
srun -n 256 python3 vod2uv_mpi.py --radius 6371000 vod.rgg.nc uv.nc   # to match CDO's radius
```
Options: `--truncation cubic|quadratic|linear` (default cubic: T = nlat/2 − 1 = 7999), `--lmax`, `--vor/--div` variable names, `--deflate N`, `--mbatch` (default 4, the fastest in benchmarks).

## Verification done (see tests/)
| test | result |
|---|---|
| Gauss weights vs long-double reference (N=1280) | ≤ 3e-11 rel (scipy: 2e-8) |
| P̄ₙᵐ, H vs scipy `lpmv` (n ≤ 60) | 4e-14 |
| Orthonormality, nlat=16000, lmax=7999, m ∈ {0,1,1500,4000,6500} | max abs(Gram − I) ≤ 7e-13 |
| Random full spectrum T127, float64, vs scipy `sph_harm_y_all` reference | 1e-13 |
| Analytic Rossby–Haurwitz (R=40,150,1000) + divergent wave, end-to-end, float32 I/O | 5–9e-8 (float32 roundoff) |
| 1 vs 3 vs 7 simulated ranks | bitwise identical |

**Not tested here** (the sandbox had no MPI library, mpi4py or netCDF4): the real `mpi4py` calls and the `netCDF4` backend. Run `tests/selftest_mpi.sh "srun -n 8"` on the cluster before production use.

## Cost (measured on the sandbox CPU, one core)
About 4800 core-seconds of Legendre work per 2-D slice at T7999, so roughly 19 s on 256 ranks or 5 s on 1024. Peak memory per rank is about 1 GB at 256 ranks. Rank 0 writes the 4 GB of output serially.
