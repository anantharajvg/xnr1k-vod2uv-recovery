# Reproducing the results

Six steps, from tests that run on a laptop in a minute to the full recovery of one XNR1K output time on a
cluster. The expected results are those recorded during the work (OLCF Andes, 7–9 October 2026) unless
marked otherwise. Test criteria were written down before each test was first run; the failures are listed as well.

Set up the environment first ([INSTALL.md](INSTALL.md)) and, for every step:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
```

| Step | What | Where | Time |
|---|---|---|---|
| 1 | Unit, analytic and parallel-layout tests | any computer, no MPI | about 1 min |
| 2 | Real MPI and GRIB self-tests | cluster allocation | minutes |
| 3 | Validation against ERA5 (M2a, M2, M4, M3) | cluster, internet for downloads | hours (downloads queue) |
| 4 | XNR1K level 74, and the cos φ diagnosis | one 32-core node, XNR1K access | about 7 min per level |
| 5 | XNR1K step 000000, all 137 levels | 4 nodes | about 4 h |
| 6 | Post-run checks | one node | about 1 h |

## Step 1. Tests without MPI or data

```bash
cd recover
python3 tests/test_unit.py
python3 tests/test_m1_analytic.py
python3 tests/test_m7_driver.py
python3 tests/test_outputs.py
python3 tests/test_plots.py /tmp/plots_test
cd ../tests
python3 test_core.py
python3 test_random_scipy.py
python3 test_pipeline.py
```

Expected output (rerun on 25 Nov 2026, Python 3.13.18, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2; every
command exited with status 0):

| Test | Key lines |
|---|---|
| `test_unit.py` | `regression vs validated kernel: max\|diff\| = 2.40e-10 (scale 8.06e+05)` (relative 3e-16; Andes gave 2.60e-10); `O8000 points: 256288000` |
| `test_m1_analytic.py` | five cases, relative errors 1e-13 to 2e-8, each `PASS`; `M1: PASS` |
| `test_m7_driver.py` | partitions cover all m; `oversubscribed partition refused (expected)`; `M7: PASS` |
| `test_outputs.py` | `output vs serial (float32 parts): max diff 0.0e+00`; `restart run skipped 2/2 levels`; `outputs: PASS` |
| `test_plots.py` | `Parseval: ... rel diff 1.3e-09`; `plots tests: PASS` |
| `test_core.py` | Gauss weights sum to 2; `Pbar vs scipy (n<=60): max rel err 4.0e-14 ; H: 3.3e-14` |
| `test_random_scipy.py` | `lmax=127 random spectrum: max\|u-ref\|/max\|V\| = 1.01e-13` |
| `test_pipeline.py` | errors about 5e-8 (float32 files); `rank-count invariance: ... 0.00e+00` |

Slow, optional: `python3 tests/test_ortho_T7999.py` (orthonormality at T7999; recorded result max |Gram − I| ≤ 7e-13).

## Step 2. MPI and GRIB self-tests (inside an allocation)

```bash
bash recover/tests/selftest_grib.sh 4          # 4 ranks in 2 level groups, small analytic GRIB2 files
bash tests/selftest_mpi.sh "srun -n 8 -c 1"    # vod2uv_mpi.py: parallel against single-rank
```

Expected: `selftest_grib.sh` prints PASS for checks A (output equals the serial pipeline on the decoded 16-bit
inputs to within half a packing step) and C (header keys identical to the template); B is reported. This passed
on Andes on 7 Oct after one fix to the output precision. `selftest_mpi.sh` must report
`MPI vs single-rank max abs diff: 0.000e+00 -> PASS`.

## Step 3. Validation against ERA5 (Gate A)

ERA5 is the one data set where both the winds and the vorticity and divergence they come from are available:
ECMWF derives ERA5 model-level u, v from the spectral vo, d when they are retrieved.

```bash
# on a machine with internet access (on Andes: a login node)
python3 era5/fetch_arco.py --time 2018-11-01T00 --out era5_arco_sh_vo_d_2018110100.npz
python3 era5/download_era5.py --outdir era5_cds --only era5_ml_F320_vo_d_u_v.grib   # needs ~/.cdsapirc

# then, on a compute node
python3 analysis/m2_era5.py --arco era5_arco_sh_vo_d_2018110100.npz --cds era5_cds/era5_ml_F320_vo_d_u_v.grib \
        --levels all --csv m2_all.csv
python3 analysis/m3_reduced.py --selftest
python3 analysis/m3_reduced.py --arco era5_arco_sh_vo_d_2018110100.npz --levels all --csv m3_all.csv
```

Expected (recorded 7 Oct 2026):

| Test | Criterion (fixed before the run) | Result |
|---|---|---|
| Calibration of the ARCO convention | simple factors; packing-level residual poleward of 30° | passed (our convention) |
| **M2a**: u, v from the T639 spectra vs ECMWF's u, v | reported | RMS = q/√12 and max = q/2 on all 137 levels (pure 16-bit packing); regression slope 1.0000000 ± 1e-7 |
| **M4**: our analysis of gridded vo, d vs the spectra | ratio ≤ 1.05 to the projected packing noise, poleward of 30° (amended before its first run: the original 1e-4 was below the data's own precision) | 274/274 field-levels PASS |
| **M2**: u, v from the *gridded* vo, d vs ECMWF's u, v | RMS ≤ q and max ≤ 4q on every level | **FAILED** at level 135, u: max 4.73e-3 (4.8q). Attributed to a tropical n = 639 component present in ECMWF's gridded vo, d but not in its u, v. Recorded as a failure; the criterion was not changed |
| **M3**: reduced-grid path | relative RMS ≤ 1e-6 on O640 (moved from O320, which cannot hold T639, before the run) | 1.3e-15 to 1.4e-14 on O640; N320 reported, ≤ 1.0e-6 |
| **M7** (from step 2, at scale): six MPI layouts | bitwise identical | identical after one partition fix |

## Step 4. XNR1K level 74 and the cos φ diagnosis

Inputs (read only), time step 000000, 70,227,474,648 bytes each, in `$X/{vo,d,u,v}/`:
`vo_138_h3f7-ndjf_000000.ml-dgg.O8000.global.grib`, `d_155_…`, and the archived (corrupted) `u_131_…`, `v_132_…`,
which serve as GRIB templates.

```bash
export W=$PWD X=/path/to/XNR1K/ml-dgg
bash scripts/run_L074.sh 32               # inside a one-node allocation; writes $W/run_L074
U=$X/u/u_131_h3f7-ndjf_000000.ml-dgg.O8000.global.grib; V=$X/v/v_132_h3f7-ndjf_000000.ml-dgg.O8000.global.grib
RU=$(ls $W/run_L074/u_*recovered*.grib); RV=$(ls $W/run_L074/v_*recovered*.grib)
srun -n 1 python3 analysis/check_recovered.py --rec-u $RU --rec-v $RV --tpl-u $U --tpl-v $V --level 74
srun -n 1 python3 analysis/cosphi_check.py    --rec-u $RU --rec-v $RV --tpl-u $U --tpl-v $V --level 74
```

Expected (recorded 7–9 Oct 2026):

- Run time about 360–390 s per level on 32 cores.
- `check_recovered.py`: M8 PASS for u and v (all checked header keys identical to the archive); 256,288,000 finite
  16-bit values.
- `cosphi_check.py`: the archived winds equal the recovered winds × cos φ. The residual r = archived −
  recovered·cos φ has an RMS 1.28 (u) and 1.04 (v) times the pure-packing expectation, and 99.992 % (u) and
  100 % (v) of points lie within twice the packing bound. In every 5° band the slope is closer to cos φ than to
  cos² φ; near the equator the margin is small, so there the point-wise residual is the real test.
- Reproducibility: level 74 was recomputed in a second Python environment and with a second MPI transport;
  `analysis/compare_runs.py` reported all three runs **byte-identical**.

## Step 5. All 137 levels of step 000000

```bash
cd $W
export W VENV=/path/to/venv               # the venv must not be on Lustre (INSTALL.md)
sbatch scripts/run_000000.sbatch          # edit #SBATCH -A and the input path X first
bash scripts/status_000000.sh <jobid>     # progress, read only
```

Expected (recorded 9 Oct 2026): `run_000000/` holds the merged files
`u_131_…recovered.grib` and `v_132_…recovered.grib`, 70,227,474,648 bytes each (the size of the archived files),
16-bit GRIB2 on the native O8000 grid. On Andes, 89 levels plus the merge took 2 h 32 min on 4 nodes (the first
48 levels had been computed in an earlier job). The md5 checksums of the merged files on Andes were
u `6242e73ce67ed4e87997e477c10ac2cd`, v `78486056b1bc5d3c6736bd1af4de7c55`. Identical checksums are expected only
with the same libraries; on another system, compare with `analysis/compare_runs.py --tolerance-steps N` (differences in packing steps).

## Step 6. Post-run checks

```bash
bash scripts/postrun_000000.sh            # about 1 h; reads about 350 GB; logs to run_000000/checks/
```

It expects the earlier run `run_000000_attempt1/` for its step 5; without it, that comparison reports missing
files and the other steps still run.

Expected (recorded 9 Oct 2026):

- M8 on all 137 levels, u and v.
- Each merged file is byte-for-byte the concatenation of its 137 per-level parts.
- First attempt against the clean run: 79 messages of levels 1–41 byte-identical (different Python
  environment, MPI transport, nodes and jobs; three messages were absent from the first attempt).
- cos φ band slopes match the band's cos φ to ≤ 4.8e-4 on levels 1–48 (80°S–80°N); levels 49–137 were checked
  without exception flags but not reviewed one by one.
- Residual beyond packing, all levels: u 0.17–2.29 mm/s (median 0.51), v 0.15–1.78 mm/s (median 0.48);
  near-Gaussian and nearly flat in latitude.
- Pre-registered Spearman test of the single-precision hypothesis (ρ > 0.5 supports, |ρ| < 0.2 counts
  against): ρ = +0.35 (u, inconclusive) and +0.17 (v, counts against). The cause of the residual remains open.

## Cost of the full archive (estimate, not run)

961 output times × 137 levels = 131,657 level-steps. At 359–388 s per level on 32 cores, that is about
420,000–454,000 core-hours, and about 135 TB of output (140.5 GB per time step).
