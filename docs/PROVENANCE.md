# Provenance

## Which code produced the results

The files in this repository are byte-for-byte copies of the files in the project's working folder, taken on
25 Nov 2026. None of the code has been edited for publication. The new files are `README.md`, `LICENSE`,
`.gitignore`, `docs/INSTALL.md`, `docs/CODE.md`, `docs/REPRODUCE.md`, this file and the copied manifests in
`docs/provenance/`; `docs/vod2uv_mpi.md` is the
development note `vod2uv_mpi_README.md`, renamed and otherwise unchanged.

The project's work log and handoff notes recorded the md5 checksums (first 8 hex digits) of the files below when
the results were produced: the level-74 run (7–8 Oct 2026) and the run of all 137 levels of step 000000 (Slurm
job 3503423 on OLCF Andes, 8–9 Oct 2026). Every recorded checksum matches the published file.

| File | md5 | Against the records |
|---|---|---|
| `analysis/check_recovered.py` | `f2f32cadca23632597b2ea8db99e7ab8` | matches `f2f32cad…` |
| `analysis/compare_runs.py` | `d5bdb06db58b34a971cf41f180f1f63d` | matches `d5bdb06d…` |
| `analysis/cosphi_budget.py` | `a3dbdb9f10ebe25fbf2078f3fc9ba678` | not in the project records |
| `analysis/cosphi_check.py` | `33b24ca80d5b4a46c4e58e2e2951e61e` | matches `33b24ca8…` |
| `analysis/m2_diag.py` | `2cdbe287290b2ffb14a5ef68d918c712` | not in the project records |
| `analysis/m2_era5.py` | `ff04bc29c486377313516c0eca895385` | matches `ff04bc29…` |
| `analysis/m3_reduced.py` | `c32ba8c547ebf4db9712ef1c2f716c6e` | matches `c32ba8c5…` |
| `analysis/packing_steps.py` | `48fd5c118138e4b6a0d1c874a6616d90` | not in the project records |
| `analysis/residual_diag.py` | `1da817c2e60c1ec8acc6d790495591a3` | not in the project records |
| `analysis/zonal_ratio_check.py` | `e31dc3abde1f11b4c83d2b8cb4e8faaf` | not in the project records |
| `docs/vod2uv_mpi.md` | `7186c5e9f005c974371880a9b2fcbdb2` | not in the project records |
| `era5/download_era5.py` | `f13e4fc720c36956e6f32bf453ae39fe` | not in the project records |
| `era5/fetch_arco.py` | `baebb53b1473de227da61db2e4196de3` | not in the project records |
| `recover/README.md` | `a7995c26834926e11cc25de984cdfa23` | not in the project records |
| `recover/comm.py` | `fac915f76ed9a68d92271c5127559b83` | not in the project records |
| `recover/compare.py` | `9d32dfd6add1f28a710f4697baac5661` | not in the project records |
| `recover/core.py` | `893d2f98f9abfb6992340f316a6c2d5a` | matches `893d2f98…` |
| `recover/gribio.py` | `f3e5e2d6d1ee847de12a7e4fbec20233` | matches `f3e5e2d6…` |
| `recover/plots.py` | `fa2b65f39be27a3f2191421641d036ac` | not in the project records |
| `recover/recover_uv.py` | `aa5e7d0896acf65ef26838d91953e816` | matches `aa5e7d08…` |
| `recover/serial.py` | `bf8ed269193f9d89931456d760237d57` | not in the project records |
| `recover/tests/grib_selftest.py` | `decfe6b52625a666955842fb72b7cc3e` | not in the project records |
| `recover/tests/selftest_grib.sh` | `259ca89c1fdb3aa2f1ed470491fd80e9` | not in the project records |
| `recover/tests/test_m1_analytic.py` | `a6639f2d5afd60506268cc1c7033fc05` | not in the project records |
| `recover/tests/test_m7_driver.py` | `e7e3d060c02aa688d17dcdcbee20a1c8` | not in the project records |
| `recover/tests/test_outputs.py` | `bb73cfa68b8793e5a56e41c5b9cc8967` | not in the project records |
| `recover/tests/test_plots.py` | `17b8e784015828a95cbac8b99ae3a986` | not in the project records |
| `recover/tests/test_unit.py` | `b84f013368f3d4c247d5533921a680dd` | not in the project records |
| `requirements.txt` | `3dbb6f74b5d99cd75ba282af4b57296f` | not in the project records |
| `scripts/check_parts.sh` | `bbfefc510b80c561c2bfcdf3415efaac` | matches `bbfefc51…` |
| `scripts/postrun_000000.sh` | `fade3e35e6506710ba5374506c350568` | not in the project records |
| `scripts/provenance.sh` | `0c8c063a0d83c1ac1a396fa7fb0a2acd` | not in the project records |
| `scripts/run_000000.sbatch` | `a0fa3097a25e9889473a171f502ae403` | matches `a0fa3097…` |
| `scripts/run_L074.sh` | `39de1ad6f79dbec5d49d60fddb3ed364` | matches `39de1ad6…` |
| `scripts/status_000000.sh` | `b89807f3de3cc9fad26827a02a730cb2` | not in the project records |
| `tests/bench_T7999.py` | `19942073721c66000000803e51b9f67f` | not in the project records |
| `tests/check_fd.py` | `6cb8551018b5e900ab608b15e58bf08d` | not in the project records |
| `tests/check_output.py` | `11260e3b999553dacb74edfa5302743d` | not in the project records |
| `tests/fakempi.py` | `cb0762b21715d109a64e8442642aa3de` | not in the project records |
| `tests/make_testfile.py` | `5000e3ead0fce1f80fdf886f5b5c449f` | not in the project records |
| `tests/selftest_mpi.sh` | `6911c4669436ccf523bf0bafce11f7a0` | not in the project records |
| `tests/test_core.py` | `dbd666af68cc17bb959eb69e202a5d13` | not in the project records |
| `tests/test_ortho_T7999.py` | `9674ef8f1b43b96f1cdb7c48a84578c5` | not in the project records |
| `tests/test_pipeline.py` | `a0a115b4dad6f06b2363c865994cbd20` | not in the project records |
| `tests/test_random_scipy.py` | `8b3da912a75e5d4b3ce0c6d9dc3ef6aa` | not in the project records |
| `vod2uv_mpi.py` | `04d73a76d11976949a00a65c716a3139` | matches `04d73a76…` |

## Run-time manifests (`docs/provenance/`)

`scripts/provenance.sh` wrote these on OLCF Andes; they were copied from Orion on 25 Nov 2026 and are unedited.

| File | md5 | What it records |
|---|---|---|
| `docs/provenance/provenance_20261007T234405Z.txt` | `a071fafadf043892be27ff3b1676226d` | After the first level-74 run (7 Oct): modules, Python packages, the MPI library mpi4py links to, checksums of code, ERA5 inputs and outputs |
| `docs/provenance/provenance_20261009T100743Z.txt` | `977a1ff241336ae1f61235e3a687e814` | After all 137 levels of step 000000 (9 Oct): the same, plus every per-level part and the merged outputs |
| `docs/provenance/venv_nfs_freeze.txt` | `6c1d2ecbdae851d64e0ba7d42ad8dff0` | `pip freeze` of the Python environment used for step 000000 |

Checked on 25 Nov 2026:

- **9 Oct manifest:** all 45 published code and test files that it lists match it exactly (the README differs
  because it is new). `scripts/run_000000.sbatch` is not listed in it; its checksum `a0fa3097…` is the one the work
  log records for job 3503423. The merged output checksums in the manifest are those given below. The manifest's
  own entry for `venv_nfs_freeze.txt` equals the checksum of the copy here, so that copy is intact.
- **7 Oct manifest:** three files were at earlier versions on 7 Oct and changed before step 000000 was run:
  `recover/recover_uv.py` (`c797ebd3…` on 7 Oct, `aa5e7d08…` published), `analysis/check_recovered.py`
  (`8dc765c7…`, `f2f32cad…`) and `scripts/run_L074.sh` (`8101b74b…`, `39de1ad6…`). The other files it lists match.
  Level 74 was later recomputed in the NFS environment and without UCX, and the GRIB output was byte-identical to
  the 7 Oct run (work log, 8 Oct); that the reruns used the published `recover_uv.py` is inferred from the dates,
  not recorded.
- The manifests contain OLCF system paths, including the lead author's OLCF user name in one Python path.

Environment for step 000000 (from the 9 Oct manifest and the freeze): Python 3.12.9; numpy 2.5.3; scipy 1.18.1;
mpi4py 4.1.2 linked to Open MPI 4.0.4 (`/sw/andes/spack-envs/...`); netCDF4 1.7.4; eccodes 2.49.0 with
eccodeslib 2.49.0.30; matplotlib 3.11.2; modules `gcc/9.3.0`, `openmpi/4.0.4`, `miniforge3/24.11.3-2`.

## Recorded checksums of the outputs (step 000000, OLCF Andes, 9 Oct 2026)

| File | Size (bytes) | md5 |
|---|---|---|
| merged recovered u, all 137 levels | 70,227,474,648 | `6242e73ce67ed4e87997e477c10ac2cd` |
| merged recovered v, all 137 levels | 70,227,474,648 | `78486056b1bc5d3c6736bd1af4de7c55` |
