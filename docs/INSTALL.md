# Installation

The code is plain Python on top of NumPy. What else you need depends on what you want to run:

| Task | Needs |
|---|---|
| Tests that need no MPI and no GRIB (step 1 of [REPRODUCE.md](REPRODUCE.md)) | Python ≥ 3.10, numpy, scipy, matplotlib |
| Recovery on GRIB input, MPI self-tests | + mpi4py **built against the cluster's MPI**, ecCodes Python bindings (`eccodes`), netCDF4 |
| ERA5 reference data | + `cdsapi` and a Copernicus CDS account (request A, B, C); + `xarray zarr gcsfs` and internet access (ARCO-ERA5) |
| Coastlines on maps (optional) | + cartopy and its Natural Earth data |

`requirements.txt` lists all of them, with notes; it is the file that was used during the work.

## Versions used for the published results

Recorded on OLCF Andes for the run of step 000000 (9 Oct 2026). The full `pip freeze` is
[provenance/venv_nfs_freeze.txt](provenance/venv_nfs_freeze.txt); to reproduce those versions, first build mpi4py 4.1.2 against your MPI
(`MPICC="mpicc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py==4.1.2`, section 2), then run
`pip install -r docs/provenance/venv_nfs_freeze.txt`, so pip finds mpi4py already installed and does not fetch a binary wheel.

| Component | Version |
|---|---|
| Python | 3.12.9 (virtual environment on NFS, not on Lustre) |
| numpy, scipy | 2.5.3, 1.18.1 |
| ecCodes | `eccodes` 2.49.0 with the binary wheel `eccodeslib` 2.49.0.30 (library 2.49.0) |
| netCDF4, matplotlib | 1.7.4, 3.11.2 |
| MPI | Open MPI 4.0.4 (Andes modules `gcc/9.3.0`, `openmpi/4.0.4`, `miniforge3/24.11.3-2`) |
| mpi4py | 4.1.2, built from source against that Open MPI |

The tests in step 1 of REPRODUCE.md were rerun on 25 Nov 2026 with Python 3.13.18, numpy 2.5.3, scipy 1.18.1 and
matplotlib 3.11.2 (Linux, no MPI), and all passed.

## 1. Workstation or laptop: tests only

```bash
git clone https://github.com/anantharajvg/xnr1k-vod2uv-recovery.git
cd xnr1k-vod2uv-recovery
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib
```

Then run step 1 of [REPRODUCE.md](REPRODUCE.md). Without mpi4py the code runs serially or with thread stand-ins
for MPI ranks (`recover/comm.py`), which is how the parallel layouts are tested.

To also run the MPI self-tests on a workstation, install an MPI library first (for example `brew install open-mpi`
on macOS, or `apt install libopenmpi-dev openmpi-bin` on Debian/Ubuntu), then build mpi4py as in section 2,
using that library's `mpicc`. We did not test this on a workstation.

## 2. HPC cluster

The recovery of one XNR1K level decodes two 256-million-point fields and needs a whole node: the batch script
uses one group of 32 MPI ranks per node (Andes nodes had 32 cores).

```bash
# 1. compiler and MPI from the system (names differ per site)
module load gcc openmpi            # Andes; on other systems the site's GNU + MPI modules

# 2. the virtual environment: put it on a home or project file system (NFS), NOT on a Lustre scratch area
python3 -m venv $HOME/venvs/vod2uv
source $HOME/venvs/vod2uv/bin/activate
pip install --upgrade pip

# 3. mpi4py compiled against the system MPI (never a wheel, never a conda MPI)
which mpicc                        # must be the module's mpicc
MPICC="mpicc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

# 4. everything else
pip install numpy scipy netCDF4 eccodes matplotlib
pip freeze > venv_freeze.txt      # keep it with your results
```

On HPE Cray systems the compiler wrapper is `cc`, so step 3 becomes `MPICC="cc -shared" ...` after
`module load PrgEnv-gnu cray-python` (see the notes in `requirements.txt`). We did not run the recovery on a
Cray system.

### Why these rules

- **Virtual environment not on Lustre.** On Andes, two batch jobs died with bus errors (SIGBUS) while
  ecCodes and the environment were mapped from Lustre. A later bus error happened with the environment on NFS,
  so Lustre was not the whole explanation, but `scripts/run_000000.sbatch` refuses a Lustre environment unless
  `ALLOW_LUSTRE_VENV=1` is set, and all published results come from the NFS environment.
- **mpi4py built against the system MPI.** A wheel or a conda MPI does not use the cluster interconnect and
  launcher correctly. Check the linkage as in section 3.
- **Open MPI without UCX.** On Andes three jobs ended with bus errors inside UCX (the MPI transport layer). The batch
  script therefore sets `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=^ucx` by default
  (`MPI_STACK=ucx` restores UCX). Level 74 was byte-identical with and without UCX, at the same speed. The cause of
  the bus errors was not established. With MPICH or Cray MPICH these variables have no effect.

## 3. Verify the installation

MPI must be started by the launcher, so run these inside an allocation:

```bash
srun -n 1 python3 -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
ldd $(python3 -c "import mpi4py, glob, os; print(glob.glob(os.path.join(os.path.dirname(mpi4py.__file__), 'MPI*.so'))[0])") | grep -i mpi
python3 -c "import eccodes, findlibs; print(eccodes.codes_get_api_version(), findlibs.find('eccodes'))"
python3 -c "import numpy, netCDF4; print(numpy.__version__, netCDF4.__version__)"
```

Expected: the MPI vendor and library of your site's module (Andes: `('Open MPI', (4, 0, 4))`); the mpi4py extension
links that library's `libmpi`; ecCodes loads from inside your environment (Andes: `eccodeslib/lib64/libeccodes.so`
in the venv). `mpi4py.get_config()` returned `{}` on Andes and is not informative.

Then run steps 1–2 of [REPRODUCE.md](REPRODUCE.md).

## 4. Runtime settings

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1     # one MPI rank per core; the kernels are single-threaded NumPy
```

Inside an interactive `srun --pty` shell, nested `srun` calls need `export SLURM_OVERLAP=1 SLURM_CPU_BIND=none`
(the scripts set these).

## 5. ERA5 access (only for step 3 of REPRODUCE.md)

- **Copernicus CDS** (`era5/download_era5.py`): create an account, accept the licence of the dataset
  "reanalysis-era5-complete", and write `~/.cdsapirc`:

  ```
  url: https://cds.climate.copernicus.eu/api
  key: <your personal access token>
  ```

  Model-level ERA5 is on tape at ECMWF; requests can queue for hours.
- **ARCO-ERA5** (`era5/fetch_arco.py`): anonymous access to the public Google Cloud bucket; needs
  `pip install xarray zarr gcsfs` and a machine with internet access (on Andes, a login node).
