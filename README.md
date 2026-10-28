# MesoChi

MesoChi (Microstructure Informed Quantitative Susceptibility Mapping) provides
QSM and μQSM reconstruction with L2 and TV regularization. It includes the
phantom and ex vivo experiments used in this project.

## Install

Use Python 3.11 and install from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[nifti]"
```

The `nifti` extra installs NiBabel for NIfTI input and export. Install `-e .`
instead when using MATLAB inputs and NumPy/MATLAB outputs only. Select this
`.venv` interpreter in VS Code before running scripts.

For NVIDIA GPU reconstruction, install the CuPy package matching your CUDA
version in the same environment. For this workstation's CUDA 13 installation:

```powershell
python -m pip install "cupy-cuda13x==14.2.0"
```

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for other CUDA versions. The public script automatically selects an available
GPU or reports CPU selection. Use `--use-gpu` or `--no-use-gpu` to override it.
FBI preparation runs on CPU; the reconstruction operators and solvers run on
the selected device.

## Reconstruct your data

Prepare the inputs described in [the input/output guide](docs/exvivo_inputs.md),
then run from the project folder:

```powershell
python Micro_QSM_Recon.py --input-dir C:\path\to\inputs --output-dir C:\path\to\outputs --input-format mat
python Micro_QSM_Recon.py --input-dir C:\path\to\inputs --output-dir C:\path\to\outputs --input-format nifti
```

Alternatively, edit `INPUT_DIR`, `OUTPUT_DIR`, and `INPUT_FORMAT` near the top
of `Micro_QSM_Recon.py`, then run the file in VS Code. Use `--help` to see all
settings. The script generates QSM L2, μQSM L2, QSM TV, and μQSM TV maps,
comparison figures, and a run record. A weight map is optional; omitting it
uses uniform weights. NIfTI inputs provide the export reference automatically;
MATLAB inputs need a matching reference image for NIfTI export.

## Reproduce the experiments

```powershell
python examples/generate_phantoms.py
python examples/reconstruct.py
python examples/exvivo_reconstruction.py --no-display
python examples/exvivo_lcurve.py --no-display
```

Edit the settings at the top of each script. Phantom reconstruction uses the
five bundled archives in `data/phantoms/`; generation writes to
`data/generated/`. The default `1024³` generation needs substantial memory and
runtime. The ex vivo example expects the six prepared MATLAB files in local
`data/exvivo/`; those files are not published. The L-curve example estimates
only the QSM and μQSM **L2** parameters, which must be reviewed and transferred
manually into the reconstruction settings. TV reconstruction remains available.

The examples resolve default locations from their own files, so they can run
from another working directory after installing the package. The ex vivo
L-curve example currently enables GPU; use `--no-use-gpu` to run it on CPU.

## Organization

- `Micro_QSM_Recon.py`: public dataset reconstruction script.
- `examples/`: scripts for phantom and ex vivo experiments.
- `src/mesochi/`: shared operators, FBI, reconstruction, plotting, and L-curve code.
- `data/phantoms/`: supplied phantom archives, formerly `data/example/`.
- `data/exvivo/`: locally supplied ex vivo MATLAB data, ignored by Git.
- `Sanity_Check/`: local diagnostics, tests, and fixtures; excluded from Git and distributions.
- `results/`: generated outputs, ignored by Git.

Original hash manifests and validation reports in `docs/` are historical;
their old `data/example/` paths refer to the unchanged files now under
`data/phantoms/`. See [simulation notes](docs/simulation_notes.md) for the
phantom model and the [input/output guide](docs/exvivo_inputs.md) for the
reconstruction conventions and limitations.
