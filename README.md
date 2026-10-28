# MesoChi

MesoChi (Microstructure Informed Quantitative Susceptibility Mapping) combines
prepared local phase data with diffusion MRI. It uses fiber-ball imaging (FBI)
to estimate scatter matrices that inform the mesoscopic contribution to the
μQSM forward model. The toolbox reconstructs conventional QSM and μQSM maps
with L2 and TV regularization, and includes this project's phantom and ex vivo
experiments.

## Install

Install Miniconda, Miniforge, or another Conda distribution. Open a terminal
in the project folder containing `environment.yml` and run:

```powershell
conda env create -f environment.yml
conda activate mesochi
python -c "import mesochi, nibabel; print(mesochi.__file__)"
```

The environment uses Python 3.11, includes NiBabel for NIfTI input and output,
and installs this checkout in editable mode. Select the `mesochi` interpreter in
VS Code before running scripts. If you move the checkout, reactivate the
environment and run `python -m pip install -e ".[nifti]"` from its new location.

GPU support is optional. For an NVIDIA GPU, install the CuPy package matching
your CUDA version in the activated `mesochi` environment. For example, with a
compatible NVIDIA driver and CUDA 13 toolkit:

```powershell
python -m pip install "cupy-cuda13x==14.2.0"
```

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for other CUDA versions. The public script automatically selects an available
GPU or reports CPU selection. Use `--use-gpu` or `--no-use-gpu` to override it.
FBI preparation runs on CPU; the reconstruction operators and solvers run on
the selected device.

## Reconstruct your data

Prepare the inputs described in [the reconstruction tutorial](docs/reconstruction_tutorial.md),
then run from the project folder:

```powershell
python Micro_QSM_Recon.py --input-dir "C:\path\to\inputs" --output-dir "C:\path\to\outputs" --input-format mat --no-display
python Micro_QSM_Recon.py --input-dir "C:\path\to\inputs" --output-dir "C:\path\to\outputs" --input-format nifti --no-display
```

Alternatively, edit `INPUT_DIR`, `OUTPUT_DIR`, and `INPUT_FORMAT` near the top
of `Micro_QSM_Recon.py`, then run the file in VS Code. Use `--help` to see all
settings. The script generates QSM L2, μQSM L2, QSM TV, and μQSM TV maps,
comparison figures, and a run record. The maps are cropped to the input grid,
CSF-referenced, and saved in NumPy and MATLAB archives. A weight map is
optional; omitting it uses uniform weights. NIfTI inputs provide the export
reference automatically; MATLAB inputs need a matching reference image for
NIfTI export. `--no-display` saves figures without opening windows.

## Reproduce the experiments

```powershell
python examples/generate_phantoms.py
python examples/phantom_reconstruction.py
python examples/exvivo_reconstruction.py --input-dir "C:\path\to\exvivo_inputs" --output-dir "C:\path\to\outputs" --no-display
python examples/exvivo_lcurve.py --input-dir "C:\path\to\exvivo_inputs" --output-dir "C:\path\to\outputs" --no-display --no-use-gpu
```

Edit the settings at the top of each script and replace the example paths with
your own. Phantom reconstruction uses the five bundled archives in
`data/phantoms/`. The default `1024³` generation needs substantial memory and
runtime. The ex vivo examples require six prepared MATLAB files. The ex vivo
dataset is not included in the repository, and a public download is not
currently provided; supply prepared data in the format described in the
tutorial. The L-curve example estimates only the QSM and μQSM
**L2** parameters, which must be reviewed and transferred manually into the
reconstruction settings. TV reconstruction remains available.

The examples resolve default locations from their own files, so they can run
from another working directory after installing the package. The ex vivo
L-curve example currently enables GPU; use `--no-use-gpu` to run it on CPU.

## Citation

When using this software, cite MesoChi, its version (0.1.0), the repository
URL, and the date you accessed it. Bibliographic details for the associated
paper are not yet provided here; add them once the publication details are
available.

## License

MesoChi is distributed under the [MIT License](LICENSE). The ex vivo dataset
is not included and is not covered by this software license.

## Organization

- `Micro_QSM_Recon.py`: public dataset reconstruction script.
- `examples/`: scripts for phantom and ex vivo experiments.
- `src/mesochi/`: shared operators, FBI, reconstruction, plotting, and L-curve code.
- `data/phantoms/`: supplied phantom archives.

See [simulation notes](docs/simulation_notes.md) for the phantom model and the
[reconstruction tutorial](docs/reconstruction_tutorial.md) for detailed input,
output, and reconstruction instructions.
