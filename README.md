# MesoChi

MesoChi is an open-source Python toolbox for microstructure-informed quantitative susceptibility mapping (μQSM). 
It combines prepared local phase data with diffusion MRI. It applys fiber-ball imaging (FBI)
to estimate scatter matrices that inform the mesoscopic contribution to the
μQSM forward model. The toolbox reconstructs conventional QSM and μQSM maps
using L2 and total variation (TV) regularization. It also includes scripts for
the phantom and ex vivo experiments.

## Install

Install Miniconda, Miniforge, or another Conda distribution. Then open a
terminal in the project folder containing `environment.yml` and run:

```powershell
conda env create -f environment.yml
conda activate mesochi
python -c "import mesochi, nibabel; print(mesochi.__file__)"
```

The environment uses Python 3.11, includes NiBabel for NIfTI input and output,
and installs the repository in editable mode. If you use VS Code, select the
`mesochi` interpreter before running the scripts. If you move the repository,
reactivate the environment and run `python -m pip install -e ".[nifti]"` from
its new location.

GPU support is optional. For an NVIDIA GPU, install the CuPy package matching
your CUDA version in the activated `mesochi` environment. For example, with a
compatible NVIDIA driver and CUDA 13 toolkit:

```powershell
python -m pip install "cupy-cuda13x==14.2.0"
```

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for other CUDA versions. The reconstruction script automatically selects an
available GPU or falls back to the CPU. Use `--use-gpu` or `--no-use-gpu` to
override this selection. FBI preparation runs on the CPU; reconstruction
operators and solvers run on the selected device.

## Verify the installation

After installing and activating the environment, run the CPU numerical tests
from the project directory:

```powershell
python -m unittest discover -s tests -v
python tests/smoke_reconstruction.py
```

The numerical tests use a small reference fixture and check the operators,
simulation helpers, and L-curve routines. The smoke check reconstructs QSM on
the bundled 32³ cylindrical phantom on CPU. It expects a finite 32³ result and
a relative field residual below 1, and saves no output. Neither check replaces
validation against experimental data or reproduces the full experiment.

## Reconstruct your data

Prepare the inputs described in [the reconstruction tutorial](docs/reconstruction_tutorial.md),
then run from the project folder:

```powershell
python Micro_QSM_Recon.py --input-dir "C:\path\to\inputs" --output-dir "C:\path\to\outputs" --input-format mat --no-display
python Micro_QSM_Recon.py --input-dir "C:\path\to\inputs" --output-dir "C:\path\to\outputs" --input-format nifti --no-display
```

Alternatively, edit `INPUT_DIR`, `OUTPUT_DIR`, and `INPUT_FORMAT` near the top
of `Micro_QSM_Recon.py`, then run the script in VS Code. Use `--help` to see all
available settings.

The script reconstructs QSM L2, μQSM L2, QSM TV, and μQSM TV maps and produces
comparison figures and a run record. Maps are cropped to the input grid and
CSF-referenced. By default, they are saved in NumPy and MATLAB formats; NIfTI
export is also available. A weight map is optional; uniform weights are used
when it is omitted. For NIfTI export, NIfTI inputs provide the reference image
automatically, while MATLAB inputs require a matching reference image. Use
`--no-display` to save figures without opening windows.

Use `--regularization l2` or `--regularization tv` to reconstruct only the
selected method, or `--regularization both` (the default) to run all four maps.
You can also edit `REGULARIZATION` near the top of the script. Exported map
variables and NIfTI filenames use `Chi_QSM_L2`, `Chi_muQSM_L2`, `Chi_QSM_TV`,
and `Chi_muQSM_TV`, according to the selected method.

## Reproduce the experiments

```powershell
python examples/generate_phantoms.py
python examples/phantom_reconstruction.py
python examples/exvivo_reconstruction.py --input-dir "C:\path\to\exvivo_inputs" --output-dir "C:\path\to\outputs" --no-display
python examples/exvivo_lcurve.py --input-dir "C:\path\to\exvivo_inputs" --output-dir "C:\path\to\outputs" --no-display --no-use-gpu
```

Edit the settings at the top of each script and replace the example paths with
your own. Phantom reconstruction uses the five bundled archives in
`data/phantoms/`. Generating the default `1024³` phantoms requires substantial
memory and runtime.

The ex vivo examples require six prepared MATLAB files. The ex vivo dataset is
not included in the repository. Dataset can be downloaded [here](https://github.com/Neurophysics-CFIN/MesoChi/releases/tag/ex-vivo_example-data). Prepare the
inputs in the format described in the reconstruction tutorial.

The L-curve example estimates the QSM and μQSM **L2** parameters. Review the
estimated values and transfer them manually into the reconstruction settings.
TV reconstruction is also available.

The examples resolve default paths relative to their own files, so they can run
from another working directory after the package is installed. The ex vivo
L-curve example enables GPU by default; use `--no-use-gpu` to run it on the CPU.

## Citation

If you use MesoChi in your research, please cite the associated publication:

{citation}

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
