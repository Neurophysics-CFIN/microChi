# MesoChi

MesoChi is an open-source Python toolbox for microstructure-informed quantitative susceptibility mapping (μQSM). 
It combines prepared local phase data with diffusion MRI. It applys fiber-ball imaging (FBI)
to estimate scatter matrices that inform the mesoscopic contribution to the
μQSM forward model. The toolbox reconstructs conventional QSM and μQSM maps
using L2 and total variation (TV) regularization. It also includes scripts for
the phantom and ex vivo experiments.

## Install

Install Miniconda, Miniforge, or another Conda distribution, which requires a minimum Conda version of 26.1.1.
Then open a terminal in the project folder containing `environment.yml` and run:

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

GPU support for acceleration is optional. For an NVIDIA GPU, install the CuPy package matching
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

## Reconstruction with your own data

Prepare the inputs described in [the reconstruction tutorial](docs/reconstruction_tutorial.md),
then run from the project folder:

```powershell
python Micro_QSM_Recon.py --input-dir "C:\path\to\inputs" --output-dir "C:\path\to\outputs" --input-format mat --no-display
python Micro_QSM_Recon.py --input-dir "C:\path\to\inputs" --output-dir "C:\path\to\outputs" --input-format nifti --no-display
```

Use `--help` to see all available settings.

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

## Examples

The scripts in `examples/` demonstrate QSM and μQSM reconstruction using
synthetic phantoms and prepared ex vivo MRI data.

### Phantom examples

The cylindrical, spherical, and composite phantom examples use the five
archives bundled in `data/phantoms/`. From the project folder, run:

```powershell
python examples/phantom_reconstruction.py
```

To generate new phantom data, run `python examples/generate_phantoms.py`.
The default `1024³` generation requires substantial memory and runtime. To
reconstruct newly generated data, change `DATA_DIR` in
`examples/phantom_reconstruction.py` to `data/generated/`.

### Ex vivo example

The ex vivo example combines prepared local phase and diffusion MRI data to
reconstruct conventional QSM and μQSM maps with L2-Tikhonov and L1-TV regularization.

Download [Ex-vivo_example_data.zip](https://github.com/Neurophysics-CFIN/MesoChi/releases/download/ex-vivo_example-data/Ex-vivo_example_data.zip)
from the **Assets** section of the [first GitHub release](https://github.com/Neurophysics-CFIN/MesoChi/releases/tag/ex-vivo_example-data).
Extract the six MATLAB files directly into `data/exvivo/`: `DWI_FBI.mat`,
`Phase_map.mat`, `mask_phase.mat`, `mask_wm.mat`, `mask_csf.mat`, and
`meta_info.mat`. With the MesoChi environment active, run from the project folder:

```powershell
python examples/exvivo_reconstruction.py --input-dir data/exvivo --output-dir results/exvivo --no-display
```

To estimate the QSM and μQSM **L2** regularization parameters, run:

```powershell
python examples/exvivo_lcurve.py --input-dir data/exvivo --output-dir results/exvivo/l_curve --no-display --no-use-gpu
```

Review the estimated L2 values and transfer them manually into the reconstruction
settings. The ex vivo L-curve example enables GPU by default (--use-gpu) and also can use `--no-use-gpu` to run it on the CPU.
Input formats and preparation requirements are described in [the reconstruction tutorial](docs/reconstruction_tutorial.md).

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
