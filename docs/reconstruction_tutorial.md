# MesoChi reconstruction tutorial

The public `Micro_QSM_Recon.py` script reconstructs QSM and μQSM maps from
prepared, registered diffusion and local phase data. Choose one input directory
and set `INPUT_FORMAT` to `mat` or `nifti`. Filenames are case-sensitive.

All images must already be registered to the same voxel grid and array ordering.
Perform registration, phase unwrapping, and background-field removal before
running the toolbox. `phase` is a prepared local phase volume in radians, not
raw echo images.

## 1. Create and activate the environment

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or another
Conda distribution, then open a terminal in the project root: the folder
containing `README.md`, `environment.yml`, and `Micro_QSM_Recon.py`. For
example, in Windows PowerShell, replace the path below with your own:

```powershell
cd "C:\path\to\MesoChi"
conda env create -f environment.yml
conda activate mesochi
```

The environment file installs the pinned scientific libraries, NiBabel, and an
editable install of this checkout. If you move the project folder later, run
`python -m pip install -e ".[nifti]"` from its new location while the
`mesochi` environment is active. Run `conda activate mesochi` in each new
terminal before using the toolbox. If PowerShell does not recognize `conda`,
use the terminal supplied with your Conda distribution or initialize Conda for
PowerShell and reopen the terminal.

Confirm which Python and MesoChi installation are active:

```powershell
python -c "import sys, mesochi; print(sys.executable); print(mesochi.__file__)"
python Micro_QSM_Recon.py --help
```

In VS Code, open this project folder and choose the Python interpreter from the
`mesochi` Conda environment using **Python: Select Interpreter**. For the
Interactive window or a notebook, select the same environment as its kernel;
restart an existing kernel after changing environments. Running a script from
the project directory alone does not install the `src/` package.

### Optional NVIDIA GPU

The environment works on CPU without CuPy. The public script selects a CUDA
GPU if compatible CuPy and a usable device are available; otherwise it reports
CPU selection. `--use-gpu` explicitly requires a GPU, while `--no-use-gpu`
forces CPU. FBI preparation and input/output work remain on CPU.

For the validated CUDA 13 setup, activate `mesochi` and, if a compatible CUDA
13 toolkit and driver are installed, run:

```powershell
python -m pip install "cupy-cuda13x==14.2.0"
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name']); print((cp.arange(5)**2).get())"
```

For a different CUDA version or an installation without a local CUDA toolkit,
follow the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for the correct package and runtime dependencies. Do not install multiple
CuPy variants into the same environment.

## 2. Prepare the input folder

Choose either the MATLAB or NIfTI layout below. The toolbox reads one input
folder per run. Supply prepared data using one of the layouts
below. Check the relevant permissions and privacy requirements before sharing
any experimental dataset.

### MATLAB inputs

Place these six files in one folder:

| File | Variables | Shape or purpose |
| --- | --- | --- |
| `DWI_FBI.mat` | `DWI_FBI`, `bval`, `bvec` | Real signal `(X,Y,Z,K)`, b-values `(K,)` in s/mm², gradient directions `(K,3)` |
| `Phase_map.mat` | `phase` | Local phase `(X,Y,Z)` in radians |
| `mask_phase.mat` | `mask_phase` | Nonempty reconstruction and FBI region |
| `mask_wm.mat` | `mask_wm` | White-matter region for the mesoscopic contribution |
| `mask_csf.mat` | `mask_csf` | Nonempty CSF reference region |
| `meta_info.mat` | `Dim`, `B0_dir`, `Voxel_size`, `gammaB0`, `TEs` | Dimensions, field direction, spacing, phase scaling, and echo times |

`Dim` is `(X,Y,Z)`, `Voxel_size` is three positive values in mm, `B0_dir` is a
nonzero direction in image-array axes, `gammaB0` is positive in rad/s per ppm,
and `TEs` contains at least two increasing echo times in milliseconds.
Diffusion b-vectors must be in the same image-array coordinate system.

### NIfTI inputs

NiBabel is included in the provided Conda environment. Put the following in one folder. For each image,
use exactly one of `.nii` or `.nii.gz`:

| File stem | Content |
| --- | --- |
| `DWI_FBI.nii.gz` | 4D diffusion signal `(X,Y,Z,K)` |
| `Phase_map.nii.gz` | 3D prepared local phase |
| `mask_phase.nii.gz` | Reconstruction and FBI mask |
| `mask_wm.nii.gz` | White-matter mask |
| `mask_csf.nii.gz` | CSF reference mask |
| `bval.txt` | K diffusion b-values in s/mm² |
| `bvec.txt` | K gradient directions, arranged as K rows by 3 columns or 3 rows by K columns |
| `meta_info.json` | Field direction, phase scaling, echo times, and b-vector convention |

Images must use millimetre units, valid spatial transforms, orthogonal voxel
axes, matching spatial shapes, and matching affines. They must also be correctly
registered; matching headers alone cannot prove registration. No image flips,
resampling, or registration are performed. The phase image supplies spatial
geometry for NIfTI output unless a matching reference is explicitly supplied.

Example `meta_info.json`:

```json
{
  "B0_dir": [1, 0, 0],
  "gammaB0": 4387.2132,
  "TEs": [1.75, 3.5],
  "bvec_convention": "image_axes"
}
```

These values illustrate the format; use values appropriate for your acquisition.
`B0_dir` must be in image-array axes, `gammaB0` is in rad/s per ppm, and `TEs`
is in milliseconds. Array-axis b-vectors use `"image_axes"`. For standard
FSL/BIDS b-vectors use `"fsl"`; the toolbox converts them to array axes using
the NIfTI affine handedness. The files must correspond to the final registered
image grid. If preprocessing changed the image axes, ensure the b-vectors were
appropriately transformed. See [FSL's b-vector convention](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/faq.html).

### Optional inputs and settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `regularization` / `--regularization` | `both` | Run `l2`, `tv`, or `both`; values are case-insensitive |
| `weight_file` / `--weight-file` | `None` | Optional `.npy` or `.mat` containing `weight`, spatial shape `(X,Y,Z)`, finite and nonnegative |
| `nifti_reference` / `--nifti-reference` | `None` | Matching 3D NIfTI geometry for export; phase image is used for NIfTI inputs |
| `use_gpu` / `--use-gpu` | Automatic in public script | Use GPU if CUDA and CuPy are available; `--no-use-gpu` forces CPU |
| `b_fbi` / `--b-fbi` | First b-value | FBI shell; if first b-value is zero, the only positive shell is selected, or an explicit shell is required when several exist |
| `l_max` / `--l-max` | `6` | Even SH degree, at least 2 |
| `fbi_method` / `--fbi-method` | `corrected` | `corrected` requires `D0`; `exact` requires `Da` |
| `D0` / `--D0` | `2e-3` mm²/s | Corrected-method diffusivity |
| `Da` / `--Da` | `None` | Exact-method diffusivity |
| `qsm_l2_lambda`, `muqsm_l2_lambda` | `8e-3` each | Independent L2 parameters |
| `qsm_tv_lambda`, `muqsm_tv_lambda` | `1e-5` each | Independent TV parameters |
| `cg_max_iter` | `200` | L2 conjugate-gradient limit |
| `tv_max_iter`, `tv_max_cg_iter` | `300`, `10` | TV ADMM budget and inner CG limit |
| `tol`, `rho` | `1e-6`, `1` | Solver tolerance and ADMM parameter |

A weight map multiplies both model and measured data, and padded weights remain
one. To use inverse variance in least squares, supply its square root. Changing
weights changes the data-term scale and can change appropriate regularization.
A MATLAB weight file must contain a variable named `weight`.

Use `python Micro_QSM_Recon.py --help` for argument names, export formats,
progress, and plotting controls. Python workflow calls default to CPU; pass
`use_gpu=True` explicitly to use GPU.

## 3. Run reconstruction

With the `mesochi` environment active and the terminal in the project root,
replace the paths below with your input and output folders. Quote paths that
contain spaces. For MATLAB inputs:

```powershell
python Micro_QSM_Recon.py --input-dir "C:\path\to\matlab_inputs" --output-dir "C:\path\to\output" --input-format mat --no-display
```

For NIfTI inputs:

```powershell
python Micro_QSM_Recon.py --input-dir "C:\path\to\nifti_inputs" --output-dir "C:\path\to\output" --input-format nifti --no-display
```

Select the regularization method with `--regularization`:

| Option | Reconstructions |
| --- | --- |
| `l2` | QSM L2 and μQSM L2 only |
| `tv` | QSM TV and μQSM TV only |
| `both` (default) | All four maps |

For example, to run each method separately:

```powershell
python Micro_QSM_Recon.py --input-dir "C:\path\to\matlab_inputs" --output-dir "C:\path\to\output_l2" --regularization l2 --no-display
python Micro_QSM_Recon.py --input-dir "C:\path\to\matlab_inputs" --output-dir "C:\path\to\output_tv" --regularization tv --no-display
```

Only the selected reconstructions are calculated and exported. Use a new output
folder when changing methods.

`--no-display` saves figures without opening their windows. To skip the figures
entirely, use `--no-plots`. For the MATLAB workflow, append
`--nifti-reference "C:\path\to\reference.nii.gz"` if you want NIfTI maps;
the reference must describe the same registered voxel grid and array ordering.
Without it, NumPy and MATLAB outputs are still saved. For either input format,
append `--weight-file "C:\path\to\weight.mat"` if you have a weight map.
Both arguments are optional.

The four regularization values can be set independently; only the values for
the selected methods are used. For example:

```powershell
python Micro_QSM_Recon.py --input-dir "C:\path\to\matlab_inputs" --output-dir "C:\path\to\output" --input-format mat --muqsm-l2-lambda 0.008 --qsm-l2-lambda 0.008 --muqsm-tv-lambda 1e-5 --qsm-tv-lambda 1e-5 --no-display
```

These values are the existing defaults, not values estimated for your dataset.
The separate L-curve example below can help estimate the two L2 values. The
public script does not load those estimates automatically.

To run from VS Code instead, edit `INPUT_DIR`, `OUTPUT_DIR`, `INPUT_FORMAT`, and
`REGULARIZATION` near the top of `Micro_QSM_Recon.py`. Set `REGULARIZATION` to
`'l2'`, `'tv'`, or `'both'`, select the `mesochi` interpreter, then
use **Run Python File in Terminal**. Command-line arguments override the
editable settings. Paths passed on the command line are relative to the
terminal's current directory unless absolute.

On the first GPU run, CuPy compiles kernels. If its cache is not writable,
choose a writable location before starting Python, for example from the project
root:

```powershell
$env:CUPY_CACHE_DIR = Join-Path (Get-Location) '.cache/cupy'
```

## 4. Understand the calculation and outputs

The supplied phase conversion and phase-mask reference are:

```python
field = phase / (gammaB0 * mean(diff(TEs)) * 1e-3)
local_field = mask_phase * (field - mean(field[mask_phase]))
```

The mesoscopic term is `-0.5 * (bᵀ T b - 1/3)`, restricted by the phase and
white-matter masks, where `T` is the FBI scatter matrix and `b` is the normalized
`B0_dir` in the same array-axis frame as the b-vectors. The original x-axis
calculation is retained for the supplied data. The arbitrary-direction
projection has synthetic rotation tests; validate coordinate conventions for
each dataset.

Outputs are cropped to the input dimensions and CSF-referenced as
`-(chi - mean(chi[mask_csf]))`, in ppm. Files are written to `--output-dir`.
With the default export settings, the reconstructed maps are saved in:

- `exvivo_reconstruction.mat` — a MATLAB file.
- `exvivo_reconstruction.npz` — a compressed NumPy archive for Python.

Both files contain the same selected maps: two maps for `l2` or `tv`, or all
four for `both`. The map variable names and corresponding NIfTI filenames are:

| Map variable in MAT/NPZ | Reconstruction | NIfTI filename |
| --- | --- | --- |
| `Chi_QSM_L2` | Standard QSM with L2 regularization | `Chi_QSM_L2.nii.gz` |
| `Chi_muQSM_L2` | μQSM with L2 regularization | `Chi_muQSM_L2.nii.gz` |
| `Chi_QSM_TV` | Standard QSM with TV regularization | `Chi_QSM_TV.nii.gz` |
| `Chi_muQSM_TV` | μQSM with TV regularization | `Chi_muQSM_TV.nii.gz` |

NIfTI files are exported only for the selected maps when NIfTI output is
requested and a matching reference is available. NIfTI inputs supply the phase
image as the reference automatically; MATLAB inputs require
`--nifti-reference`. Array ordering and validated spatial geometry are preserved
without resampling. Without a reference, NIfTI export is skipped when MAT or NPZ
output is also selected; NIfTI-only output requires a matching reference.
Use `--output-formats` to choose the map export formats, for example
`--output-formats npz mat`, or edit `OUTPUT_FORMATS` in the script.

The other default output files are:

| File | Content |
| --- | --- |
| `exvivo_slices.png` | Orthogonal slices of μQSM L2 when available, otherwise μQSM TV |
| `exvivo_comparison.png` | QSM/μQSM maps and their differences for the selected methods |
| `exvivo_run.json` | Inputs, selected regularization, parameters, iterations, and saved file paths |

`--no-display` saves the PNG figures without opening windows. `--no-plots`
skips both figure files; map exports and the JSON run record are still saved.

## 5. Ex vivo L-curve example

Run the following after placing the supplied MATLAB data in `data/exvivo/`.
The example defaults to GPU; `--no-use-gpu` uses the portable CPU environment:

```powershell
python examples/exvivo_lcurve.py --no-use-gpu --no-display
```

To use another prepared MATLAB input folder or output folder, pass
`--input-dir` and `--output-dir`. The script estimates separate QSM and μQSM L2
parameters using 21 candidates from `1e-5` to `1e-1` for each model. It saves
raw padded residual norms `||Ax-y||₂`, solution norms `||x||₂`, iteration
counts, plots, selections, and selected maps. Each solve starts from zero.

Inspect the plots and flags, then manually transfer accepted L2 estimates to
`Micro_QSM_Recon.py`. A boundary or nonmonotonic curve needs closer inspection;
an unresolved curve is reported without substituting a fixed value. L-curve
selection does not estimate TV parameters or automatically change reconstruction
settings. The supplied full-size sweep can take substantial time.
Use a new output directory if an older four-curve run exists: existing files are
not cleared, so old TV plots and archives can remain alongside the new results.

