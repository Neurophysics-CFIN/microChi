"""Run MesoChi on prepared ex vivo MATLAB inputs and save four susceptibility maps."""
#%% Imports
import argparse
from pathlib import Path
try:
    from mesochi.exvivo.pipeline import run_exvivo_reconstruction
except ModuleNotFoundError as error:
    if error.name != 'mesochi':
        raise
    import sys
    raise ModuleNotFoundError(
        'Install MesoChi in the Python environment running this script.\n'
        f'Active Python: {sys.executable}\n'
        'From the repository root: python -m pip install -e ".[nifti]"\n'
        'In Jupyter, use %pip install -e "/full/path/to/the/repository[nifti]", '
        'then restart the kernel. See docs/reconstruction_tutorial.md.',
        name='mesochi',
    ) from error

#%% Editable settings (command-line arguments override these defaults)
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / 'data' / 'exvivo'
OUTPUT_DIR = ROOT / 'results' / 'exvivo'
WEIGHT_FILE = None  # Optional .npy or .mat containing weight; None means ones.
NIFTI_REFERENCE = None  # Matching .nii/.nii.gz required only for NIfTI export.
OUTPUT_FORMATS = ('npz', 'mat', 'nifti')
REGULARIZATION = 'both'  # 'l2', 'tv', or 'both'.
USE_GPU = False
B_FBI = None  # First b-value by default.
L_MAX = 6
FBI_METHOD = 'corrected'
DA = None  # Required for method='exact'.
D0 = 2e-3  # Required for method='corrected', in mm^2/s.
QSM_L2_LAMBDA = 8e-3
MUQSM_L2_LAMBDA = 8e-3
QSM_TV_LAMBDA = 1e-5
MUQSM_TV_LAMBDA = 1e-5
CG_MAX_ITER = 200
TV_MAX_ITER = 300
TV_MAX_CG_ITER = 10
TOL = 1e-6
RHO = 1.0
MAKE_PLOTS = True
SHOW_PLOTS = None  # Automatically disabled for Agg.
FIGURE_DPI = 400
SAVE_DPI = 800
SLICE_XS = None  # Supplied data: (65,85,105); smaller grids use mask slices.
SLICE_INDICES = None  # Orthogonal slices (x,y,z).
SHOW_PROGRESS = False

#%% Script interface
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--weight-file', type=Path, default=WEIGHT_FILE,
                        help='Optional .npy or .mat containing weight; default: uniform ones.')
    parser.add_argument('--nifti-reference', type=Path, default=NIFTI_REFERENCE,
                        help='Matching reference image; NIfTI export is skipped without one.')
    parser.add_argument('--output-formats', nargs='+', choices=('npz', 'mat', 'nifti'), default=OUTPUT_FORMATS)
    parser.add_argument('--regularization', type=str.lower, choices=('l2', 'tv', 'both'),
                        default=REGULARIZATION,
                        help='Reconstruct QSM and muQSM with L2, TV, or both (default).')
    parser.add_argument('--use-gpu', action=argparse.BooleanOptionalAction, default=USE_GPU)
    parser.add_argument('--b-fbi', type=float, default=B_FBI)
    parser.add_argument('--l-max', type=int, default=L_MAX)
    parser.add_argument('--fbi-method', choices=('corrected', 'exact'), default=FBI_METHOD)
    parser.add_argument('--Da', type=float, default=DA)
    parser.add_argument('--D0', type=float, default=D0)
    parser.add_argument('--qsm-l2-lambda', type=float, default=QSM_L2_LAMBDA)
    parser.add_argument('--muqsm-l2-lambda', type=float, default=MUQSM_L2_LAMBDA)
    parser.add_argument('--qsm-tv-lambda', type=float, default=QSM_TV_LAMBDA)
    parser.add_argument('--muqsm-tv-lambda', type=float, default=MUQSM_TV_LAMBDA)
    parser.add_argument('--cg-max-iter', type=int, default=CG_MAX_ITER)
    parser.add_argument('--tv-max-iter', type=int, default=TV_MAX_ITER)
    parser.add_argument('--tv-max-cg-iter', type=int, default=TV_MAX_CG_ITER)
    parser.add_argument('--tol', type=float, default=TOL)
    parser.add_argument('--rho', type=float, default=RHO)
    parser.add_argument('--no-plots', dest='make_plots', action='store_false', default=MAKE_PLOTS,
                        help='Skip figure creation and saving.')
    parser.add_argument('--no-display', dest='show_plots', action='store_const', const=False, default=SHOW_PLOTS)
    parser.add_argument('--figure-dpi', type=float, default=FIGURE_DPI)
    parser.add_argument('--save-dpi', type=float, default=SAVE_DPI)
    parser.add_argument('--slice-xs', nargs=3, type=int, default=SLICE_XS)
    parser.add_argument('--slice-indices', nargs=3, type=int, default=SLICE_INDICES)
    parser.add_argument('--show-progress', action=argparse.BooleanOptionalAction, default=SHOW_PROGRESS)
    args = parser.parse_args(argv)
    try:
        run_exvivo_reconstruction(**vars(args))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    return 0

#%% Run
if __name__ == '__main__':
    raise SystemExit(main())
