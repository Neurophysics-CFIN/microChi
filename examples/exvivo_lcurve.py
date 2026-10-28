"""Estimate ex vivo QSM and muQSM L2 parameters, separately from reconstruction."""
#%% Imports
import argparse
from pathlib import Path
import numpy as np
try:
    from mesochi.exvivo.lcurve import run_exvivo_lcurve
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
OUTPUT_DIR = ROOT / 'results' / 'exvivo' / 'l_curve'
WEIGHT_FILE = None
NIFTI_REFERENCE = None
OUTPUT_FORMATS = ('npz', 'mat', 'nifti')
USE_GPU = True
B_FBI = None
L_MAX = 6
FBI_METHOD = 'corrected'
DA = None
D0 = 2e-3
L2_VALUES = np.logspace(-5, -1, 21)
CG_MAX_ITER = 200
TOL = 1e-6
MAKE_PLOTS = True
SHOW_PLOTS = True
SHOW_PROGRESS = True

#%% Script interface
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--weight-file', type=Path, default=WEIGHT_FILE)
    parser.add_argument('--nifti-reference', type=Path, default=NIFTI_REFERENCE)
    parser.add_argument('--output-formats', nargs='+', choices=('npz', 'mat', 'nifti'), default=OUTPUT_FORMATS)
    parser.add_argument('--use-gpu', action=argparse.BooleanOptionalAction, default=USE_GPU)
    parser.add_argument('--b-fbi', type=float, default=B_FBI)
    parser.add_argument('--l-max', type=int, default=L_MAX)
    parser.add_argument('--fbi-method', choices=('corrected', 'exact'), default=FBI_METHOD)
    parser.add_argument('--Da', type=float, default=DA)
    parser.add_argument('--D0', type=float, default=D0)
    parser.add_argument('--l2-values', nargs='+', type=float, default=L2_VALUES,
                        help='At least five distinct positive candidates.')
    parser.add_argument('--cg-max-iter', type=int, default=CG_MAX_ITER)
    parser.add_argument('--tol', type=float, default=TOL)
    parser.add_argument('--no-plots', dest='make_plots', action='store_false', default=MAKE_PLOTS)
    parser.add_argument('--no-display', dest='show_plots', action='store_const', const=False, default=SHOW_PLOTS)
    parser.add_argument('--show-progress', action=argparse.BooleanOptionalAction, default=SHOW_PROGRESS)
    args = parser.parse_args(argv)
    try:
        run_exvivo_lcurve(**vars(args))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    return 0

#%% Run (42 candidates + selected reruns)
if __name__ == '__main__':
    raise SystemExit(main())
