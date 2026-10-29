"""Run the multi-orientation comparisons and single-orientation L2/TV examples."""
#%% Imports
from pathlib import Path
from mesochi.simulation.reconstruction import run_reconstruction

#%% Editable settings
SIM_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SIM_DIR / 'data' / 'phantoms'  # Use data/generated for new simulations.
RESULT_DIR = SIM_DIR / 'results'
USE_GPU = False
SHOW_PLOTS = None  # Automatically disabled for the Agg backend.
FIGURE_DPI = 100  # Interactive display; lower this further for smaller screens.
SAVE_DPI = 800
L2_LAMBDA = 1e-2
TV_LAMBDA = 1e-3
B0_DIR = (1, 0, 0)

#%% Run
if __name__ == '__main__':
    results = run_reconstruction(
        DATA_DIR, RESULT_DIR, use_gpu=USE_GPU, show_plots=SHOW_PLOTS,
        figure_dpi=FIGURE_DPI, save_dpi=SAVE_DPI,
        l2_lambda=L2_LAMBDA, tv_lambda=TV_LAMBDA, b0_dir=B0_DIR,
    )
