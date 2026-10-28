"""Generate a new simulation dataset; the default grid requires tens of GiB."""
#%% Imports
from pathlib import Path
from mesochi.simulation.phantoms import generate_phantoms

#%% Editable settings
SIM_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = SIM_DIR / 'data' / 'generated'
N_HR = (1024, 1024, 1024)
VOXEL_SIZE_HR = (0.03125, 0.03125, 0.03125)
DOWNSAMPLE_FACTOR = 32
WINDOW_SIZE = 32
R_CONTAINER = 8.0
NUM_ORIENTATIONS = 6
VOL_FRAC_CYL = 0.5
VOL_FRAC_SPH = 0.05
CONTRIBUTION_CYL = 0.9
B0_DIR = (1, 0, 0)
B0_T = 3.0
TE = 5e-3
BACKGROUND_MAG = 0.5
SNR_LEVELS = (float('inf'), 100, 50, 30, 10)

#%% Run (also usable as an editor cell)
if __name__ == '__main__':
    files = generate_phantoms(
        OUTPUT_DIR, n_hr=N_HR, voxel_size_hr=VOXEL_SIZE_HR,
        downsample_factor=DOWNSAMPLE_FACTOR, window_size=WINDOW_SIZE,
        r_container=R_CONTAINER, num_orientations=NUM_ORIENTATIONS,
        vol_frac_cyl=VOL_FRAC_CYL, vol_frac_sph=VOL_FRAC_SPH,
        contribution_cyl=CONTRIBUTION_CYL, b0_dir=B0_DIR, b0_t=B0_T,
        te=TE, background=BACKGROUND_MAG, snr_levels=SNR_LEVELS,
    )
    print(f'Saved {len(files)} datasets in {OUTPUT_DIR}')
