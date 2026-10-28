"""CPU QSM smoke check using the bundled cylindrical phantom.

From the project root, after installing MesoChi and its dependencies, run:

    python tests/smoke_reconstruction.py

The result must be a finite 32-by-32-by-32 array whose relative field residual
is below 1. This check does not save files or reproduce the full experiment.
"""

from pathlib import Path

import numpy as np

from mesochi.simulation.operators import QSMLinop
from mesochi.simulation.reconstruction import _solve_cg


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "phantoms"
    with np.load(data_dir / "Cylindrical_Phantom.npz") as phantom, np.load(
        data_dir / "LoFds.npz"
    ) as fields:
        mask = phantom["mask_roi"]
        observed = fields["cylindrical"]
        operator = QSMLinop(
            mask.shape, fields["voxel_size_lr"], fields["dir_samp"], mask,
            use_gpu=False,
        )
        result = _solve_cg(
            operator, observed, max_iter=20, lambda_reg=0.01,
            show_progress=False,
        )
        relative_residual = np.linalg.norm(operator(result) - observed) / np.linalg.norm(observed)

    if result.shape != (32, 32, 32) or not np.isfinite(result).all():
        raise RuntimeError("Reconstruction must be a finite 32-by-32-by-32 array")
    if not np.isfinite(relative_residual) or relative_residual >= 1:
        raise RuntimeError(f"Unexpected relative residual: {relative_residual}")
    print(f"CPU QSM smoke check passed; relative field residual: {relative_residual:.6f}")


if __name__ == "__main__":
    main()
