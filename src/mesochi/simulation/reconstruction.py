"""Multi-orientation and noisy single-orientation reconstruction experiments."""
from pathlib import Path
import numpy as np
import sigpy as sgp
import matplotlib.pyplot as plt
from . import plotting
from .operators import (
    QSMLinop, MuQSMLinop, MuQSMLinop_S, MuQSMLinop_CS,
    prepare_y_for_sigpy, split_solution_S, split_solution_CS, _to_numpy,
)


def _load_arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in (archive.files if keys is None else keys)}


def _solve_cg(A, y, max_iter, lambda_reg=0.0, show_progress=True):
    """Start each independent CG solve from a fresh float32 zero buffer."""
    xp = A.xp
    return sgp.app.LinearLeastSquares(
        A=A, y=y, x=xp.zeros(A.ishape, dtype=xp.float32),
        lamda=lambda_reg, solver="ConjugateGradient", max_iter=max_iter,
        tol=1e-6, show_pbar=show_progress, leave_pbar=True,
    ).run()


def _solve_tv(A, y, G, proxg, *, max_iter=300, max_cg_iter=10):
    """Run TV from zero, handling SigPy's float32 zero-gradient warning."""
    solver = sgp.app.LinearLeastSquares(
        A=A, y=y, x=A.xp.zeros(A.ishape, dtype=A.dtype),
        G=G, proxg=proxg, lamda=0, solver='ADMM', rho=1.0,
        max_iter=max_iter, max_cg_iter=max_cg_iter, tol=1e-6, show_pbar=False,
    )
    # SigPy's Numba soft threshold can flag invalid arithmetic for zero input
    # while returning the correct zero. Keep handling local to this solve and
    # reject genuinely nonfinite reconstructions before plotting or exporting.
    with np.errstate(divide='ignore', invalid='ignore'):
        chi = solver.run()
    if not bool(_to_numpy(A.xp.all(A.xp.isfinite(chi)))):
        raise RuntimeError('TV solver returned nonfinite values; inspect inputs and solver settings.')
    return chi


def run_reconstruction(
    data_dir, result_dir, *, use_gpu=False, show_plots=None,
    figure_dpi=100, save_dpi=800, l2_lambda=1e-2, tv_lambda=1e-3,
    b0_dir=(1, 0, 0), show_progress=True,
):
    """Run the paper experiments, save three archives and five figures.

    Return a dictionary of independent arrays grouped under the archive stems:
    multi_orientation, single_orientation_l2, and single_orientation_tv.
    Input/output directories are explicit; the examples supply checkout defaults.
    show_plots=None displays figures unless Matplotlib uses the Agg backend.
    figure_dpi controls the interactive display; save_dpi controls file exports.
    """
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    show_plots = plt.get_backend().lower() != 'agg' if show_plots is None else show_plots

    # Load only the arrays needed for reconstruction, then close each archive.
    P_cyl = _load_arrays(data_dir / 'Cylindrical_Phantom.npz', ('chi_cyl_lr', 'mask_roi'))
    P_sph = _load_arrays(data_dir / 'Spherical_Phantom.npz', ('chi_sph_lr',))
    P_comp = _load_arrays(data_dir / 'Composite_Phantom.npz',
                          ('chi_comp_lr', 'chi_cyl_lr', 'chi_sph_lr'))
    LoFds = _load_arrays(data_dir / 'LoFds.npz')
    noisy_fields = _load_arrays(data_dir / 'LoFd_cyl_noisy.npz')
    chi_cyl_gt, chi_sph_gt = P_cyl['chi_cyl_lr'], P_sph['chi_sph_lr']
    chi_comp_gt = P_comp['chi_comp_lr']
    mask_roi = P_cyl['mask_roi']
    dir_samp, voxel_size_lr = LoFds['dir_samp'], LoFds['voxel_size_lr']
    N_lr = chi_cyl_gt.shape
    B0_dir = np.asarray(b0_dir)
    N_meso = [-0.5 * (direction[2]**2 - 1.0/3.0) * mask_roi
              for direction in dir_samp]

    # Standard QSM has the same geometry and masks in all three experiments.
    A_qsm = QSMLinop(N_lr, voxel_size_lr, dir_samp, mask_roi, use_gpu=use_gpu)
    A_cyl = MuQSMLinop(N_lr, voxel_size_lr, dir_samp, mask_roi, N_meso, use_gpu=use_gpu)
    A_sph = MuQSMLinop_S(N_lr, voxel_size_lr, dir_samp, mask_roi, N_meso, use_gpu=use_gpu)
    A_comp = MuQSMLinop_CS(N_lr, voxel_size_lr, dir_samp, mask_roi, N_meso, use_gpu=use_gpu)
    # Match each consuming operator: center raw fields before applying its weights.
    # Prepare observations once per solve; _solve_cg accepts already prepared data.
    chi_cyl_muqsm = _to_numpy(_solve_cg(
        A_cyl, prepare_y_for_sigpy(LoFds['cylindrical'], A_cyl,
                                  remove_mean=True, apply_weight=True),
        100, show_progress=show_progress,
    )).copy()
    chi_cyl_qsm = _to_numpy(_solve_cg(
        A_qsm, prepare_y_for_sigpy(LoFds['cylindrical'], A_qsm,
                                  remove_mean=True, apply_weight=True),
        100, show_progress=show_progress,
    )).copy()
    solution_sph = _solve_cg(
        A_sph, prepare_y_for_sigpy(LoFds['spherical'], A_sph,
                                  remove_mean=True, apply_weight=True),
        50, show_progress=show_progress,
    )
    chi_sph_muqsm, u_sph = split_solution_S(solution_sph, A_sph)
    chi_sph_muqsm, u_sph = _to_numpy(chi_sph_muqsm).copy(), _to_numpy(u_sph).copy()
    chi_sph_qsm = _to_numpy(_solve_cg(
        A_qsm, prepare_y_for_sigpy(LoFds['spherical'], A_qsm,
                                  remove_mean=True, apply_weight=True),
        100, show_progress=show_progress,
    )).copy()
    solution_comp = _solve_cg(
        A_comp, prepare_y_for_sigpy(LoFds['composite'], A_comp,
                                  remove_mean=True, apply_weight=True),
        100, show_progress=show_progress,
    )
    chi_comp_cyl, chi_comp_sph, u_comp = split_solution_CS(solution_comp, A_comp)
    chi_comp_cyl = _to_numpy(chi_comp_cyl).copy()
    chi_comp_sph = _to_numpy(chi_comp_sph).copy()
    u_comp = _to_numpy(u_comp).copy()
    chi_comp_muqsm = chi_comp_cyl + chi_comp_sph
    chi_comp_qsm = _to_numpy(_solve_cg(
        A_qsm, prepare_y_for_sigpy(LoFds['composite'], A_qsm,
                                  remove_mean=True, apply_weight=True),
        100, show_progress=show_progress,
    )).copy()

    multi = dict(
        chi_cyl_multi_muqsm=chi_cyl_muqsm, chi_cyl_multi_qsm=chi_cyl_qsm,
        chi_sph_multi_muqsm=chi_sph_muqsm, chi_sph_multi_qsm=chi_sph_qsm,
        chi_comp_multi_muqsm=chi_comp_muqsm, chi_comp_multi_qsm=chi_comp_qsm,
        chi_comp_cyl_multi_muqsm=chi_comp_cyl,
        chi_comp_sph_multi_muqsm=chi_comp_sph,
        u_sph_multi_muqsm=u_sph, u_comp_multi_muqsm=u_comp,
    )

    # Figure settings are scoped to this run and restored even if plotting fails.
    result_dir.mkdir(parents=True, exist_ok=True)
    styles = (plotting.RC_COMPONENTS, plotting.RC_COMPARISON, plotting.RC_SNR)
    original_styles = [style.copy() for style in styles]
    before_figures = set(plt.get_fignums())
    with plt.rc_context():
        try:
            for style in styles:
                style.update({'figure.dpi': figure_dpi, 'savefig.dpi': save_dpi})

            # Multi-orientation figures precede the single-orientation experiments.
            fig = plotting.plot_components(
                gt={'': chi_comp_gt, 'c': P_comp['chi_cyl_lr'], 's': P_comp['chi_sph_lr']},
                rec={'': chi_comp_muqsm, 'c': chi_comp_cyl, 's': chi_comp_sph},
                mask_roi=mask_roi, save_path=result_dir / 'composite_components.png',
                show=show_plots,
            )
            if not show_plots:
                plt.close(fig)
            fig = plotting.plot_comparison(
                chi_sph_gt, {plotting.QSM: chi_sph_qsm, plotting.MUQSM: chi_sph_muqsm},
                mask_roi, component='s', save_path=result_dir / 'spherical_comparison.png',
                show=show_plots,
            )
            if not show_plots:
                plt.close(fig)
            fig = plotting.plot_comparison(
                chi_cyl_gt, {plotting.QSM: chi_cyl_qsm, plotting.MUQSM: chi_cyl_muqsm},
                mask_roi, component='c', save_path=result_dir / 'cylindrical_comparison.png',
                show=show_plots,
            )
            if not show_plots:
                plt.close(fig)

            # Prepare each field once for the two independent regularized solves.
            N_meso_single = -0.5 * (B0_dir[2]**2 - 1.0/3.0) * mask_roi
            A_single = MuQSMLinop(N_lr, voxel_size_lr, B0_dir, mask_roi,
                                  N_meso_single, use_gpu=use_gpu)
            prepared_fields = {
                key: prepare_y_for_sigpy(field, A_single, remove_mean=True, apply_weight=True)
                for key, field in noisy_fields.items()
            }
            recon_l2 = {}
            for key, y in prepared_fields.items():
                chi = _solve_cg(A_single, y, 100, l2_lambda, show_progress)
                recon_l2[key] = _to_numpy(chi).copy()

            # The TV operator and L1 penalty are identical across SNR levels.
            G = sgp.linop.FiniteDifference(A_single.ishape, axes=(0, 1, 2))
            proxg = sgp.prox.L1Reg(G.oshape, tv_lambda)
            recon_tv = {}
            for key, y in prepared_fields.items():
                chi = _solve_tv(A_single, y, G, proxg)
                recon_tv[key] = _to_numpy(chi).copy()

            fig = plotting.plot_snr_comparison(
                chi_cyl_gt, recon_tv, mask_roi,
                save_path=result_dir / 'snr_tv.png', show=show_plots,
            )
            if not show_plots:
                plt.close(fig)
            fig = plotting.plot_snr_comparison(
                chi_cyl_gt, recon_l2, mask_roi,
                save_path=result_dir / 'snr_l2.png', show=show_plots,
            )
            if not show_plots:
                plt.close(fig)
        finally:
            for style, original in zip(styles, original_styles):
                style.clear()
                style.update(original)
            if not show_plots:
                for number in set(plt.get_fignums()) - before_figures:
                    plt.close(number)

    np.savez(result_dir / 'multi_orientation.npz', **multi)
    np.savez(result_dir / 'single_orientation_l2.npz', **recon_l2)
    np.savez(result_dir / 'single_orientation_tv.npz', **recon_tv)
    return dict(multi_orientation=multi, single_orientation_l2=recon_l2,
                single_orientation_tv=recon_tv)
