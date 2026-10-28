"""Ex vivo QSM and μQSM L2 curves, with selections for manual transfer."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from mesochi.lcurve import find_lcurve_corner, plot_lcurve_analysis, RC_LCURVE
from mesochi.operators import _get_backend, _to_numpy
from .pipeline import (
    MAP_KEYS, prepare_exvivo_data, build_operators,
    validate_solver_settings, solve_case, crop_and_reference,
)
from .io import read_metadata, prepare_exports, save_maps, save_record


def _candidate_grid(values, default):
    values = np.asarray(default if values is None else values, dtype=float)
    if (values.ndim != 1 or len(values) < 5 or not np.all(np.isfinite(values))
            or np.any(values <= 0) or len(np.unique(values)) != len(values)):
        raise ValueError('Each L-curve grid needs at least five distinct positive, finite values.')
    return values


def select_candidate(residuals, penalties, values):
    """Keep unresolved curves unresolved, and attach practical inspection flags."""
    order = np.argsort(values)
    flags = []
    if np.all(residuals > 0) and np.all(penalties > 0):
        if (np.any(np.diff(np.log10(residuals[order])) < -1e-8)
                or np.any(np.diff(np.log10(penalties[order])) > 1e-8)):
            flags.append('Norms are not monotonic; inspect solver convergence and the curve.')
    try:
        selected, index, info = find_lcurve_corner(
            residuals, penalties, values, method='curvature', smoothing=True, plot=False,
        )
    except ValueError as error:
        return None, None, None, flags + [str(error)]
    position = int(np.flatnonzero(order == index)[0])
    edge = max(1, int(np.ceil(0.1 * (len(values) - 1))))
    if position <= edge or position >= len(values) - 1 - edge:
        flags.append('Selection is near a grid boundary; consider expanding the grid.')
    return selected, index, info, flags


def _save_curve(path, residuals, penalties, values, selected, index, info, show):
    label = 'Solution Norm ||x||'
    with plt.rc_context(RC_LCURVE):
        if selected is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            order = np.argsort(values)
            if np.all(residuals > 0) and np.all(penalties > 0):
                ax.loglog(residuals[order], penalties[order], 'bo-')
            else:
                ax.plot(residuals[order], penalties[order], 'bo-')
            ax.set(xlabel='Residual Norm ||Ax-y||_2', ylabel=label,
                   title='Unresolved L-curve: inspect norms and convergence')
            fig.tight_layout()
        else:
            fig = plot_lcurve_analysis(residuals, penalties, values, index, selected,
                                       info, show=False, penalty_label=label)
        try:
            fig.savefig(path)
            if show:
                plt.show()
        finally:
            if not show:
                plt.close(fig)


def run_exvivo_lcurve(
    input_dir, output_dir, *, weight_file=None, nifti_reference=None,
    output_formats=('npz', 'mat', 'nifti'), use_gpu=False,
    b_fbi=None, l_max=6, fbi_method='corrected', Da=None, D0=2e-3,
    l2_values=None,
    cg_max_iter=200, tol=1e-6,
    make_plots=True, show_plots=None, show_progress=False,
):
    """Sweep QSM/μQSM with L2 and report two independent parameter estimates.

    Defaults perform 42 candidates, plus at most two selected reruns. Each solve
    starts from zero. Norms use raw padded solutions: ||Ax-y||_2 and ||x||_2.
    Only selected, cropped, CSF-referenced maps are retained.
    Main reconstruction does not read selections.json automatically.
    """
    l2_values = _candidate_grid(l2_values, np.logspace(-5, -1, 21))
    validate_solver_settings([], cg_max_iter, 300, 10, tol, 1)
    _get_backend(use_gpu)
    meta = read_metadata(input_dir)
    exports = prepare_exports(output_formats, nifti_reference, meta['Dim'], meta['Voxel_size'])
    data = prepare_exvivo_data(input_dir, weight_file=weight_file, b_fbi=b_fbi,
                              l_max=l_max, fbi_method=fbi_method, Da=Da, D0=D0)
    del data['FBI']  # Sweep norms only need the prepared field and mesoscopic volume.
    operators, y, G = build_operators(data, use_gpu=use_gpu)
    settings = dict(cg_max_iter=cg_max_iter, tol=tol, show_progress=show_progress)
    show = plt.get_backend().lower() != 'agg' if show_plots is None else show_plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves, selections, saved_files = {}, {}, []
    for case in ('muqsm_l2', 'qsm_l2'):
        model, penalty = case.split('_')
        A, xp = operators[model], operators[model].xp
        values = l2_values
        residuals, penalties, iterations = [], [], []
        for value in values:
            chi, count = solve_case(A, y, penalty, value, G, **settings)
            residuals.append(float(_to_numpy(xp.linalg.norm((A(chi) - y).ravel()))))
            penalties.append(float(_to_numpy(xp.linalg.norm(chi.ravel()))))
            iterations.append(count)
            del chi
            print(f'{case} lambda={value:.3g}: residual={residuals[-1]:.6g}, penalty={penalties[-1]:.6g}')
        residuals, penalties = np.asarray(residuals), np.asarray(penalties)
        selected, index, info, flags = select_candidate(residuals, penalties, values)
        arrays = dict(lambda_values=values.copy(), residual_norms=residuals,
                      penalty_norms=penalties, iterations=np.asarray(iterations),
                      selected_lambda=np.nan if selected is None else selected,
                      selected_index=-1 if index is None else index)
        # Always preserve raw measurements, including unresolved curves.
        archive = output_dir / f'{case}_lcurve.npz'
        np.savez_compressed(archive, **arrays)
        saved_files.append(str(archive.resolve()))
        if 'mat' in exports['formats']:
            archive = output_dir / f'{case}_lcurve.mat'
            savemat(archive, arrays, do_compression=True)
            saved_files.append(str(archive.resolve()))
        if make_plots:
            path = output_dir / f'{case}_lcurve.png'
            _save_curve(path, residuals, penalties, values, selected, index, info, show)
            saved_files.append(str(path.resolve()))
        curves[case] = arrays
        selected_iterations = None
        if selected is not None:
            chi, selected_iterations = solve_case(A, y, penalty, selected, G, **settings)
            selected_map = crop_and_reference(chi, data)
            del chi
            saved_files += save_maps({MAP_KEYS[case]: selected_map}, output_dir, exports,
                                     stem=f'{case}_selected')
            del selected_map
        selections[case] = dict(lambda_value=selected, candidate_index=index,
                                status='selected' if selected is not None else 'unresolved',
                                flags=flags, selected_rerun_iterations=selected_iterations)
        print(f'{case}: ' + ('unresolved' if selected is None else f'L-curve estimate {selected:.6g}'))
        for flag in flags:
            print(f'  {flag}')
    record_path = output_dir / 'selections.json'
    saved_files.append(str(record_path.resolve()))
    run_settings = dict(**data['settings'], **settings, metadata=meta,
                        use_gpu=use_gpu, l2_values=l2_values,
                        output_formats=exports['formats'], nifti_status=exports['nifti_status'],
                        nifti_reference=None if nifti_reference is None else str(Path(nifti_reference).resolve()),
                        make_plots=make_plots, show_plots=show)
    save_record(record_path, dict(selections=selections, settings=run_settings,
                                 saved_files=saved_files,
                                 instruction='Inspect all curves, then manually copy selected values into reconstruction settings.'))
    print('Selected parameters for manual transfer:')
    for case in ('qsm_l2', 'muqsm_l2'):
        value = selections[case]['lambda_value']
        print(f'  {case}_lambda = {value!r}')
    return dict(curves=curves, selections=selections, settings=run_settings, saved_files=saved_files)
