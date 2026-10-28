"""Prepare ex vivo MATLAB data and reconstruct QSM/μQSM with L2 and TV."""
from pathlib import Path
import numpy as np
import sigpy as sp

from mesochi.diffusion import Calc_FBI_SctMtx
from mesochi.padding import padarray_post
from mesochi.operators import (
    QSMLinop, MuQSMLinop, prepare_y_for_lsq, _get_backend, _to_numpy,
)
from .io import (load_required_mat, read_metadata, load_weight, prepare_exports,
                 save_maps, save_record, load_nifti_data, _nifti_path)

CASES = ('muqsm_l2', 'qsm_l2', 'muqsm_tv', 'qsm_tv')
MAP_KEYS = {
    'muqsm_l2': 'chi_muqsm_ref', 'qsm_l2': 'chi_qsm_ref',
    'muqsm_tv': 'chi_muqsm_tv_ref', 'qsm_tv': 'chi_qsm_tv_ref',
}


def mesoscopic_contribution(scatter, direction, mask_phase, mask_wm):
    """Project the scatter tensor onto B0, retaining the original x-axis arithmetic."""
    b = np.asarray(direction, dtype=float)
    b = b / np.linalg.norm(b)
    if b[1] == 0 and b[2] == 0:
        projection = scatter[..., 0, 0]
    else:
        projection = np.einsum('i,...ij,j->...', b, scatter, b)
    return -0.5 * mask_phase * np.real(projection - 1/3) * mask_wm


def prepare_exvivo_data(input_dir, *, input_format='mat', weight_file=None, b_fbi=None, l_max=6,
                       fbi_method='corrected', Da=None, D0=2e-3):
    """Load MATLAB or NIfTI inputs, calculate FBI, and return unpadded arrays.

    Required variables, NIfTI layout, and units are described in docs/reconstruction_tutorial.md.
    weight_file is optional (.npy or .mat containing weight); None means ones.
    B0 and diffusion directions are expressed in image-array axes.
    No registration, unwrapping, or background-field removal is performed.
    """
    input_dir = Path(input_dir)
    meta = read_metadata(input_dir, input_format)
    shape = meta['Dim']
    nifti = load_nifti_data(input_dir, meta) if input_format == 'nifti' else None
    phase = nifti['Phase_map'] if nifti is not None else load_required_mat(input_dir / 'Phase_map.mat', ('phase',))['phase']
    masks = {}
    for name in ('mask_phase', 'mask_wm', 'mask_csf'):
        raw = nifti[name] if nifti is not None else load_required_mat(input_dir / f'{name}.mat', (name,))[name]
        if raw.shape != shape or not np.isrealobj(raw) or not np.all(np.isfinite(raw)):
            raise ValueError(f'{name} must be a finite, real 3D volume matching Dim.')
        masks[name] = raw.astype(bool)
    if phase.shape != shape or not np.isrealobj(phase) or not np.all(np.isfinite(phase)):
        raise ValueError('phase must be a finite, real 3D volume matching Dim.')
    for name in ('mask_phase', 'mask_csf'):
        if not masks[name].any():
            raise ValueError(f'{name} must contain at least one voxel.')
    weight = load_weight(weight_file, shape)
    dwi = nifti if nifti is not None else load_required_mat(input_dir / 'DWI_FBI.mat', ('DWI_FBI', 'bval', 'bvec'))
    signal, bvals, bvecs = dwi['DWI_FBI'], dwi['bval'].ravel(), dwi['bvec']
    if signal.ndim != 4 or signal.shape[:3] != shape or not np.isrealobj(signal):
        raise ValueError('DWI_FBI must be a real (X,Y,Z,K) volume matching Dim.')
    shell = float(bvals[0]) if b_fbi is None and bvals.size else b_fbi
    if b_fbi is None and shell == 0:
        positive = bvals[bvals > 0]
        shells = np.unique(np.round(positive))
        if len(shells) != 1:
            raise ValueError('Specify b_fbi when there are zero or multiple positive b-value shells.')
        shell = float(positive[0])
    if shell is None:
        raise ValueError('bval must contain diffusion measurements.')
    print('Calculating FBI scatter matrices...')
    fbi = Calc_FBI_SctMtx(signal, bvals, bvecs, mask=masks['mask_phase'],
                        b_FBI=shell, l_max=l_max, method=fbi_method, Da=Da, D0=D0)
    n_meso_wm = mesoscopic_contribution(fbi['ScatterMatrix'], meta['B0_dir'],
                                       masks['mask_phase'], masks['mask_wm'])
    field = phase / (meta['gammaB0'] * np.mean(np.diff(meta['TEs'])) * 1e-3)
    field_reference = np.mean(field[masks['mask_phase']])
    local_field = masks['mask_phase'] * (field - field_reference)
    if not np.all(np.isfinite(local_field)) or not np.all(np.isfinite(local_field.astype(np.float32))):
        raise ValueError('Converted local field must be finite and fit float32.')
    return dict(metadata=meta, local_field=local_field, N_meso=n_meso_wm,
                weight=weight, FBI=fbi, **masks,
                settings=dict(input_dir=str(input_dir.resolve()), input_format=input_format,
                              weight_file=None if weight_file is None else str(Path(weight_file).resolve()),
                              b_fbi=float(shell), l_max=l_max, fbi_method=fbi_method,
                              Da=Da, D0=D0, field_reference=float(field_reference)))


def build_operators(data, *, use_gpu=False):
    """Prepare float32 padded data once and reuse both operators and TV G."""
    shape = data['metadata']['Dim']
    float32_like = np.empty(0, dtype=np.float32)
    field = padarray_post(data['local_field'], shape, float32_like)
    roi = padarray_post(data['mask_phase'], shape, float32_like)
    n_meso = padarray_post(data['N_meso'], shape, float32_like)
    weight = data['weight']
    if weight is not None:
        # Ones outside the input volume retain the original padded residual.
        weight = np.pad(weight, tuple((0, n) for n in shape), constant_values=1)[None, ...]
    args = dict(N=tuple(2*n for n in shape), voxel_size=data['metadata']['Voxel_size'],
                B0_dirs=data['metadata']['B0_dir'], mask_roi=roi,
                mask_ref=None, weight=weight, use_gpu=use_gpu)
    muqsm = MuQSMLinop(N_meso=n_meso, **args)
    qsm = QSMLinop(**args)
    y = prepare_y_for_lsq(field, muqsm, remove_mean=True, apply_weight=True)
    G = sp.linop.FiniteDifference(muqsm.ishape, axes=(0, 1, 2))
    return {'muqsm': muqsm, 'qsm': qsm}, y, G


def validate_solver_settings(lambdas, cg_max_iter, tv_max_iter, tv_max_cg_iter, tol, rho):
    if any(not np.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError('Regularization parameters must be finite and nonnegative.')
    if any(not isinstance(n, (int, np.integer)) or n < 1
           for n in (cg_max_iter, tv_max_iter, tv_max_cg_iter)):
        raise ValueError('Solver iteration limits must be positive integers.')
    if not np.isfinite(tol) or tol < 0 or not np.isfinite(rho) or rho <= 0:
        raise ValueError('tol must be finite and nonnegative; rho must be positive and finite.')


def solve_case(A, y, penalty, lambda_value, G, *, cg_max_iter=200, tv_max_iter=300,
               tv_max_cg_iter=10, tol=1e-6, rho=1, show_progress=False):
    """An independent zero start for every solve, including selected reruns."""
    x0 = A.xp.zeros(A.ishape, dtype=A.xp.float32)
    if penalty == 'l2':
        solver = sp.app.LinearLeastSquares(
            A, y, x=x0, lamda=float(lambda_value), solver='ConjugateGradient',
            max_iter=cg_max_iter, tol=tol, show_pbar=show_progress,
        )
    elif penalty == 'tv':
        proxg = sp.prox.L1Reg(G.oshape, float(lambda_value))
        solver = sp.app.LinearLeastSquares(
            A, y, x=x0, G=G, proxg=proxg, lamda=0, solver='ADMM',
            rho=rho, max_iter=tv_max_iter, max_cg_iter=tv_max_cg_iter,
            tol=tol, show_pbar=False,
        )
    else:
        raise ValueError('penalty must be l2 or tv.')
    # SigPy's zero-gradient soft threshold evaluates 0/0 before selecting zero.
    # The original implementation does this too; check the actual result below.
    with np.errstate(divide='ignore', invalid='ignore'):
        if penalty == 'tv' and show_progress:
            # SigPy 0.1.27's least-squares bar expects a residual that ADMM
            # does not provide. Its generic App shows iterations instead.
            sp.app.App(solver.alg, show_pbar=True).run()
            chi = solver.x
        else:
            chi = solver.run()
    if not bool(_to_numpy(A.xp.all(A.xp.isfinite(chi)))):
        raise RuntimeError('Solver returned nonfinite values; inspect inputs and solver settings.')
    return chi, int(solver.alg.iter)


def crop_and_reference(chi, data):
    """Copy only the unpadded volume to CPU, then apply the original CSF sign."""
    slices = tuple(slice(0, n) for n in data['metadata']['Dim'])
    cropped = _to_numpy(chi[slices]).copy()
    return -(cropped - np.mean(cropped[data['mask_csf']]))


def run_exvivo_reconstruction(
    input_dir, output_dir, *, input_format='mat', weight_file=None, nifti_reference=None,
    output_formats=('npz', 'mat', 'nifti'), use_gpu=False,
    b_fbi=None, l_max=6, fbi_method='corrected', Da=None, D0=2e-3,
    qsm_l2_lambda=8e-3, muqsm_l2_lambda=8e-3,
    qsm_tv_lambda=1e-5, muqsm_tv_lambda=1e-5,
    cg_max_iter=200, tv_max_iter=300, tv_max_cg_iter=10, tol=1e-6, rho=1,
    make_plots=True, show_plots=None, figure_dpi=400, save_dpi=800,
    slice_xs=None, slice_indices=None, show_progress=False,
):
    """Run four reconstructions and export maps, figures, and a JSON run record.

    NIfTI needs a matching reference image. Without one, NPZ/MAT exports continue.
    L-curve selections are never read automatically; supply the four lambdas.
    Returns independent cropped arrays under maps, and settings/saved_files.
    """
    lambdas = dict(qsm_l2=qsm_l2_lambda, muqsm_l2=muqsm_l2_lambda,
                   qsm_tv=qsm_tv_lambda, muqsm_tv=muqsm_tv_lambda)
    validate_solver_settings(lambdas.values(), cg_max_iter, tv_max_iter, tv_max_cg_iter, tol, rho)
    _get_backend(use_gpu)
    meta = read_metadata(input_dir, input_format)
    if input_format == 'nifti':
        phase_reference = _nifti_path(input_dir, 'Phase_map')
        if nifti_reference is None:
            nifti_reference = phase_reference
    exports = prepare_exports(output_formats, nifti_reference, meta['Dim'], meta['Voxel_size'])
    if input_format == 'nifti' and exports['reference'] is not None:
        from .io import _nibabel
        if not np.allclose(exports['reference'].affine, _nibabel().load(str(phase_reference)).affine,
                           rtol=1e-5, atol=1e-5):
            raise ValueError('Reference NIfTI must share the input phase spatial geometry.')
    data = prepare_exvivo_data(input_dir, input_format=input_format, weight_file=weight_file, b_fbi=b_fbi,
                              l_max=l_max, fbi_method=fbi_method, Da=Da, D0=D0)
    del data['FBI']  # Reconstruction only needs N_meso; release the tensor/eigen volumes.
    operators, y, G = build_operators(data, use_gpu=use_gpu)
    solver_settings = dict(cg_max_iter=cg_max_iter, tv_max_iter=tv_max_iter,
                           tv_max_cg_iter=tv_max_cg_iter, tol=tol, rho=rho,
                           show_progress=show_progress)
    maps, iterations = {}, {}
    for case in CASES:
        model, penalty = case.split('_')
        print(f'Reconstructing {case}: lambda={lambdas[case]:.6g}')
        chi, iterations[case] = solve_case(operators[model], y, penalty, lambdas[case], G, **solver_settings)
        maps[MAP_KEYS[case]] = crop_and_reference(chi, data)
        del chi
    output_dir = Path(output_dir)
    saved_files = save_maps(maps, output_dir, exports)
    # Release padded solver data before allocating publication-sized figures.
    del operators, y, G
    if make_plots:
        from .plotting import save_reconstruction_figures
        saved_files += save_reconstruction_figures(
            maps, data['mask_phase'], output_dir, show_plots=show_plots,
            figure_dpi=figure_dpi, save_dpi=save_dpi,
            slice_xs=slice_xs, slice_indices=slice_indices,
        )
    record_path = output_dir / 'exvivo_run.json'
    saved_files.append(str(record_path.resolve()))
    settings = dict(**data['settings'], metadata=meta, lambdas=lambdas, **solver_settings,
                    use_gpu=use_gpu, output_formats=exports['formats'],
                    nifti_reference=None if nifti_reference is None else str(Path(nifti_reference).resolve()),
                    nifti_status=exports['nifti_status'], make_plots=make_plots,
                    figure_dpi=figure_dpi, save_dpi=save_dpi, show_plots=show_plots,
                    slice_xs=slice_xs, slice_indices=slice_indices,
                    reference_convention='-(cropped_chi - mean(cropped_chi[mask_csf]))',
                    susceptibility_units='ppm')
    save_record(record_path, dict(settings=settings, iterations=iterations, saved_files=saved_files))
    print(f'Reconstruction outputs saved in {output_dir.resolve()}')
    return dict(maps=maps, settings=settings, iterations=iterations, saved_files=saved_files)
