"""MATLAB inputs and NumPy, MATLAB, and reference-based NIfTI exports."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat, savemat


def load_required_mat(path, names):
    """Read only the named MATLAB variables and explain missing inputs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required input file is missing: {path}')
    arrays = loadmat(path, variable_names=names)
    missing = [name for name in names if name not in arrays]
    if missing:
        raise ValueError(f'{path.name} is missing variables: {", ".join(missing)}')
    return {name: arrays[name] for name in names}


def _validate_metadata(raw):
    names = ('Dim', 'B0_dir', 'Voxel_size', 'gammaB0', 'TEs')
    try:
        meta = {name: np.asarray(raw[name], dtype=float).ravel() for name in names}
    except (TypeError, ValueError) as error:
        raise ValueError('Metadata values must be numeric.') from error
    dims, spacing, direction = meta['Dim'], meta['Voxel_size'], meta['B0_dir']
    if (dims.shape != (3,) or not np.all(np.isfinite(dims))
            or np.any(dims < 1) or np.any(dims != np.floor(dims))):
        raise ValueError('Dim must contain three positive integers.')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Voxel_size must contain three positive spacings in mm.')
    if (direction.shape != (3,) or not np.all(np.isfinite(direction))
            or not np.isfinite(np.linalg.norm(direction)) or np.linalg.norm(direction) == 0):
        raise ValueError('B0_dir must be a finite, nonzero 3-vector.')
    if meta['gammaB0'].size != 1 or not np.isfinite(meta['gammaB0'][0]) or meta['gammaB0'][0] <= 0:
        raise ValueError('gammaB0 must be a positive, finite scalar.')
    tes = meta['TEs']
    if len(tes) < 2 or not np.all(np.isfinite(tes)) or np.any(np.diff(tes) <= 0):
        raise ValueError('TEs must contain at least two increasing echo times in ms.')
    meta['Dim'] = tuple(int(v) for v in dims)
    meta['gammaB0'] = float(meta['gammaB0'][0])
    return meta


def _nifti_path(input_dir, stem):
    paths = [Path(input_dir) / (stem + suffix) for suffix in ('.nii', '.nii.gz')]
    found = [path for path in paths if path.is_file()]
    if len(found) != 1:
        raise ValueError(f'Provide exactly one of {paths[0].name} or {paths[1].name}.')
    return found[0]


def _nibabel():
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError('NIfTI input or export requires: python -m pip install -e ".[nifti]"') from error
    return nib


def _check_nifti_geometry(image, name):
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError(f'{name} must specify spatial units in mm.')
    if not int(image.header['qform_code']) and not int(image.header['sform_code']):
        raise ValueError(f'{name} needs a defined qform or sform.')
    affine = image.affine
    if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-15:
        raise ValueError(f'{name} must have a finite, invertible spatial transform.')
    axes = affine[:3, :3]
    axes = axes / np.linalg.norm(axes, axis=0)
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError(f'{name} must have orthogonal voxel axes.')


def nifti_inputs(input_dir):
    """Open the five registered NIfTI inputs, checking geometry before loading data."""
    nib = _nibabel()
    stems = ('DWI_FBI', 'Phase_map', 'mask_phase', 'mask_wm', 'mask_csf')
    paths = {name: _nifti_path(input_dir, name) for name in stems}
    images = {name: nib.load(str(path)) for name, path in paths.items()}
    phase = images['Phase_map']
    if len(phase.shape) != 3:
        raise ValueError('Phase_map must be a 3D NIfTI volume.')
    for name, image in images.items():
        _check_nifti_geometry(image, name)
        expected_ndim = 4 if name == 'DWI_FBI' else 3
        if len(image.shape) != expected_ndim or image.shape[:3] != phase.shape:
            raise ValueError(f'{name} must match the phase spatial dimensions.')
        if not np.allclose(image.affine, phase.affine, rtol=1e-5, atol=1e-5):
            raise ValueError(f'{name} must share the phase NIfTI spatial geometry.')
    return paths, images


def read_metadata(input_dir, input_format='mat'):
    names = ('Dim', 'B0_dir', 'Voxel_size', 'gammaB0', 'TEs')
    if input_format == 'mat':
        return _validate_metadata(load_required_mat(Path(input_dir) / 'meta_info.mat', names))
    if input_format != 'nifti':
        raise ValueError("input_format must be 'mat' or 'nifti'.")
    path = Path(input_dir) / 'meta_info.json'
    with path.open(encoding='utf-8') as handle:
        raw = json.load(handle)
    for name in ('B0_dir', 'gammaB0', 'TEs', 'bvec_convention'):
        if name not in raw:
            raise ValueError(f'meta_info.json is missing {name}.')
    if raw['bvec_convention'] not in ('image_axes', 'fsl'):
        raise ValueError("bvec_convention must be 'image_axes' or 'fsl'.")
    _, images = nifti_inputs(input_dir)
    phase = images['Phase_map']
    raw['Dim'] = phase.shape
    raw['Voxel_size'] = _nibabel().affines.voxel_sizes(phase.affine)
    meta = _validate_metadata(raw)
    meta['bvec_convention'] = raw['bvec_convention']
    return meta


def load_nifti_data(input_dir, metadata):
    """Load NIfTI volumes and diffusion tables in image-array coordinates."""
    _, images = nifti_inputs(input_dir)
    values = {name: np.asanyarray(image.dataobj) for name, image in images.items()}
    bvals = np.loadtxt(Path(input_dir) / 'bval.txt', ndmin=1).ravel()
    bvecs = np.loadtxt(Path(input_dir) / 'bvec.txt', ndmin=2)
    count = values['DWI_FBI'].shape[-1]
    if bvecs.shape == (3, count) and count != 3:
        bvecs = bvecs.T
    elif bvecs.shape != (count, 3) or count == 3:
        raise ValueError('bvec.txt must be K x 3 or 3 x K, with unambiguous K.')
    if metadata['bvec_convention'] == 'fsl' and np.linalg.det(images['DWI_FBI'].affine[:3, :3]) > 0:
        bvecs = bvecs.copy()
        bvecs[:, 0] *= -1
    values.update(bval=bvals, bvec=bvecs)
    return values


def load_weight(weight_file, shape):
    """None means uniform weights; no map is inferred from masks or signal."""
    if weight_file is None:
        return None
    path = Path(weight_file)
    if path.suffix.lower() == '.npy':
        weight = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.mat':
        weight = load_required_mat(path, ('weight',))['weight']
    else:
        raise ValueError('weight_file must be .npy or .mat containing weight.')
    if (weight.shape != tuple(shape) or not np.isrealobj(weight)
            or not np.all(np.isfinite(weight)) or np.any(weight < 0)):
        raise ValueError('weight must match Dim and contain finite, nonnegative real values.')
    weight = np.asarray(weight, dtype=np.float32)
    if not np.all(np.isfinite(weight)):
        raise ValueError('weight values must fit float32.')
    return weight


def prepare_exports(output_formats, nifti_reference, shape, voxel_size):
    """Validate exports before expensive preparation. NIfTI is optional."""
    if isinstance(output_formats, str):
        output_formats = (output_formats,)
    formats = tuple(dict.fromkeys(output_formats))
    if not formats or any(fmt not in ('npz', 'mat', 'nifti') for fmt in formats):
        raise ValueError('output_formats must select npz, mat, and/or nifti.')
    reference = None
    status = 'not requested'
    if 'nifti' in formats:
        if nifti_reference is None:
            status = 'skipped: no matching reference NIfTI supplied'
            print(f'NIfTI export {status}.')
        else:
            try:
                import nibabel as nib
            except ImportError as error:
                raise RuntimeError('NIfTI export requires: python -m pip install -e ".[nifti]"') from error
            reference = nib.load(str(nifti_reference))
            if reference.shape != tuple(shape):
                raise ValueError('Reference NIfTI must have the same 3D shape and array ordering as Dim.')
            if reference.header.get_xyzt_units()[0] != 'mm':
                raise ValueError('Reference NIfTI must specify spatial units in mm.')
            if not np.allclose(nib.affines.voxel_sizes(reference.affine), voxel_size, rtol=1e-5, atol=1e-6):
                raise ValueError('Reference NIfTI voxel spacing must match Voxel_size.')
            qform, qcode = reference.get_qform(coded=True)
            sform, scode = reference.get_sform(coded=True)
            if not qcode and not scode:
                raise ValueError('Reference NIfTI needs a defined qform or sform spatial transform.')
            for transform, code in ((qform, qcode), (sform, scode)):
                if code and (not np.all(np.isfinite(transform))
                             or abs(np.linalg.det(transform[:3, :3])) < 1e-15):
                    raise ValueError('Reference NIfTI spatial transforms must be finite and invertible.')
            if not np.all(np.isfinite(reference.affine)) or abs(np.linalg.det(reference.affine[:3, :3])) < 1e-15:
                raise ValueError('Reference NIfTI must have a finite, invertible spatial transform.')
            status = 'enabled'
    if 'nifti' in formats and reference is None and not any(f in formats for f in ('npz', 'mat')):
        raise ValueError('NIfTI-only output requires a matching reference NIfTI.')
    return dict(formats=formats, reference=reference, nifti_status=status)


def save_maps(maps, output_dir, exports, *, stem='exvivo_reconstruction'):
    """Save independent, cropped maps without changing ordering or units."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if 'npz' in exports['formats']:
        path = output_dir / f'{stem}.npz'
        np.savez_compressed(path, **maps)
        paths.append(str(path.resolve()))
    if 'mat' in exports['formats']:
        path = output_dir / f'{stem}.mat'
        savemat(path, maps, do_compression=True)
        paths.append(str(path.resolve()))
    reference = exports['reference']
    if reference is not None:
        import nibabel as nib
        for name, volume in maps.items():
            header = reference.header.copy()
            header.set_data_dtype(np.float32)
            header.set_slope_inter(1, 0)
            header['cal_min'], header['cal_max'] = float(volume.min()), float(volume.max())
            header['descrip'] = b'MesoChi susceptibility (ppm), CSF referenced'
            image = nib.Nifti1Image(volume, reference.affine, header=header)
            image.set_qform(*reference.get_qform(coded=True))
            image.set_sform(*reference.get_sform(coded=True))
            path = output_dir / f'{name}.nii.gz'
            nib.save(image, str(path))
            paths.append(str(path.resolve()))
    return paths


def save_record(path, record):
    """Write a readable run record; never used to override reconstruction settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value.resolve())
        raise TypeError(f'Cannot serialize {type(value).__name__}')
    path.write_text(json.dumps(record, indent=2, default=serializable, allow_nan=False) + '\n', encoding='utf-8')
