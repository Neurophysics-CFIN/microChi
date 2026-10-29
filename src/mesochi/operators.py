"""QSM operators with float32/complex64 or float64/complex128 coefficients.

The constructor dtype controls coefficient and kernel precision. Forward and
adjoint results preserve the higher precision of the input and operator.
"""
#%% imports and shared CPU / GPU helpers
# QSM and cylindrical, spherical, and composite muQSM linear operators.
import numpy as np
import sigpy as sp
from scipy.fft import fftn as np_fftn, ifftn as np_ifftn

try:
    import cupy as cp
    from cupyx.scipy.fft import fftn as cu_fftn, ifftn as cu_ifftn
except ImportError:
    cp = None


def _get_array_module(x):
    return cp.get_array_module(x) if cp is not None else np


def _get_backend(use_gpu):
    if not use_gpu:
        return np
    if cp is None:
        raise RuntimeError("GPU execution requires CuPy and a compatible CUDA installation.")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device found.")
    except Exception as error:
        raise RuntimeError("GPU execution requested, but a CUDA device is unavailable.") from error
    return cp


def _validate_directions(B0_dirs):
    directions = _to_numpy(B0_dirs)
    if directions.shape == (3,):
        directions = directions[None, :]
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] == 0:
        raise ValueError("B0 directions must have shape (3,) or (K, 3), with K > 0.")
    if not np.all(np.isfinite(directions)) or np.any(np.linalg.norm(directions, axis=1) == 0):
        raise ValueError("B0 directions must be finite, nonzero vectors.")


def _get_module_and_fft(x):
    xp = _get_array_module(x)
    if xp is cp:
        return xp, cu_fftn, cu_ifftn
    return xp, np_fftn, np_ifftn


def _to_numpy(x):
    if _get_array_module(x) is cp:
        return cp.asnumpy(x)
    return np.asarray(x)


def _operator_dtypes(dtype):
    """Resolve the supported real precision and its matching complex dtype."""
    try:
        real_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise TypeError("dtype must be float32 or float64.") from error
    if real_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64.")
    return real_dtype, np.result_type(real_dtype, np.complex64)


def Dipole_Kernel(N, voxel_size, B0_dir, xp=np, dtype=None):
    """Dipole kernel in k-space, matching np.fft / cupy.fft layout (no fftshift)."""
    if dtype is None:
        dtype = xp.float32

    _validate_directions(B0_dir)
    if _to_numpy(B0_dir).shape != (3,):
        raise ValueError("B0_dir must be a single vector of shape (3,).")
    B0 = xp.asarray(B0_dir, dtype=dtype)
    B0 = B0 / xp.linalg.norm(B0)

    fftfreq = xp.fft.fftfreq
    kx = (2*xp.pi * fftfreq(N[0], voxel_size[0])).astype(dtype)
    ky = (2*xp.pi * fftfreq(N[1], voxel_size[1])).astype(dtype)
    kz = (2*xp.pi * fftfreq(N[2], voxel_size[2])).astype(dtype)

    kx_, ky_, kz_ = kx[:,None,None], ky[None,:,None], kz[None,None,:]
    k2    = kx_**2 + ky_**2 + kz_**2
    kdotB = kx_*B0[0] + ky_*B0[1] + kz_*B0[2]

    with np.errstate(divide='ignore', invalid='ignore'):
        D = xp.asarray(1.0/3.0, dtype=dtype) - kdotB**2 / k2
    D[0, 0, 0] = 0
    return D.astype(dtype)


#%% standard QSM operator and adjoint
class QSMLinop(sp.linop.Linop):
    """
    Standard QSM linear operator without mesoscopic correction.

    Forward model:

        chi
        -> reference subtraction
        -> ROI masking
        -> dipole convolution
        -> field mean removal
        -> weighting

    Mathematical form:

        y_i = W_i P D_i R chi

    where:
        R : reference subtraction + ROI mask
        D : dipole convolution
        P : remove spatial mean
        W : data weight

    Input shape:
        chi.shape = (Nx, Ny, Nz)

    Output shape:
        y.shape = (K, Nx, Ny, Nz)

    K = number of B0 orientations.

    dtype selects float32 or float64 coefficients and matching complex kernels.
    Outputs preserve the higher precision of the input and operator.
    """

    def __init__(
        self,
        N,
        voxel_size,
        B0_dirs,
        mask_roi,
        mask_ref=None,
        weight=None,
        use_gpu=False,
        dtype=np.float32,
    ):

        self.vol_shape = tuple(int(v) for v in N)
        self.num_vox = int(np.prod(self.vol_shape))
        self.dtype, self.complex_dtype = _operator_dtypes(dtype)

        # B0 directions
        B0_dirs_np = _to_numpy(B0_dirs).astype(self.dtype)
        _validate_directions(B0_dirs_np)

        if B0_dirs_np.shape == (3,):
            B0_dirs_np = B0_dirs_np[None, :]
        elif B0_dirs_np.ndim != 2 or B0_dirs_np.shape[1] != 3:
            raise ValueError(
                f"B0_dirs shape {B0_dirs_np.shape} must be (3,) or (K, 3)."
            )

        self.B0_dirs = B0_dirs_np
        self.num_orients = B0_dirs_np.shape[0]

        # Device
        self.xp = _get_backend(use_gpu)
        xp = self.xp

        # SigPy shapes
        ishape = self.vol_shape
        oshape = (self.num_orients, *self.vol_shape)

        super().__init__(
            oshape=oshape,
            ishape=ishape,
            repr_str="QSMLinop",
        )

        # ROI mask
        self.mask_roi = xp.asarray(mask_roi, dtype=self.dtype)

        if self.mask_roi.shape != self.vol_shape:
            raise ValueError(
                f"mask_roi shape {self.mask_roi.shape} != vol_shape {self.vol_shape}."
            )

        # Reference mask
        if mask_ref is None:
            self.mask_ref = 1.0 - self.mask_roi
        else:
            self.mask_ref = xp.asarray(mask_ref, dtype=self.dtype)

        if self.mask_ref.shape != self.vol_shape:
            raise ValueError(
                f"mask_ref shape {self.mask_ref.shape} != vol_shape {self.vol_shape}."
            )

        self.N_ref = xp.sum(self.mask_ref, dtype=self.dtype)
        n_ref_val = float(cp.asnumpy(self.N_ref) if xp is cp else self.N_ref)

        if n_ref_val == 0:
            raise ValueError("mask_ref is empty: sum(mask_ref) == 0.")

        # Data weight
        if weight is None:
            self.weight = xp.ones(
                (self.num_orients, *self.vol_shape),
                dtype=self.dtype,
            )
        else:
            self.weight = xp.asarray(weight, dtype=self.dtype)

            # Allow single 3D weight for one or multiple orientations
            if self.weight.shape == self.vol_shape:
                self.weight = xp.broadcast_to(
                    self.weight,
                    (self.num_orients, *self.vol_shape),
                ).copy()

        if self.weight.shape != (self.num_orients, *self.vol_shape):
            raise ValueError(
                f"weight shape {self.weight.shape} must be "
                f"{(self.num_orients, *self.vol_shape)}."
            )

        # Dipole kernels
        self.dip_k_list = [
            xp.asarray(
                Dipole_Kernel(self.vol_shape, voxel_size, B0_dir=b0, dtype=self.dtype),
                dtype=self.complex_dtype,
            )
            for b0 in self.B0_dirs
        ]

    def _check_device(self, arr):
        xp = _get_array_module(arr)

        if xp is not self.xp:
            expected = "CuPy/GPU" if self.xp is cp else "NumPy/CPU"
            received = "CuPy/GPU" if xp is cp else "NumPy/CPU"
            raise TypeError(
                f"Input is on {received}, but this Linop was built for {expected}."
            )

    def _apply(self, input):
        """
        Forward operation:

            chi -> weighted dipole field

        Input:
            input.shape = self.vol_shape

        Output:
            output.shape = (num_orients, *self.vol_shape)
        """
        self._check_device(input)

        xp, fftn, ifftn = _get_module_and_fft(input)

        chi = input.reshape(self.vol_shape)

        # Reference subtraction
        chi_ref = xp.sum(chi * self.mask_ref) / self.N_ref

        # ROI-restricted susceptibility
        chi_roi = (chi - chi_ref) * self.mask_roi

        Fchi = fftn(chi_roi)

        out = xp.empty(
            (self.num_orients, *self.vol_shape),
            dtype=xp.result_type(input.real.dtype, self.dtype),
        )

        for i in range(self.num_orients):
            # Standard QSM dipole forward model only
            field = ifftn(Fchi * self.dip_k_list[i]).real

            # Remove spatial mean, same as your muQSM operator
            field = field - xp.mean(field)

            # Apply data weight
            out[i] = self.weight[i] * field

        return out

    def _adjoint_linop(self):
        return QSMAdjointLinop(self)


class QSMAdjointLinop(sp.linop.Linop):
    """
    Adjoint of QSMLinop.

    Input shape:
        (K, Nx, Ny, Nz)

    Output shape:
        (Nx, Ny, Nz)
    """

    def __init__(self, A: QSMLinop):
        self.A = A

        super().__init__(
            oshape=A.ishape,
            ishape=A.oshape,
            repr_str="QSMAdjointLinop",
        )

    def _apply(self, input):
        A = self.A
        A._check_device(input)

        xp, fftn, ifftn = _get_module_and_fft(input)

        y = input.reshape((A.num_orients, *A.vol_shape))

        acc = xp.zeros(
            A.vol_shape,
            dtype=xp.result_type(input.real.dtype, A.dtype),
        )

        for i in range(A.num_orients):
            yi = y[i]

            # Adjoint of weighting
            yw = A.weight[i] * yi

            # Adjoint of mean-removal projection
            yw = yw - xp.mean(yw)

            # Adjoint dipole convolution
            adj_dipole = ifftn(
                xp.conj(A.dip_k_list[i]) * fftn(yw)
            ).real

            # Adjoint of reference subtraction + ROI mask
            g = A.mask_roi * adj_dipole

            ref_corr = A.mask_ref * xp.sum(g) / A.N_ref

            acc += g - ref_corr

        return acc

    def _adjoint_linop(self):
        return self.A


#%% cylindrical muQSM operator and adjoint
class MuQSMLinop(sp.linop.Linop):
    """Cylindrical μQSM with shared 3-D or per-orientation 4-D weights.

    dtype selects float32 or float64 coefficients and matching complex kernels.
    Outputs preserve the higher precision of the input and operator.
    """
    def __init__(
        self,
        N,
        voxel_size,
        B0_dirs,
        mask_roi,
        N_meso,
        mask_ref=None,
        weight=None,
        use_gpu=False,
        dtype=np.float32,
    ):
    
        self.vol_shape = tuple(N)
        self.num_vox   = int(np.prod(self.vol_shape))
        self.dtype, self.complex_dtype = _operator_dtypes(dtype)

        # ── B0 directions ────────────────────────────────────────────────────
        B0_dirs_np = _to_numpy(B0_dirs).astype(self.dtype)
        _validate_directions(B0_dirs_np)
        if B0_dirs_np.shape == (3,):
            B0_dirs_np = B0_dirs_np[None, :]
        elif B0_dirs_np.ndim != 2 or B0_dirs_np.shape[1] != 3:
            raise ValueError(
                f"B0_dirs shape {B0_dirs_np.shape} must be (3,) or (K, 3)."
            )
        self.B0_dirs    = B0_dirs_np
        self.num_orients = B0_dirs_np.shape[0]

        # ── device ───────────────────────────────────────────────────────────
        self.xp = _get_backend(use_gpu)
        xp = self.xp

        # ── sigpy shapes ─────────────────────────────────────────────────────
        ishape = self.vol_shape
        oshape = (self.num_orients, *self.vol_shape)
        super().__init__(oshape=oshape, ishape=ishape, repr_str="MuQSMLinop")

        # ── masks ─────────────────────────────────────────────────────────────
        self.mask_roi = xp.asarray(mask_roi, dtype=self.dtype)
        if self.mask_roi.shape != self.vol_shape:
            raise ValueError(
                f"mask_roi shape {self.mask_roi.shape} != vol_shape {self.vol_shape}."
            )

        self.mask_ref = (
            1.0 - self.mask_roi
            if mask_ref is None
            else xp.asarray(mask_ref, dtype=self.dtype)
        )
        if self.mask_ref.shape != self.vol_shape:
            raise ValueError(
                f"mask_ref shape {self.mask_ref.shape} != vol_shape {self.vol_shape}."
            )

        self.N_ref = xp.sum(self.mask_ref, dtype=self.dtype)
        n_ref_val  = float(cp.asnumpy(self.N_ref) if xp is cp else self.N_ref)
        if n_ref_val == 0:
            raise ValueError("mask_ref is empty: sum(mask_ref) == 0.")

        # ── N_meso ───────────────────────────────────────────────────────────
        self.N_meso = xp.asarray(N_meso, dtype=self.dtype)
        if self.N_meso.shape == self.vol_shape:
            self.N_meso = xp.broadcast_to(
                self.N_meso, (self.num_orients, *self.vol_shape)
            ).copy()
        elif self.N_meso.shape != (self.num_orients, *self.vol_shape):
            raise ValueError(
                f"N_meso shape {self.N_meso.shape} must be "
                f"{self.vol_shape} or {(self.num_orients, *self.vol_shape)}."
            )

        # ── weight ───────────────────────────────────────────────────────────
        if weight is None:
            self.weight = xp.ones((self.num_orients, *self.vol_shape), dtype=self.dtype)
        else:
            self.weight = xp.asarray(weight, dtype=self.dtype)
            if self.weight.shape == self.vol_shape:
                self.weight = xp.broadcast_to(
                    self.weight, (self.num_orients, *self.vol_shape)
                ).copy()
        if self.weight.shape != (self.num_orients, *self.vol_shape):
            raise ValueError(
                f"weight shape {self.weight.shape} must be "
                f"{self.vol_shape} or {(self.num_orients, *self.vol_shape)}."
            )

        # ── dipole kernels ───────────────────────────────────────────────────
        self.dip_k_list = [
            xp.asarray(
                Dipole_Kernel(self.vol_shape, voxel_size, B0_dir=b0, dtype=self.dtype),
                dtype=self.complex_dtype,
            )
            for b0 in self.B0_dirs
        ]

    def _check_device(self, arr):
        xp = _get_array_module(arr)
        if xp is not self.xp:
            expected = "CuPy/GPU" if self.xp is cp else "NumPy/CPU"
            received = "CuPy/GPU" if xp is cp else "NumPy/CPU"
            raise TypeError(
                f"Input is on {received}, but this Linop was built for {expected}."
            )

    def _apply(self, input):
        self._check_device(input)
        xp, fftn, ifftn = _get_module_and_fft(input)

        x     = input.reshape(self.vol_shape)
        x_ref = xp.sum(x * self.mask_ref) / self.N_ref
        x_roi = (x - x_ref) * self.mask_roi
        Fx    = fftn(x_roi)

        out = xp.empty(
            (self.num_orients, *self.vol_shape),
            dtype=xp.result_type(input.real.dtype, self.dtype),
        )
        for i in range(self.num_orients):
            field   = ifftn(Fx * self.dip_k_list[i]).real + self.N_meso[i] * x_roi
            out[i]  = self.weight[i] * (field - xp.mean(field))

        return out

    def _adjoint_linop(self):
        return MuQSMAdjointLinop(self)


class MuQSMAdjointLinop(sp.linop.Linop):
    def __init__(self, A: MuQSMLinop):
        self.A = A
        super().__init__(oshape=A.ishape, ishape=A.oshape, repr_str="MuQSMAdjointLinop")

    def _apply(self, input):
        A = self.A
        A._check_device(input)
        xp, fftn, ifftn = _get_module_and_fft(input)

        y   = input.reshape((A.num_orients, *A.vol_shape))
        acc = xp.zeros(A.vol_shape, dtype=xp.result_type(input.real.dtype, A.dtype))

        for i in range(A.num_orients):
            yw = A.weight[i] * y[i]
            yw = yw - xp.mean(yw)

            adj_dipole = ifftn(xp.conj(A.dip_k_list[i]) * fftn(yw)).real
            adj_meso   = A.N_meso[i] * yw

            g        = A.mask_roi * (adj_dipole + adj_meso)
            ref_corr = A.mask_ref * xp.sum(g) / A.N_ref
            acc     += g - ref_corr

        return acc

    def _adjoint_linop(self):
        return self.A


#%% spherical muQSM operator and adjoint
class MuQSMLinop_S(sp.linop.Linop):
    """
    Spherical muQSM forward operator with two unknowns, chi^s and u = chi^s * lambda^c.

    Forward model (orientation i = 1..K):

        y_i = W_i P [ D_i R chi^s - N_meso_i M u ]

    where
        R        : reference subtraction (mean of chi^s in mask_ref) + ROI mask
        M        : ROI mask
        D_i      : dipole convolution for B0 direction i
        N_meso_i : mesoscopic coefficient for orientation i (given by the caller),
                   e.g. -1/2 (cos^2 theta_i - 1/3)
        P        : remove spatial mean
        W_i      : data weight

    Input shape:   (2, Nx, Ny, Nz)   -> input[0] = chi^s, input[1] = u
    Output shape:  (K, Nx, Ny, Nz)

    Parameters
    ----------
    N : tuple of 3 ints
        Volume shape.
    voxel_size : tuple of 3 floats
        Voxel size [mm].
    B0_dirs : (3,) or (K, 3) array
        B0 directions (normalised internally).
    mask_roi : (Nx, Ny, Nz) array
        Reconstruction mask for chi^s.
    N_meso : (Nx, Ny, Nz) or (K, Nx, Ny, Nz) array
        Coefficient multiplying u, per orientation. A 3D array is used for all
        orientations (same as MuQSMLinop). Setting N_meso = 0 in a region (e.g. CSF)
        removes u there: it then has no effect on the data and stays 0 in a
        least-squares solve started from zero.
    mask_ref : (Nx, Ny, Nz) array or None
        Reference region for chi^s (e.g. CSF). Default: 1 - mask_roi.
    weight : (Nx, Ny, Nz) or (K, Nx, Ny, Nz) array or None
        Data weight. Default: ones.
    use_gpu : bool
    dtype : numpy dtype
        float32 or float64 coefficient precision, with matching complex kernels.
        Outputs preserve the higher precision of the input and operator.
    """

    def __init__(
        self,
        N,
        voxel_size,
        B0_dirs,
        mask_roi,
        N_meso,
        mask_ref=None,
        weight=None,
        use_gpu=False,
        dtype=np.float32,
    ):
        self.vol_shape = tuple(int(v) for v in N)
        self.num_vox   = int(np.prod(self.vol_shape))
        self.dtype, self.complex_dtype = _operator_dtypes(dtype)

        # ── B0 directions ────────────────────────────────────────────────────
        B0_dirs_np = _to_numpy(B0_dirs).astype(self.dtype)
        _validate_directions(B0_dirs_np)
        if B0_dirs_np.shape == (3,):
            B0_dirs_np = B0_dirs_np[None, :]
        elif B0_dirs_np.ndim != 2 or B0_dirs_np.shape[1] != 3:
            raise ValueError(f"B0_dirs shape {B0_dirs_np.shape} must be (3,) or (K, 3).")
        B0_dirs_np /= np.linalg.norm(B0_dirs_np, axis=1, keepdims=True)
        self.B0_dirs     = B0_dirs_np
        self.num_orients = B0_dirs_np.shape[0]

        # ── device ───────────────────────────────────────────────────────────
        self.xp = _get_backend(use_gpu)
        xp = self.xp

        # ── sigpy shapes ─────────────────────────────────────────────────────
        ishape = (2, *self.vol_shape)
        oshape = (self.num_orients, *self.vol_shape)
        super().__init__(oshape=oshape, ishape=ishape, repr_str="MuQSMLinop_S")

        # ── masks ────────────────────────────────────────────────────────────
        self.mask_roi = xp.asarray(mask_roi, dtype=self.dtype)
        if self.mask_roi.shape != self.vol_shape:
            raise ValueError(f"mask_roi shape {self.mask_roi.shape} != vol_shape {self.vol_shape}.")

        self.mask_ref = (1.0 - self.mask_roi if mask_ref is None
                         else xp.asarray(mask_ref, dtype=self.dtype))
        if self.mask_ref.shape != self.vol_shape:
            raise ValueError(f"mask_ref shape {self.mask_ref.shape} != vol_shape {self.vol_shape}.")

        self.N_ref = xp.sum(self.mask_ref, dtype=self.dtype)
        if float(_to_numpy(self.N_ref)) == 0:
            raise ValueError("mask_ref is empty: sum(mask_ref) == 0.")

        # ── N_meso ───────────────────────────────────────────────────────────
        self.N_meso = xp.asarray(N_meso, dtype=self.dtype)
        if self.N_meso.shape == self.vol_shape:
            self.N_meso = xp.broadcast_to(
                self.N_meso, (self.num_orients, *self.vol_shape)
            ).copy()
        elif self.N_meso.shape != (self.num_orients, *self.vol_shape):
            raise ValueError(
                f"N_meso shape {self.N_meso.shape} must be "
                f"{self.vol_shape} or {(self.num_orients, *self.vol_shape)}."
            )

        # ── weight ───────────────────────────────────────────────────────────
        if weight is None:
            self.weight = xp.ones((self.num_orients, *self.vol_shape), dtype=self.dtype)
        else:
            self.weight = xp.asarray(weight, dtype=self.dtype)
            if self.weight.shape == self.vol_shape:
                self.weight = xp.broadcast_to(self.weight, (self.num_orients, *self.vol_shape)).copy()
        if self.weight.shape != (self.num_orients, *self.vol_shape):
            raise ValueError(f"weight shape {self.weight.shape} must be "
                             f"{(self.num_orients, *self.vol_shape)}.")

        # ── dipole kernels ───────────────────────────────────────────────────
        self.dip_k_list = [
            xp.asarray(Dipole_Kernel(self.vol_shape, voxel_size, B0_dir=b0, dtype=self.dtype),
                       dtype=self.complex_dtype)
            for b0 in self.B0_dirs
        ]

    def _check_device(self, arr):
        xp = _get_array_module(arr)
        if xp is not self.xp:
            expected = "CuPy/GPU" if self.xp is cp else "NumPy/CPU"
            received = "CuPy/GPU" if xp is cp else "NumPy/CPU"
            raise TypeError(f"Input is on {received}, but this Linop was built for {expected}.")

    def _apply(self, input):
        self._check_device(input)
        xp, fftn, ifftn = _get_module_and_fft(input)

        x   = input.reshape((2, *self.vol_shape))
        chi = x[0]
        u   = x[1]

        # R chi^s : reference subtraction + ROI mask
        chi_ref = xp.sum(chi * self.mask_ref) / self.N_ref
        chi_roi = (chi - chi_ref) * self.mask_roi
        Fchi    = fftn(chi_roi)

        # M u
        u_roi = u * self.mask_roi

        out = xp.empty((self.num_orients, *self.vol_shape),
                       dtype=xp.result_type(input.real.dtype, self.dtype))
        for i in range(self.num_orients):
            field  = ifftn(Fchi * self.dip_k_list[i]).real - self.N_meso[i] * u_roi
            out[i] = self.weight[i] * (field - xp.mean(field))
        return out

    def _adjoint_linop(self):
        return MuQSMAdjointLinop_S(self)


class MuQSMAdjointLinop_S(sp.linop.Linop):
    """Adjoint of MuQSMLinop_S: (K, Nx, Ny, Nz) -> (2, Nx, Ny, Nz)."""

    def __init__(self, A: MuQSMLinop_S):
        self.A = A
        super().__init__(oshape=A.ishape, ishape=A.oshape, repr_str="MuQSMAdjointLinop_S")

    def _apply(self, input):
        A = self.A
        A._check_device(input)
        xp, fftn, ifftn = _get_module_and_fft(input)

        y     = input.reshape((A.num_orients, *A.vol_shape))
        dtype = xp.result_type(input.real.dtype, A.dtype)
        g_chi = xp.zeros(A.vol_shape, dtype=dtype)
        g_u   = xp.zeros(A.vol_shape, dtype=dtype)

        for i in range(A.num_orients):
            # adjoint of weighting and of mean removal (P is self-adjoint)
            yw = A.weight[i] * y[i]
            yw = yw - xp.mean(yw)

            # chi^s branch: adjoint dipole, then adjoint of R
            g_chi += ifftn(xp.conj(A.dip_k_list[i]) * fftn(yw)).real

            # u branch: adjoint of the pointwise multiplication by N_meso_i
            g_u -= A.N_meso[i] * yw

        # adjoint of R (applied once, it is linear): M g - mask_ref * sum(M g) / N_ref
        g_chi = A.mask_roi * g_chi
        g_chi = g_chi - A.mask_ref * xp.sum(g_chi) / A.N_ref
        g_u   = A.mask_roi * g_u

        return xp.stack([g_chi, g_u]).astype(dtype, copy=False)

    def _adjoint_linop(self):
        return self.A


#%% composite muQSM operator and adjoint
# Fields depend on total chi and chi^c - chi^s * lambda^c.
# The three component maps are not uniquely determined by the data alone.
class MuQSMLinop_CS(sp.linop.Linop):
    """
    General muQSM forward operator with three unknowns,
    u = chi^c, v = chi^s and w = chi^s * lambda^c.

    Forward model (orientation i = 1..K):

        y_i = W_i P [ D_i (R u + R v) + N_meso_i (R u - M w) ]

    where
        R        : reference subtraction (mean in mask_ref) + ROI mask,
                   applied to u and v separately
        M        : ROI mask
        D_i      : dipole convolution for B0 direction i
        N_meso_i : mesoscopic coefficient for orientation i (given by the caller),
                   N_meso_i = -1/2 (cos^2 theta_i - 1/3) reproduces the model above
        P        : remove spatial mean
        W_i      : data weight

    Input shape:   (3, Nx, Ny, Nz)   -> input[0] = u, input[1] = v, input[2] = w
    Output shape:  (K, Nx, Ny, Nz)

    Parameters
    ----------
    N : tuple of 3 ints
        Volume shape.
    voxel_size : tuple of 3 floats
        Voxel size [mm].
    B0_dirs : (3,) or (K, 3) array
        B0 directions (normalised internally).
    mask_roi : (Nx, Ny, Nz) array
        Reconstruction mask for u, v and w.
    N_meso : (Nx, Ny, Nz) or (K, Nx, Ny, Nz) array
        Mesoscopic coefficient per orientation. A 3D array is used for all
        orientations (same as MuQSMLinop). Set N_meso = 0 in CSF.
    mask_ref : (Nx, Ny, Nz) array or None
        Reference region for u and v (e.g. CSF). Default: 1 - mask_roi.
    weight : (Nx, Ny, Nz) or (K, Nx, Ny, Nz) array or None
        Data weight. Default: ones.
    use_gpu : bool
    dtype : numpy dtype
        float32 or float64 coefficient precision, with matching complex kernels.
        Outputs preserve the higher precision of the input and operator.
    """

    def __init__(
        self,
        N,
        voxel_size,
        B0_dirs,
        mask_roi,
        N_meso,
        mask_ref=None,
        weight=None,
        use_gpu=False,
        dtype=np.float32,
    ):
        self.vol_shape = tuple(int(v) for v in N)
        self.num_vox   = int(np.prod(self.vol_shape))
        self.dtype, self.complex_dtype = _operator_dtypes(dtype)

        # ── B0 directions ────────────────────────────────────────────────────
        B0_dirs_np = _to_numpy(B0_dirs).astype(self.dtype)
        _validate_directions(B0_dirs_np)
        if B0_dirs_np.shape == (3,):
            B0_dirs_np = B0_dirs_np[None, :]
        elif B0_dirs_np.ndim != 2 or B0_dirs_np.shape[1] != 3:
            raise ValueError(f"B0_dirs shape {B0_dirs_np.shape} must be (3,) or (K, 3).")
        B0_dirs_np /= np.linalg.norm(B0_dirs_np, axis=1, keepdims=True)
        self.B0_dirs     = B0_dirs_np
        self.num_orients = B0_dirs_np.shape[0]

        # ── device ───────────────────────────────────────────────────────────
        self.xp = _get_backend(use_gpu)
        xp = self.xp

        # ── sigpy shapes ─────────────────────────────────────────────────────
        ishape = (3, *self.vol_shape)
        oshape = (self.num_orients, *self.vol_shape)
        super().__init__(oshape=oshape, ishape=ishape, repr_str="MuQSMLinop_CS")

        # ── masks ────────────────────────────────────────────────────────────
        self.mask_roi = xp.asarray(mask_roi, dtype=self.dtype)
        if self.mask_roi.shape != self.vol_shape:
            raise ValueError(f"mask_roi shape {self.mask_roi.shape} != vol_shape {self.vol_shape}.")

        self.mask_ref = (1.0 - self.mask_roi if mask_ref is None
                         else xp.asarray(mask_ref, dtype=self.dtype))
        if self.mask_ref.shape != self.vol_shape:
            raise ValueError(f"mask_ref shape {self.mask_ref.shape} != vol_shape {self.vol_shape}.")

        self.N_ref = xp.sum(self.mask_ref, dtype=self.dtype)
        if float(_to_numpy(self.N_ref)) == 0:
            raise ValueError("mask_ref is empty: sum(mask_ref) == 0.")

        # ── N_meso ───────────────────────────────────────────────────────────
        self.N_meso = xp.asarray(N_meso, dtype=self.dtype)
        if self.N_meso.shape == self.vol_shape:
            self.N_meso = xp.broadcast_to(
                self.N_meso, (self.num_orients, *self.vol_shape)
            ).copy()
        elif self.N_meso.shape != (self.num_orients, *self.vol_shape):
            raise ValueError(
                f"N_meso shape {self.N_meso.shape} must be "
                f"{self.vol_shape} or {(self.num_orients, *self.vol_shape)}."
            )

        # ── weight ───────────────────────────────────────────────────────────
        if weight is None:
            self.weight = xp.ones((self.num_orients, *self.vol_shape), dtype=self.dtype)
        else:
            self.weight = xp.asarray(weight, dtype=self.dtype)
            if self.weight.shape == self.vol_shape:
                self.weight = xp.broadcast_to(self.weight, (self.num_orients, *self.vol_shape)).copy()
        if self.weight.shape != (self.num_orients, *self.vol_shape):
            raise ValueError(f"weight shape {self.weight.shape} must be "
                             f"{(self.num_orients, *self.vol_shape)}.")

        # ── dipole kernels ───────────────────────────────────────────────────
        self.dip_k_list = [
            xp.asarray(Dipole_Kernel(self.vol_shape, voxel_size, B0_dir=b0, dtype=self.dtype),
                       dtype=self.complex_dtype)
            for b0 in self.B0_dirs
        ]

    def _check_device(self, arr):
        xp = _get_array_module(arr)
        if xp is not self.xp:
            expected = "CuPy/GPU" if self.xp is cp else "NumPy/CPU"
            received = "CuPy/GPU" if xp is cp else "NumPy/CPU"
            raise TypeError(f"Input is on {received}, but this Linop was built for {expected}.")

    def _R(self, a, xp):
        """Reference subtraction + ROI mask."""
        return (a - xp.sum(a * self.mask_ref) / self.N_ref) * self.mask_roi

    def _R_adj(self, g, xp):
        """Adjoint of _R: M g - mask_ref * sum(M g) / N_ref."""
        g = self.mask_roi * g
        return g - self.mask_ref * xp.sum(g) / self.N_ref

    def _apply(self, input):
        self._check_device(input)
        xp, fftn, ifftn = _get_module_and_fft(input)

        x = input.reshape((3, *self.vol_shape))
        u_R   = self._R(x[0], xp)              # R u   (chi^c)
        v_R   = self._R(x[1], xp)              # R v   (chi^s)
        w_roi = x[2] * self.mask_roi           # M w   (chi^s * lambda^c)

        F_tot = fftn(u_R + v_R)                # total susceptibility
        meso  = u_R - w_roi                    # chi^c - chi^s * lambda^c

        out = xp.empty((self.num_orients, *self.vol_shape),
                       dtype=xp.result_type(input.real.dtype, self.dtype))
        for i in range(self.num_orients):
            field  = ifftn(F_tot * self.dip_k_list[i]).real + self.N_meso[i] * meso
            out[i] = self.weight[i] * (field - xp.mean(field))
        return out

    def _adjoint_linop(self):
        return MuQSMAdjointLinop_CS(self)


class MuQSMAdjointLinop_CS(sp.linop.Linop):
    """Adjoint of MuQSMLinop_CS: (K, Nx, Ny, Nz) -> (3, Nx, Ny, Nz)."""

    def __init__(self, A: MuQSMLinop_CS):
        self.A = A
        super().__init__(oshape=A.ishape, ishape=A.oshape, repr_str="MuQSMAdjointLinop_CS")

    def _apply(self, input):
        A = self.A
        A._check_device(input)
        xp, fftn, ifftn = _get_module_and_fft(input)

        y      = input.reshape((A.num_orients, *A.vol_shape))
        dtype  = xp.result_type(input.real.dtype, A.dtype)
        g_dip  = xp.zeros(A.vol_shape, dtype=dtype)    # adjoint of the dipole term
        g_meso = xp.zeros(A.vol_shape, dtype=dtype)    # adjoint of the meso term

        for i in range(A.num_orients):
            # adjoint of weighting and of mean removal (P is self-adjoint)
            yw = A.weight[i] * y[i]
            yw = yw - xp.mean(yw)

            g_dip  += ifftn(xp.conj(A.dip_k_list[i]) * fftn(yw)).real
            g_meso += A.N_meso[i] * yw

        g_u = A._R_adj(g_dip + g_meso, xp)            # u enters both terms through R
        g_v = A._R_adj(g_dip, xp)                     # v enters the dipole term through R
        g_w = -A.mask_roi * g_meso                    # w enters the meso term as -M w

        return xp.stack([g_u, g_v, g_w]).astype(dtype, copy=False)

    def _adjoint_linop(self):
        return self.A


#%% data preparation and solution helpers
def prepare_y_for_sigpy(LoFd, A, remove_mean=False, apply_weight=False):
    """
    Prepare an unweighted measured field for a QSM or μQSM operator.

    Normal reconstruction call sites should explicitly set remove_mean=True
    and apply_weight=True to match the operator: center each orientation over
    the spatial axes, then multiply by its weights. Defaults are retained for
    callers that supply data already prepared in either respect. Do not apply
    weighting twice. Data are converted to the operator's real dtype.

    Parameters
    ----------
    LoFd : array
        Measured field map.
        Accepted shapes:
            A.vol_shape
            (K, *A.vol_shape)
            flattened A.oshape
    A : QSMLinop, MuQSMLinop, MuQSMLinop_S, or MuQSMLinop_CS
        The operator that will consume the prepared field.
    remove_mean : bool
        Should be True because your forward operator removes spatial mean.
    apply_weight : bool
        Should be True if A uses weight in the forward model.

    Returns
    -------
    y : NumPy or CuPy array
        Shape = A.oshape
    """
    xp = A.xp

    y = xp.asarray(LoFd, dtype=A.dtype)

    # Case 1: single orientation field, shape = vol_shape
    if y.shape == A.vol_shape:
        y = y[None, ...]

    # Case 2: already shape = A.oshape
    elif y.shape == A.oshape:
        pass

    # Case 3: flattened vector
    elif y.size == int(np.prod(A.oshape)):
        y = y.reshape(A.oshape)

    else:
        raise ValueError(
            f"LoFd shape {y.shape} is incompatible with A.oshape={A.oshape}."
        )


    if remove_mean:
        spatial_axes = tuple(range(1, y.ndim))
        y = y - xp.mean(y, axis=spatial_axes, keepdims=True)

    if apply_weight:
        y = A.weight * y

    return y


def prepare_y_for_lsq(delta_B, A, remove_mean=True, apply_weight=True):
    """Ex vivo data preparation, preserving its mean-removal/weight defaults."""
    return prepare_y_for_sigpy(delta_B, A, remove_mean, apply_weight)


def split_solution_S(x, A):
    """Split a solution of shape A.ishape into (chi^s, u). chi^s is relative to mask_ref."""
    x = x.reshape(A.ishape)
    return x[0], x[1]


def split_solution_CS(x, A):
    """
    Split a solution of shape A.ishape into (u, v, w) = (chi^c, chi^s, chi^s*lambda^c).

    u and v are returned referenced to mask_ref inside the ROI (R u, R v), which is
    what the data see; w is returned inside the ROI.
    """
    xp = A.xp
    x = x.reshape(A.ishape)
    return A._R(x[0], xp), A._R(x[1], xp), x[2] * A.mask_roi


def lambda_from_solution(chi_s, u, mask, chi_min=1e-3):
    """
    lambda^c = u / chi^s, only where |chi^s| > chi_min (elsewhere 0).

    The ratio is ill-conditioned where chi^s ~ 0, e.g. the centre column of the
    gradient phantom, so check those voxels before interpreting them.
    """
    xp = _get_array_module(chi_s)
    ok = (xp.asarray(mask) > 0) & (xp.abs(chi_s) > chi_min)
    lam = xp.zeros_like(chi_s)
    lam[ok] = u[ok] / chi_s[ok]
    return lam, ok


