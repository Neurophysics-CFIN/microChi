"""FBI spherical harmonics and scatter matrices for diffusion MRI."""
import numpy as np
from scipy.special import sph_harm_y, eval_legendre, hyp1f1, gamma, factorial

# spherical harmonics expansion of dMRI data
def cart2sph(bvecs):
    """Cartesian unit vectors to physics-convention spherical angles.
    params:
    bvecs: (N, 3) array of Cartesian coordinates (x, y, z)
    returns:
    theta: (N,) array of polar angles
    phi: (N,) array of azimuthal angles
    """
    x, y, z = bvecs.T
    # theta is the polar angle from +z, phi is the azimuth in the xy-plane.
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    return theta, phi

def get_real_sym_sh(l, m, theta, phi):
    """
    Computes the real, symmetric spherical harmonic basis function.
    Uses scipy.special.sph_harm_y(degree, order, polar_angle, azimuthal_angle).
    params:
    l: degree of the spherical harmonic
    m: order of the spherical harmonic
    theta: polar angle (colatitude) in radians
    phi: azimuthal angle (longitude) in radians
    returns:
    The real, symmetric spherical harmonic value for the given (l, m) at angles (theta, phi).
    """
    # Get the complex spherical harmonic value
    Y_complex = sph_harm_y(l, m, theta, phi)
    
    # Convert to Real SH Basis (Descoteaux/Tournier convention)
    if m < 0:
        # For negative m, use the imaginary part
        return np.sqrt(2) * Y_complex.imag
    elif m == 0:
        # For m=0, the function is already strictly real
        return Y_complex.real
    else:
        # For positive m, use the real part
        return np.sqrt(2) * Y_complex.real
    
def build_sh_basis_matrix(bvecs, l_max=6):
    """
    Builds the N x K design matrix B for a given set of gradient vectors.
    params:
    bvecs: (N, 3) array of gradient directions in Cartesian coordinates
    l_max: maximum degree of the spherical harmonics
    returns:
    B: (N, K) design matrix
    """
    if l_max % 2 != 0:
        raise ValueError("l_max / degree must be even.")
    
    theta, phi = cart2sph(bvecs)
    N = len(bvecs)
    
    # Calculate how many coefficients we will have
    K = int((l_max + 1) * (l_max + 2) / 2)
    B = np.zeros((N, K))
    
    col_idx = 0
    # Loop over only EVEN degrees (l = 0, 2, 4, 6...)
    for l in range(0, l_max + 1, 2):
        # Loop over phase orders (m = -l to l)
        for m in range(-l, l + 1):
            B[:, col_idx] = get_real_sym_sh(l, m, theta, phi)
            col_idx += 1
            
    return B




# calculate fODF coefficients from spherical harmonics expansion of dMRI data
def g_factor(l, b, D):
    """
    Calculate the g_l(b D) factor for the given degree l, b-value, and diffusivity D.
    This factor accounts for the diffusion weighting in the spherical harmonics expansion.
    params:
    l: degree of the spherical harmonic
    b: b-value of the diffusion weighting
    D: diffusivity (either Da or D0 depending on context)
    returns:
    g_l: the g_l(b D) factor
    """
    bD = float(b) * float(D)
    l = int(l)
    g_2l = factorial(l) * (bD)**(l + 0.5) / gamma(2.0 * l + 1.5) * hyp1f1(l + 0.5, 2.0 * l + 1.5, -bD)

    return g_2l

def compute_fODF_coefficients(B, dMRI_signal):
    """
    Computes fODF coefficients for millions of voxels simultaneously.
    params:
    B: (N, K) design matrix
    dMRI_signal: (V, N) array of signals (V = number of masked voxels)
    """
    # Calculate the Pseudo-Inverse of the Basis Matrix ONCE
    B_inv = np.linalg.pinv(B)
    # Tensor dot product: (V, N) @ (N, K) -> (V, K)
    sig_sh = dMRI_signal @ B_inv.T
    return sig_sh

def calc_fbi(sig_sh, l_max, b, Da=None, D0=None, method='exact'):
    """
    Calculate the fODF coefficients from the spherical harmonics expansion of dMRI data.
    params:
    sig_sh: (K,) array of spherical harmonics coefficients for the dMRI signal
    l_max: maximum degree of the spherical harmonics used in the expansion
    b: b-value of the diffusion weighting
    Da: diffusivity of the anisotropic component (optional, used for 'exact' method)
    D0: positive diffusivity required for the 'corrected' method
    method: method to calculate fODF coefficients ('exact' or 'corrected')
    returns:
    fODF_coeffs: (K,) array of fODF coefficients
    """
    if not np.isfinite(b) or b <= 0:
        raise ValueError('The FBI b-value must be positive and finite.')
    diffusivity = Da if method == 'exact' else D0
    if method not in ('exact', 'corrected'):
        raise ValueError("method must be 'exact' or 'corrected'.")
    if diffusivity is None or not np.isfinite(diffusivity) or diffusivity <= 0:
        name = 'Da' if method == 'exact' else 'D0'
        raise ValueError(f"'{method}' method requires positive, finite {name}.")
    out = np.zeros_like(sig_sh)
    idx = 0

    # ---- 'exact': ground-truth transform with known Da ---------------
    if method == 'exact':
        if Da is None:
            raise ValueError("'exact' method requires Da (the true intra-axonal diffusivity).")
        sqrt_bDa_over_pi = np.sqrt(b * Da / np.pi)

        for l in range(0, l_max + 1, 2):
            Pl0 = eval_legendre(l, 0.0)
            base = sqrt_bDa_over_pi / (2.0 * np.pi * Pl0)
            factor = base / g_factor(l/2, b,  Da)
            for _ in range(-l, l + 1):
                out[:, idx] = sig_sh[:, idx] * factor
                idx += 1

    elif method == 'corrected':
        a00 = sig_sh[:, 0]
        g0_bD0 = g_factor(0, b, D0)

        for l in range(0, l_max + 1, 2):
            Pl0 = eval_legendre(l, 0.0)
            factor = g0_bD0 / (np.sqrt(4.0 * np.pi) * Pl0 * a00 * g_factor(l/2, b, D0))
            for _ in range(-l, l + 1):
                out[:, idx] = sig_sh[:, idx] * factor
                idx += 1

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'exact' or 'corrected'.")
    
    out[~np.isfinite(out)] = 0.0
    return out

# calculate the scatter matrix for each voxel from the fODF coefficients, which is needed for muQSM reconstruction
def real_to_complex_l0l2(c_real):
    """
    Convert real SH coefficients to complex SH coefficients for l=0 and l=2.
    This is necessary because the scatter-matrix formulas are written in the complex SH basis.
    params:
    c_real: (V,K) array of real SH coefficients for l=0 and l=2
    returns:
    out: dictionary mapping (l, m) to complex SH coefficients
    """
    # The scatter-matrix formulas are written in the complex SH basis, so this
    # helper converts the package's real-basis coefficients into that form.

    def sh_order_index(l_max):
        """Ordered (l, m) pairs for even-order SH basis."""
        # Diffusion ODF/signal models are antipodally symmetric, so we keep only even SH degrees.
        return [(l, m) for l in range(0, l_max + 1, 2) for m in range(-l, l + 1)]

    lm_idx = {(l, m): i for i, (l, m) in enumerate(sh_order_index(2))}
    cR = c_real
    
    def C(l, m):
        if m == 0:
            return cR[:, lm_idx[(l, 0)]] + 0j
        
        am = abs(m)
        rp = cR[:, lm_idx[(l, am)]]   # Real part coefficient
        rm = cR[:, lm_idx[(l, -am)]]  # Imaginary part coefficient
        
        if m > 0:
            # CORRECT: No phase factor here
            return 1.0 / np.sqrt(2) * (rp - 1j * rm)
        elif m < 0:
            # CORRECT: Phase factor (-1)**|m| belongs here
            return ((-1) ** am) / np.sqrt(2) * (rp + 1j * rm)

    out = {}
    for l in (0, 2):
        for m in range(-l, l + 1):
            out[(l, m)] = C(l, m)
    return out


def scatter_matrix(c_complex):
    """
    Build (V,3,3) scatter matrices from complex SH coefficients (l=0..2).
    params:
    c_complex: dictionary mapping (l, m) to complex SH coefficients for l=0 and l=2
    returns:
    (A)    A: Vx3 real scatter matrix with unit trace
    """
    c00 = c_complex[(0, 0)].real
    c20 = c_complex[(2, 0)]
    c21 = c_complex[(2, 1)]
    c2m1 = c_complex[(2, -1)]
    c22 = c_complex[(2, 2)]
    c2m2 = c_complex[(2, -2)]

    r30 = np.sqrt(30.0)
    r6 = np.sqrt(6.0)

    a11 = r30 / 3 * c00 - r6 / 3 * c20 + c22 + c2m2
    a22 = r30 / 3 * c00 - r6 / 3 * c20 - c22 - c2m2
    a33 = r30 / 3 * c00 + 2 * r6 / 3 * c20
    a12 = 1j * (c22 - c2m2)
    a13 = -c21 + c2m1
    a23 = -1j * (c21 + c2m1)

    # Initialize a (V, 3, 3) matrix block
    V_len = len(c00)
    A = np.zeros((V_len, 3, 3), dtype=np.complex128)
    
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = a11, a12, a13
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = np.conj(a12), a22, a23
    A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = np.conj(a13), np.conj(a23), a33

    # Safely normalize trace
    valid_mask = (np.isfinite(c00)) & (~np.isclose(c00, 0.0))
    A[valid_mask] = A[valid_mask] / (c00[valid_mask, np.newaxis, np.newaxis] * r30)
    A[~valid_mask] = 0

    # Force real and symmetric
    A = A.real
    A = 0.5 * (A + A.transpose(0, 2, 1))

    return A


# wrapper function to calculate the scatter matrix from raw diffusion MRI signal
def Calc_FBI_SctMtx(signal, bvals, bvecs, mask, b_FBI, l_max=6, Da=None, D0=None, method='exact'):
    """
    Calculate the scatter matrix from raw diffusion MRI signal.
    params:
    signal: (X,Y,Z,N) 4D array of diffusion MRI signal values for a whole brain volume (X, Y, Z, N) where N is the number of gradient directions
    bvals: (N,) array of b-values for each gradient direction
    bvecs: (N, 3) array of gradient directions in Cartesian coordinates
    mask: (X,Y,Z) array; nonzero voxels are included, others remain zero
    b_FBI: b-value to use for the FBI calculation
    l_max: maximum degree of the spherical harmonics used in the expansion
    method: method to calculate fODF coefficients ('exact' or 'corrected')
    Da: diffusivity of the anisotropic component (optional, used for 'exact' method)
    D0: diffusivity of the isotropic component (optional, used for 'corrected' method)
    returns:
    Dictionary with ScatterMatrix, EigenValues, and EigenVectors volumes.
    """
    signal, bvals, bvecs, mask = map(np.asarray, (signal, bvals, bvecs, mask))
    if signal.ndim != 4 or mask.shape != signal.shape[:3]:
        raise ValueError('signal must be (X,Y,Z,K), with a matching 3D mask.')
    if bvals.size != signal.shape[-1] or bvecs.shape != (signal.shape[-1], 3):
        raise ValueError('bvals and bvecs must match the number of diffusion measurements.')
    if not isinstance(l_max, (int, np.integer)) or l_max < 2 or l_max % 2:
        raise ValueError('Scatter matrices require an even l_max >= 2.')
    if not np.all(np.isfinite(bvals)) or not np.all(np.isfinite(bvecs)):
        raise ValueError('bvals and bvecs must be finite.')
    # Check conditional diffusivity requirements before the large signal selection.
    calc_fbi(np.ones((1, (l_max + 1)*(l_max + 2)//2)), l_max, b_FBI,
             Da=Da, D0=D0, method=method)
    dim = signal.shape
    num_voxels = np.prod(dim[:3])

    # Extract ONLY the target b-shell for FBI calculation (no b0)
    bvals_flat = bvals.flatten()
    idx_fbi = np.where(np.round(bvals_flat) == np.round(b_FBI))[0]
    if not len(idx_fbi):
        raise ValueError('The requested FBI b-shell is absent from bvals.')
    if not np.allclose(np.linalg.norm(bvecs[idx_fbi], axis=1), 1, atol=1e-4):
        raise ValueError('FBI gradient directions must be unit vectors.')
    if not np.any(mask > 0):
        raise ValueError('The FBI mask is empty.')

    # Slice the Signal and B-vectors
    sig_masked = signal[mask > 0][:, idx_fbi]
    if not np.all(np.isfinite(sig_masked)):
        raise ValueError('Masked diffusion signals must be finite.')
    bvecs_shell = bvecs[idx_fbi, :]

    print("--> Vectorizing brain mask...")
    mask_flat = mask.flatten() > 0

    print("--> Building SH Basis and fitting coefficients...")
    B = build_sh_basis_matrix(bvecs_shell, l_max)
    sig_sh = compute_fODF_coefficients(B, sig_masked)

    print("--> Deconvolving fODF...")
    fODF_coeffs = calc_fbi(sig_sh, l_max, b_FBI, Da=Da, D0=D0, method=method)

    print("--> Constructing Mesoscopic Scatter Matrices...")
    c_complex = real_to_complex_l0l2(fODF_coeffs)
    A_masked = scatter_matrix(c_complex)

    print("--> Performing Stacked Eigendecomposition...")
    # np.linalg.eigh perfectly handles stacks of symmetric matrices (V, 3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(A_masked)

    # Optional: eigh returns ascending order. We often want descending (Principal e-vec first).
    idx = np.argsort(eigenvalues, axis=-1)[..., ::-1]
    eigenvalues = np.take_along_axis(eigenvalues, idx, axis=-1)
    eigenvectors = np.take_along_axis(eigenvectors, idx[:, np.newaxis, :], axis=-1)

    print("--> Reconstructing 3D Volume...")
    # Create empty arrays to hold the full brain volume of scatter matrices
    A_full = np.zeros((num_voxels, 3, 3), dtype=np.float32)
    eigval_full = np.zeros((num_voxels, 3), dtype=np.float32)
    eigvec_full = np.zeros((num_voxels, 3, 3), dtype=np.float32)

    # Fill back only the masked voxels
    A_full[mask_flat] = A_masked
    eigval_full[mask_flat] = eigenvalues
    eigvec_full[mask_flat] = eigenvectors

    # Reshape back to (X, Y, Z, 3, 3) for scatter matrices  
    OUT = {
        'ScatterMatrix': A_full.reshape(dim[0], dim[1], dim[2], 3, 3),
        'EigenValues': eigval_full.reshape(dim[0], dim[1], dim[2], 3),
        'EigenVectors': eigvec_full.reshape(dim[0], dim[1], dim[2], 3, 3)
    }

    print("FBI calculation complete.")
    return OUT
