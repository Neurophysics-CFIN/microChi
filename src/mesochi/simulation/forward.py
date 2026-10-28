# functions needed for simulation
import numpy as np
from scipy.fft import fftn, ifftn, fftfreq

# dipole kernel
def B_dip_k(N, voxel_size, B0_dir):
    """
    Memory-Optimized Dipole Kernel in k-space.
    Args:
        N: tuple of ints, the size of the 3D volume (Nx, Ny, Nz)
        voxel_size: tuple of floats, the size of each voxel in mm (dx, dy, dz)
        B0_dir: tuple of floats, the direction of the main magnetic field (Bx, By, Bz)
    Returns:
        kernel: 3D numpy array of shape (Nx, Ny, Nz), the dipole kernel in k-space
    """
    # FIX: normalise B0_dir so a non-unit vector doesn't scale the kernel
    B0_dir = np.asarray(B0_dir, dtype=np.float64)
    if B0_dir.shape != (3,) or not np.all(np.isfinite(B0_dir)) or np.linalg.norm(B0_dir) == 0:
        raise ValueError("B0_dir must be a finite, nonzero vector of shape (3,).")
    B0_dir = B0_dir / np.linalg.norm(B0_dir)

    kx = np.fft.fftfreq(N[0], d=voxel_size[0]).astype(np.float32).reshape(N[0], 1, 1)
    ky = np.fft.fftfreq(N[1], d=voxel_size[1]).astype(np.float32).reshape(1, N[1], 1)
    kz = np.fft.fftfreq(N[2], d=voxel_size[2]).astype(np.float32).reshape(1, 1, N[2])

    k_squared = kx**2 + ky**2 + kz**2
    kernel = kx * np.float32(B0_dir[0]) + ky * np.float32(B0_dir[1]) + kz * np.float32(B0_dir[2])
    kernel **= 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        kernel /= k_squared
    kernel *= -1.0
    kernel += np.float32(1.0 / 3.0)
    kernel[0, 0, 0] = 0.0

    del k_squared

    return kernel


# downsampling function
def downsample_3d_fft(image, window_size, factor, agg_func=np.mean, dtype=np.float32):
    """Downsample a 3D image by averaging (or summing) over a window_size^3 box
    centred on each factor^3 block.
    Args:
        image: 3D numpy array of shape (Nx, Ny, Nz), the input image to downsample.
        window_size: int, the size of the uniform kernel.
        factor: int, the downsampling factor.
        agg_func: np.mean / 'mean' or np.sum / 'sum'.
        dtype: numpy dtype, the data type of the output (default is np.float32).
    Returns:
        3D numpy array of shape (Nx/factor, Ny/factor, Nz/factor), the downsampled image.
    """
    if not isinstance(factor, (int, np.integer)) or factor <= 0:
        raise ValueError("Downsampling factor must be a positive integer.")
    # FIX: any other agg_func used to leave the kernel all zeros -> silent all-zero output
    if agg_func in (np.mean, 'mean'):
        mode = 'mean'
    elif agg_func in (np.sum, 'sum'):
        mode = 'sum'
    else:
        raise ValueError("agg_func must be np.mean/'mean' or np.sum/'sum'.")

    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if image.ndim != 3 or window_size > min(image.shape):
        raise ValueError("image must be 3D and window_size must fit its dimensions.")

    in_shape = image.shape
    if any(n % factor for n in in_shape):
        raise ValueError(f"Image shape {in_shape} is not divisible by factor {factor}.")
    out_shape = tuple(n // factor for n in in_shape)

    # Fast, exact path: window == factor is a plain block average (no FFT, no extra
    # full-size complex arrays, no FFT round-off in volume fractions)
    if window_size == factor:
        blocks = image.reshape(out_shape[0], factor, out_shape[1], factor, out_shape[2], factor)
        out = blocks.sum(axis=(1, 3, 5), dtype=np.float64)
        if mode == 'mean':
            out /= factor ** 3
        return out.astype(dtype)

    # General path: FFT (circular) convolution with a box kernel
    kernel_padded = np.zeros(in_shape, dtype=dtype)
    kernel_padded[0:window_size, 0:window_size, 0:window_size] = (
        1.0 / (window_size ** 3) if mode == 'mean' else 1.0)

    shift_val = -(window_size // 2)
    kernel_shifted = np.roll(kernel_padded, shift=(shift_val, shift_val, shift_val), axis=(0, 1, 2))
    del kernel_padded

    kernel_k = fftn(kernel_shifted)
    del kernel_shifted

    img_k = fftn(image)
    img_k *= kernel_k
    del kernel_k

    convolved = ifftn(img_k).real
    del img_k

    # FIX: sample by `factor` (was `window_size`) so the output is N/factor, with each
    # sample at the centre of its factor^3 block
    start_idx = factor // 2 - 1 if factor > 1 else 0
    idx_x = np.arange(start_idx, in_shape[0], factor)[:out_shape[0]]
    idx_y = np.arange(start_idx, in_shape[1], factor)[:out_shape[1]]
    idx_z = np.arange(start_idx, in_shape[2], factor)[:out_shape[2]]

    return convolved[np.ix_(idx_x, idx_y, idx_z)].astype(dtype)


# create meshgrid function
def create_meshgrid(N, voxel_size=np.array([1.0, 1.0, 1.0]), space='real', endpoint=False,
                    sparse=False, dtype=np.float64):
    """
    Create a 3D meshgrid for given dimensions N and voxel size.
    Args:
        N: tuple of grid dimensions (Nx, Ny, Nz)
        voxel_size: size of each voxel in each dimension
        space: 'real' or 'k'
        endpoint: kept for backward compatibility; ignored (coordinates are exact)
        sparse: if True, return broadcastable (Nx,1,1), (1,Ny,1), (1,1,Nz) arrays
                instead of full N^3 grids (saves ~all the memory)
        dtype: dtype of the returned grids
    Returns:
        coordinate grids for x, y, and z axes
        x_grid, y_grid, z_grid
    """
    if space == 'real':
        # FIX: linspace(..., endpoint=False) gave spacing (N-1)/N * voxel_size.
        # Coordinates are now exactly (i - N/2) * voxel_size, i = 0..N-1.
        x = (np.arange(N[0]) - N[0] / 2) * voxel_size[0]
        y = (np.arange(N[1]) - N[1] / 2) * voxel_size[1]
        z = (np.arange(N[2]) - N[2] / 2) * voxel_size[2]
    elif space == 'k':
        x = 2 * np.pi * fftfreq(N[0], d=voxel_size[0])
        y = 2 * np.pi * fftfreq(N[1], d=voxel_size[1])
        z = 2 * np.pi * fftfreq(N[2], d=voxel_size[2])
    else:
        raise ValueError("space must be 'real' or 'k'")

    x_grid, y_grid, z_grid = np.meshgrid(x.astype(dtype), y.astype(dtype), z.astype(dtype),
                                         indexing='ij', sparse=sparse)
    return x_grid, y_grid, z_grid


# generate Fibonacci hemisphere function
def generate_fibonacci_hemisphere(num_orients):
    """
    Generate uniformly distributed unit vectors on a hemisphere using Fibonacci spiral.
    Args:
        num_orients: int, number of orientations (unit vectors) to generate
    Returns:
        list of 3D numpy arrays, shape [x, y, z]
    """
    b0_vecs = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle (approximately 137.5 degrees)

    for i in range(num_orients):
        z = 1 - (i / float(num_orients - 1)) if num_orients > 1 else 1.0
        radius = np.sqrt(1 - z * z)
        theta_golden = phi * i

        x = np.cos(theta_golden) * radius
        y = np.sin(theta_golden) * radius
        b0_vecs.append(np.array([x, y, z]))

    return b0_vecs

# simulate local fields with multiple orientations measurements without noise
def simulate_local_fields(N, voxel_size, chi, water_vol, mask_water, dir_samp, dwn_samp_f, win_size, dtype=np.float32):
    """Simulate local fields with multiple orientations measurements without noise.
    Args:
        N: tuple of ints, the size of the 3D volume (Nx, Ny, Nz).
        voxel_size: tuple of floats, the size of each voxel in mm (dx, dy, dz).
        chi: 3D numpy array of shape (Nx, Ny, Nz), the susceptibility distribution.
        water_vol: low-resolution water voxel counts, shape N / dwn_samp_f.
        mask_water: 3D numpy array of shape (Nx, Ny, Nz), the water mask.
        dir_samp: 2D numpy array of shape (n_dir, 3), the sampling directions.
        dwn_samp_f: int, the downsampling factor.
        win_size: int, the size of the uniform kernel for downsampling.
        dtype: numpy dtype, the data type of the output (default is np.float32).
    Returns:
        List of low-resolution 3D local-field maps, one per orientation."""
    if not isinstance(dwn_samp_f, (int, np.integer)) or dwn_samp_f <= 0:
        raise ValueError("dwn_samp_f must be a positive integer.")
    water_vol = np.asarray(water_vol)
    expected_shape = tuple(int(n) // dwn_samp_f for n in N)
    if water_vol.shape != expected_shape or not np.all(np.isfinite(water_vol)) or np.any(water_vol < 0):
        raise ValueError("water_vol must contain finite, nonnegative low-resolution water counts.")
    LoFds = []
    for dir in dir_samp:
        dk = (B_dip_k(N, voxel_size, dir)).astype(dtype)
        LoFd_hr = ifftn(fftn(chi) * dk).real
        LoFd_hr *= mask_water

        LoFd_lr = downsample_3d_fft(LoFd_hr, window_size=win_size, factor=dwn_samp_f, agg_func=np.sum, dtype=dtype)
        # No measurable water signal where the water volume is zero.
        LoFd_lr = np.divide(LoFd_lr, water_vol, out=np.zeros_like(LoFd_lr), where=water_vol > 0)
        LoFd    = LoFd_lr - np.mean(LoFd_lr)
        LoFds.append(LoFd)
    return LoFds


# simulate complex noise added to magnitude and phase images
def _wrap_phase(phi):
    """Wrap phase to [-pi, pi)."""
    return (phi + np.pi) % (2 * np.pi) - np.pi

def _field_to_phase_scale(B0, TE, gamma_hz, LoFd_unit):
    """Return the conversion factor from local field to phase [rad]."""
    if LoFd_unit == "ppm":
        return 2 * np.pi * gamma_hz * B0 * TE * 1e-6
    if LoFd_unit == "relative":
        return 2 * np.pi * gamma_hz * B0 * TE
    if LoFd_unit == "hz":
        return 2 * np.pi * TE
    raise ValueError("LoFd_unit must be 'ppm', 'relative', or 'hz'.")

def simulate_complex_noise(
    LoFd,
    mask,
    B0=1.5,
    TE=4.5e-3,
    SNR=10,
    gamma_hz=42.57747892e6,
    LoFd_unit="ppm",
    magnitude_mode="uniform",
    water_volume=None,
    seed=0,
    background=0.5,
    dtype=np.float32,
):
    """
    Simulate single-echo GRE images from a clean local field map with complex noise.

    Model
    -----
    phi = phase_scale * LoFd
    S_clean = magnitude_clean * exp(1j * phi)
    S_noisy = S_clean + sigma_complex * (n_real + 1j * n_imag)

    n_real and n_imag are independent standard normal noise. sigma_complex
    is the noise standard deviation in each channel, defined as the mean
    clean magnitude inside the mask divided by SNR.

    Parameters
    ----------
    LoFd : ndarray
        Clean local field map, in the unit specified by LoFd_unit.
    mask : ndarray
        Phantom / reconstruction mask; nonzero values indicate the interior.
    B0 : float
        Main field strength [T].
    TE : float
        Echo time [s].
    SNR : float
        Mean clean magnitude inside the mask / per-channel noise std.
    gamma_hz : float
        Proton gyromagnetic ratio [Hz/T].
    LoFd_unit : {"ppm", "relative", "hz"}
        "ppm": local field in ppm; "relative": dimensionless field / B0;
        "hz": frequency offset in Hz.
    magnitude_mode : {"uniform", "water_fraction"}
        "uniform": magnitude is 1 inside the mask and background outside.
        "water_fraction": normalize water_volume by its maximum inside
        the mask and set the magnitude outside the mask to zero.
    water_volume : ndarray or None
        Water volume map; required for magnitude_mode="water_fraction".
    seed : int or None
        Random seed; None generates a different realization on each call.
    background : float
        Clean magnitude outside the mask in uniform mode.
    dtype : np.float32 or np.float64
        Output precision; complex arrays use complex64 or complex128.

    Returns
    -------
    result : dict
        Clean and noisy magnitude, phase, and complex signal; circular
        phase error; per-channel noise std; predicted phase-noise std;
        and LoFd_noisy_wrapped in the same unit as LoFd.

    Notes
    -----
    sigma_phi_pred = sigma_complex / magnitude_clean is a high-SNR
    approximation. It is set to infinity where the clean magnitude is zero.
    LoFd_noisy_wrapped is obtained from wrapped phase; no unwrapping is done.
    Noise is added throughout the image, including outside the mask.
    """
    if SNR <= 0:
        raise ValueError("SNR must be positive.")

    dtype = np.dtype(dtype)
    complex_dtype = np.result_type(dtype, np.complex64)
    LoFd = np.asarray(LoFd)
    mask = np.asarray(mask, dtype=bool)

    # 1. Define the clean magnitude image.
    if magnitude_mode == "uniform":
        magnitude_clean = np.where(mask, 1.0, background).astype(dtype)
    elif magnitude_mode == "water_fraction":
        if water_volume is None:
            raise ValueError("water_volume is required for water_fraction mode.")
        magnitude_clean = np.array(water_volume, dtype=dtype, copy=True)
        magnitude_clean /= np.max(magnitude_clean[mask]) + 1e-12
        magnitude_clean[~mask] = 0.0
    else:
        raise ValueError("magnitude_mode must be 'uniform' or 'water_fraction'.")

    # 2. Convert the local field to phase.
    phase_scale = _field_to_phase_scale(B0, TE, gamma_hz, LoFd_unit)
    phi_clean = (phase_scale * LoFd).astype(dtype, copy=False)

    # 3. Construct the clean complex signal.
    phasor_clean = np.exp(1j * phi_clean).astype(complex_dtype, copy=False)
    S_clean = magnitude_clean * phasor_clean

    # 4. Add independent Gaussian noise to the real and imaginary channels.
    sigma_complex = float(np.mean(magnitude_clean[mask])) / SNR
    rng = np.random.default_rng(seed)
    S_noisy = np.empty(LoFd.shape, dtype=complex_dtype)
    S_noisy.real = rng.standard_normal(LoFd.shape)
    S_noisy.imag = rng.standard_normal(LoFd.shape)
    S_noisy *= sigma_complex
    S_noisy += S_clean

    # 5. Extract magnitude, wrapped phase, and circular phase error.
    magnitude_noisy = np.abs(S_noisy)
    phi_clean_wrapped = _wrap_phase(phi_clean)
    phi_noisy_wrapped = np.angle(S_noisy)
    phase_error = np.angle(S_noisy * np.conj(phasor_clean))

    # 6. Convert the wrapped noisy phase back to local field.
    LoFd_noisy_wrapped = phi_noisy_wrapped / phase_scale

    # High-SNR phase-noise approximation; undefined at zero clean magnitude.
    sigma_phi_pred = np.full_like(magnitude_clean, np.inf)
    np.divide(
        sigma_complex,
        magnitude_clean,
        out=sigma_phi_pred,
        where=magnitude_clean > 0,
    )

    return {
        "magnitude_clean": magnitude_clean,
        "magnitude_noisy": magnitude_noisy,
        "phi_clean": phi_clean,
        "phi_clean_wrapped": phi_clean_wrapped,
        "phi_noisy_wrapped": phi_noisy_wrapped,
        "phase_error": phase_error,
        "S_clean": S_clean,
        "S_noisy": S_noisy,
        "sigma_complex": sigma_complex,
        "sigma_phi_pred": sigma_phi_pred,
        "LoFd_noisy_wrapped": LoFd_noisy_wrapped,
    }
