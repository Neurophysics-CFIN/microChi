"""Generate the cylindrical, spherical, and composite MesoChi phantoms."""
from pathlib import Path
import numpy as np
from . import forward as sf


def generate_phantoms(
    output_dir, *, n_hr=(1024, 1024, 1024),
    voxel_size_hr=(0.03125, 0.03125, 0.03125), downsample_factor=32,
    window_size=32, r_container=8.0, num_orientations=6,
    snr_levels=(np.inf, 100, 50, 30, 10), vol_frac_cyl=0.5,
    vol_frac_sph=0.05, contribution_cyl=0.9, b0_dir=(1, 0, 0),
    b0_t=3.0, te=5e-3, background=0.5, seed_cyl=42, seed_sph=100,
    dtype=np.float32,
):
    """Write the five NPZ datasets and return their paths keyed by filename.

    Spatial dimensions are high-resolution voxel counts; voxel sizes and
    container radius are in mm, susceptibility is in ppm, and TE is in seconds.
    The default 1024-cubed grid needs tens of GiB of RAM. For a small check use
    n_hr=(64,64,64), voxel_size_hr=(0.5,0.5,0.5), and both factors equal to 2.
    Legacy RandomState generators preserve the original seeded microstructure
    without changing the caller's global random state.
    """
    N_hr = np.asarray(n_hr)
    if N_hr.shape != (3,) or not np.issubdtype(N_hr.dtype, np.integer) or np.any(N_hr < 16):
        raise ValueError("n_hr must contain three integer dimensions of at least 16.")
    if not isinstance(downsample_factor, (int, np.integer)) or downsample_factor <= 0:
        raise ValueError("downsample_factor must be a positive integer.")
    if np.any(N_hr % downsample_factor):
        raise ValueError("n_hr must be divisible by downsample_factor.")
    if not isinstance(window_size, (int, np.integer)) or not 0 < window_size <= min(N_hr):
        raise ValueError("window_size must be a positive integer fitting the grid.")
    voxel_size_hr = np.asarray(voxel_size_hr, dtype=float)
    if voxel_size_hr.shape != (3,) or not np.all(np.isfinite(voxel_size_hr)) or np.any(voxel_size_hr <= 0):
        raise ValueError("voxel_size_hr must contain three positive finite values.")
    if not (0 < vol_frac_cyl < 1 and 0 <= vol_frac_sph <= 1 - vol_frac_cyl):
        raise ValueError("Volume fractions must leave a physical water compartment.")
    if vol_frac_sph == 0:
        raise ValueError("vol_frac_sph must be positive for the spherical phantom.")
    if not np.isfinite(r_container) or r_container <= 0:
        raise ValueError("r_container must be positive and finite.")
    if not 0 <= contribution_cyl <= 1:
        raise ValueError("contribution_cyl must be between zero and one.")
    if not isinstance(num_orientations, (int, np.integer)) or num_orientations < 1:
        raise ValueError("num_orientations must be a positive integer.")
    DATA_DIR = Path(output_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    B0_dir = np.asarray(b0_dir)
    dwn_samp_f = downsample_factor
    N_lr = N_hr // dwn_samp_f
    voxel_size_lr = voxel_size_hr * dwn_samp_f
    n_dir_samp = num_orientations
    SNRs = snr_levels
    d_type = dtype

    # Create meshgrid and container mask
    x_hr, y_hr, z_hr  = sf.create_meshgrid(N_hr, voxel_size_hr, 'real', sparse=True, dtype=d_type)
    mask_container_2d = ((x_hr**2 + y_hr**2) < r_container**2).astype(d_type)
    mask_container    = np.repeat(mask_container_2d, N_hr[2], axis=2)
    x_2d              = x_hr[:, :, :1].astype(d_type)

    del x_hr, y_hr, z_hr

    #%% reference (CSF) region -- defined first so no cylinders/spheres are placed in it
    c           = N_hr // 2
    hw          = N_hr // 16                       
    mask_csf_hr = np.zeros(tuple(N_hr), dtype=d_type)
    mask_csf_hr[c[0]-hw[0]:c[0]+hw[0], c[1]-hw[1]:c[1]+hw[1], :] = 1.0

    #%% create cylindrical phantom
    mask_2d  = mask_container_2d[:, :, 0] * (1.0 - mask_csf_hr[:, :, 0])
    n_pixels = int(np.sum(mask_2d))

    rng_cyl = np.random.RandomState(seed_cyl)
    micro_cyl_2d               = np.zeros_like(mask_2d, dtype=d_type)
    micro_cyl_2d[mask_2d == 1] = (rng_cyl.rand(n_pixels) < vol_frac_cyl).astype(d_type)
    mask_cyl_hr                = np.repeat(micro_cyl_2d[:, :, np.newaxis], N_hr[2], axis=2)
    mask_water_cyl_hr          = 1.0 - mask_cyl_hr
    chi_cyl_hr                 = mask_cyl_hr * (1.0 / vol_frac_cyl)

    del mask_2d, micro_cyl_2d, mask_container_2d

    #%% create spherical phantom
    mask_extra_cyl_hr = mask_container * (1.0 - mask_cyl_hr) * (1.0 - mask_csf_hr)
    n_extra_pixels    = int(np.sum(mask_extra_cyl_hr))

    rng_sph = np.random.RandomState(seed_sph)
    vol_frac_extra_cyl                  = 1 - vol_frac_cyl
    mask_sph_hr                         = np.zeros_like(mask_container, dtype=d_type)
    mask_sph_hr[mask_extra_cyl_hr == 1] = (rng_sph.rand(n_extra_pixels) < (vol_frac_sph/vol_frac_extra_cyl)).astype(d_type)
    mask_water_sph_hr                   = (1.0 - mask_sph_hr) * mask_water_cyl_hr
    chi_sph_hr                          = mask_sph_hr * (1.0 / vol_frac_sph)

    del mask_extra_cyl_hr, n_extra_pixels

    #%% create composite phantom
    contribution_sph   = 1.0 - contribution_cyl
    chi_comp_hr        = contribution_cyl * chi_cyl_hr + contribution_sph * chi_sph_hr
    # These phantoms use the same water compartment; neither mask is modified.
    mask_water_comp_hr = mask_water_sph_hr

    #%% add gradient along x to phantoms (range -0.5 ... +0.5 across the container)
    grad         = np.clip((x_2d + r_container) / (2 * r_container), 0.0, 1.0) - 0.5
    chi_cyl_hr  *= grad
    chi_sph_hr  *= grad
    chi_comp_hr *= grad
    del x_2d

    #%% downsample phantoms
    chi_comp_lr       = sf.downsample_3d_fft(chi_comp_hr, window_size=window_size, factor=dwn_samp_f, agg_func=np.mean)
    chi_sph_lr        = sf.downsample_3d_fft(chi_sph_hr, window_size=window_size, factor=dwn_samp_f, agg_func=np.mean)
    chi_cyl_lr        = sf.downsample_3d_fft(chi_cyl_hr, window_size=window_size, factor=dwn_samp_f, agg_func=np.mean)
    mask_cont_lr      = sf.downsample_3d_fft(mask_container, window_size=window_size, factor=dwn_samp_f, agg_func=np.mean)
    mask_csf_lr       = sf.downsample_3d_fft(mask_csf_hr, window_size=window_size, factor=dwn_samp_f, agg_func=np.mean)
    vol_frac_C        = sf.downsample_3d_fft(mask_cyl_hr, window_size=window_size, factor=dwn_samp_f, agg_func=np.mean)
    vol_frac_W        = sf.downsample_3d_fft(mask_water_comp_hr, window_size=window_size, factor=dwn_samp_f, agg_func=np.mean)
    water_volume_cyl  = sf.downsample_3d_fft(mask_water_cyl_hr, window_size=window_size, factor=dwn_samp_f, agg_func=np.sum)
    water_volume_sph  = sf.downsample_3d_fft(mask_water_sph_hr, window_size=window_size, factor=dwn_samp_f, agg_func=np.sum)
    water_volume_comp = water_volume_sph


    #%% calculate parameter: lambda of ground truth for the forward model
    mask_roi   = mask_cont_lr > 0.5
    vol_frac_S = 1.0 - vol_frac_C - vol_frac_W
    ratio_C    = np.divide(vol_frac_C, 1.0 - vol_frac_C,
                           out=np.zeros_like(vol_frac_C), where=(1.0 - vol_frac_C) > 0)
    lambda_gt  = np.divide(vol_frac_C - vol_frac_S * ratio_C, vol_frac_W,
                           out=np.zeros_like(vol_frac_W), where=mask_roi & (vol_frac_W > 0))

    #%% simulate local fields with multiple orientations measurements without noise
    LoFds_dict = {}
    dir_samp   = sf.generate_fibonacci_hemisphere(n_dir_samp)
    LoFds_cyl  = sf.simulate_local_fields(N_hr, voxel_size_hr, chi_cyl_hr, water_volume_cyl, mask_water_cyl_hr, dir_samp, dwn_samp_f, window_size)
    LoFds_sph  = sf.simulate_local_fields(N_hr, voxel_size_hr, chi_sph_hr, water_volume_sph, mask_water_sph_hr, dir_samp, dwn_samp_f, window_size)
    LoFds_comp = sf.simulate_local_fields(N_hr, voxel_size_hr, chi_comp_hr, water_volume_comp, mask_water_comp_hr, dir_samp, dwn_samp_f, window_size)

    LoFds_dict['cylindrical']   = LoFds_cyl
    LoFds_dict['spherical']     = LoFds_sph
    LoFds_dict['composite']     = LoFds_comp
    LoFds_dict['dir_samp']      = dir_samp
    LoFds_dict['voxel_size_lr'] = voxel_size_lr

    #%% simuate single orientation local field with noise --- cylindrical phantom
    mask_phase = mask_roi.copy()
    mask_phase = mask_phase * (1.0 - mask_csf_lr)
    LoFd_cyl_nf = sf.simulate_local_fields(N_hr, voxel_size_hr, chi_cyl_hr, water_volume_cyl, mask_water_cyl_hr, [B0_dir], dwn_samp_f, window_size)
    LoFd_dict = {}

    for i, snr in enumerate(SNRs):
        snr = np.inf if snr >= 1e10 else snr
        result = sf.simulate_complex_noise(
            LoFd_cyl_nf[0], 
            mask_phase,
            B0=b0_t,
            TE=te,
            SNR=snr,
            LoFd_unit='ppm',
            magnitude_mode="uniform",
            background=background,
            seed=i*10)
        LoFd_cyl = result['LoFd_noisy_wrapped']
        LoFd_dict[str(snr)] = LoFd_cyl

    #%% save phantoms and parameters
    np.savez(DATA_DIR / "Cylindrical_Phantom.npz",
             chi_cyl_lr=chi_cyl_lr, water_volume=water_volume_cyl,
             mask_roi=mask_roi, mask_csf=mask_csf_lr)

    np.savez(DATA_DIR / "Spherical_Phantom.npz",
             chi_sph_lr=chi_sph_lr, water_volume=water_volume_sph,
             mask_roi=mask_roi, mask_csf=mask_csf_lr, lambda_gt=lambda_gt)

    np.savez(DATA_DIR / "Composite_Phantom.npz",
             chi_comp_lr=chi_comp_lr, chi_cyl_lr=chi_cyl_lr*contribution_cyl, chi_sph_lr=chi_sph_lr*contribution_sph,
             water_volume=water_volume_comp, mask_roi=mask_roi, mask_csf=mask_csf_lr, lambda_gt=lambda_gt)

    np.savez(DATA_DIR / "LoFds.npz", **LoFds_dict)
    np.savez(DATA_DIR / "LoFd_cyl_noisy.npz", **LoFd_dict)
    # %%
    filenames = ("Cylindrical_Phantom.npz", "Spherical_Phantom.npz",
                 "Composite_Phantom.npz", "LoFds.npz", "LoFd_cyl_noisy.npz")
    return {name: DATA_DIR / name for name in filenames}
