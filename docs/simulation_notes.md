# Phantom simulations

The phantom experiments compare conventional quantitative susceptibility
mapping (QSM) with microstructure-informed QSM (μQSM). They use known
susceptibility distributions to examine how a model represents different
microstructures, and how noise and regularization affect a single-orientation
reconstruction.

## Microstructure and susceptibility

The default simulation uses a 1024 × 1024 × 1024 grid with 0.03125 mm voxels,
covering a 32 mm field of view along each axis. A cylindrical container of
radius 8 mm extends along the z-axis. A centered square CSF reference column
runs through the container; inclusions are not placed in that column.

The three phantoms use the following susceptibility sources:

- **Cylindrical:** Each eligible position in the two-dimensional cross-section
  is selected independently with probability 0.5. The selected positions are
  repeated along z, creating parallel cylinders. The random seed is 42.
- **Spherical-inclusion:** Positions are independently selected in the
  remaining container voxels outside the cylinders and CSF, with conditional
  probability 0.05 / (1 − 0.5). The target spherical volume fraction is 0.05;
  the random seed is 100. These are voxel-scale inclusions; the code does not
  construct explicitly resolved geometric spheres.
- **Composite:** The cylindrical and spherical susceptibility components are
  combined with weights 0.9 and 0.1, respectively.

Each component's susceptibility is scaled by the reciprocal of its target
volume fraction. A clipped linear gradient along x, ranging from −0.5 to +0.5
across the container, then modulates the susceptibility. This provides a
spatially varying ground truth for the comparisons.

Water occupies locations outside the relevant susceptibility inclusions. The
spherical and composite models use the same water mask, which excludes both
spherical inclusions and cylinders. Therefore, the spherical experiment
represents the spherical contribution in a shared microstructure, rather than
an isolated sphere-only water model.

## Voxel-scale maps and local fields

The high-resolution susceptibility and fraction maps are averaged over blocks
of 32 voxels per axis, giving a 32 × 32 × 32 grid with 1 mm voxels. Water voxel
counts are summed within each block. The container and CSF masks and a
ground-truth microstructural parameter are also derived at this resolution.

For each field direction, the simulation convolves the high-resolution
susceptibility with the dipole kernel using an FFT. It retains the field in
water, sums it within each low-resolution block, divides by the corresponding
water count, and subtracts the whole-volume mean. Six directions sampled over
a hemisphere generate noise-free multi-orientation fields for all three
phantoms.

A separate cylindrical experiment uses a single field direction along x.
It converts the local field to phase at 3 T and an echo time of 5 ms, forms a
complex signal, and adds independent Gaussian noise to its real and imaginary
channels. The signal magnitude is 1 inside the phase mask and 0.5 outside it.
The simulated signal-to-noise ratios are infinity, 100, 50, 30, and 10. Wrapped
phase is extracted and converted back to a local-field estimate.

## Reconstruction comparisons

The multi-orientation experiment reconstructs each phantom with standard QSM
and its corresponding μQSM model, then compares the reconstructed maps with
the known ground truth. For the composite phantom it also displays the
cylindrical and spherical components. The unconstrained composite forward
model does not uniquely determine those components from the field alone, so
their separation requires careful interpretation.

The single-orientation cylindrical experiment reconstructs each noise level
with μQSM using L2 and total-variation regularization. The default parameters
are 0.01 for L2 and 0.001 for TV. Comparing the maps across noise levels
illustrates the effect of noise and regularization on the reconstruction.
