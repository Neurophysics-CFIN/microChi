"""Small CPU checks; no full-resolution phantom generation is required."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(__file__).resolve().parents[1] / '.cache' / 'numba'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache' / 'matplotlib'))

import numpy as np
import sigpy as sp
from mesochi.simulation import operators as linop
from mesochi.simulation import forward as sf
from mesochi.lcurve import find_lcurve_corner, find_l_curve_corner, moving_average_smoothing


class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent / 'fixtures' / 'operator_reference.npz') as data:
            cls.data = dict(data)

    def operators(self):
        for key, cls in (
            ('qsm', linop.QSMLinop), ('cyl', linop.MuQSMLinop),
            ('sph', linop.MuQSMLinop_S), ('comp', linop.MuQSMLinop_CS),
        ):
            args = {name: self.data[name] for name in
                    ('voxel_size', 'B0_dirs', 'mask_roi', 'mask_ref', 'weight')}
            args['N'] = self.data['mask_roi'].shape
            if key != 'qsm':
                args['N_meso'] = self.data['N_meso']
            yield key, cls(**args)

    def test_consolidation_preserves_original_forward_and_adjoint(self):
        for key, A in self.operators():
            with self.subTest(model=key):
                np.testing.assert_allclose(A(self.data[key + '_x']), self.data[key + '_forward'], rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(A.H(self.data[key + '_y']), self.data[key + '_adjoint'], rtol=2e-6, atol=2e-6)

    def test_adjoint_identity_with_weights_and_reference(self):
        for key, A in self.operators():
            with self.subTest(model=key):
                x, y = self.data[key + '_x'], self.data[key + '_y']
                lhs = np.vdot(A(x), y)
                rhs = np.vdot(x, A.H(y))
                self.assertLess(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1), 1e-5)

    def test_single_orientation_preparation_and_split(self):
        roi = self.data['mask_roi']
        for cls in (linop.MuQSMLinop, linop.MuQSMLinop_S, linop.MuQSMLinop_CS):
            A = cls(roi.shape, [1, 1, 1], [1, 0, 0], roi, roi / 6)
            y = linop.prepare_y_for_sigpy(np.ones(roi.shape), A, remove_mean=True)
            self.assertEqual(y.shape, tuple(A.oshape))
            np.testing.assert_allclose(y, 0)
            with self.assertRaises(ValueError):
                linop.prepare_y_for_sigpy(np.ones((2, 2)), A)
        A = linop.MuQSMLinop_S(roi.shape, [1, 1, 1], [1, 0, 0], roi, roi / 6)
        x = np.ones(A.ishape, dtype=np.float32)
        chi, u = linop.split_solution_S(x, A)
        self.assertEqual(chi.shape, roi.shape)
        lam, valid = linop.lambda_from_solution(chi, 2 * u, roi)
        np.testing.assert_allclose(lam[valid], 2)

    def test_composite_null_direction(self):
        A = dict(self.operators())['comp']
        x = self.data['comp_x'].copy()
        a = self.data['mask_roi'] * np.arange(x[0].size).reshape(x[0].shape) / 100
        np.testing.assert_allclose(A(x + np.stack([a, -a, a])), A(x), rtol=2e-5, atol=2e-6)

    def test_solver_results_are_independent(self):
        A = sp.linop.Identity((8,))
        results = []
        for value in (1, 2):
            result = sp.app.LinearLeastSquares(
                A, np.full(8, value, dtype=np.float32),
                x=np.zeros(8, dtype=np.float32), lamda=0.1,
                max_iter=10, tol=1e-6, show_pbar=False,
            ).run()
            results.append(linop._to_numpy(result).copy())
        np.testing.assert_allclose(results[0], 1 / 1.1, rtol=1e-6)
        np.testing.assert_allclose(results[1], 2 / 1.1, rtol=1e-6)
        self.assertFalse(np.shares_memory(*results))

    def test_cpu_import_without_cupy(self):
        script = '''import builtins, sys
original_import = builtins.__import__
def no_cupy(name, *args, **kwargs):
    if name.split('.')[0] in ('cupy', 'cupyx'):
        raise ImportError('CuPy intentionally unavailable in this test')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_cupy
from mesochi.simulation import operators as LinOP, forward as SimF, plotting as Sim_Plot
import numpy as np
assert LinOP.cp is None
from mesochi.exvivo.pipeline import prepare_exvivo_data, run_exvivo_reconstruction
from mesochi.exvivo.lcurve import run_exvivo_lcurve
roi = np.zeros((4,4,4), dtype=np.float32)
roi[1:3,1:3,1:3] = 1
A = LinOP.MuQSMLinop(roi.shape, [1,1,1], [1,0,0], roi, roi/6)
assert np.isfinite(A(np.ones(roi.shape, dtype=np.float32))).all()
try:
    LinOP.QSMLinop(roi.shape, [1,1,1], [1,0,0], roi, use_gpu=True)
except RuntimeError as error:
    assert 'CuPy' in str(error)
else:
    raise AssertionError('GPU request should fail clearly without CuPy')
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-B', '-c', script],
                cwd=directory, capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


class SimulationTests(unittest.TestCase):
    def test_kernel_normalization_and_invalid_directions(self):
        a = sf.B_dip_k((8, 6, 4), [1, 1.5, 2], [1, 2, 3])
        b = linop.Dipole_Kernel((8, 6, 4), [1, 1.5, 2], [2, 4, 6])
        # The original helpers differ by a few float32 ULPs (2*pi cancels
        # algebraically but changes intermediate rounding in LinOP).
        np.testing.assert_allclose(a, b, atol=5e-7)
        self.assertEqual(a[0, 0, 0], 0)
        for bad in ([0, 0, 0], [1, np.nan, 0], [1, 0]):
            for kernel in (sf.B_dip_k, linop.Dipole_Kernel):
                with self.assertRaises(ValueError):
                    kernel((4, 4, 4), [1, 1, 1], bad)

    def test_block_downsampling(self):
        a = np.arange(8**3, dtype=np.float32).reshape(8, 8, 8)
        means = np.empty((4, 4, 4))
        for i, j, k in np.ndindex(means.shape):
            means[i, j, k] = a[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2].mean()
        np.testing.assert_allclose(sf.downsample_3d_fft(a, 2, 2), means)
        np.testing.assert_allclose(sf.downsample_3d_fft(a, 2, 2, np.sum), means * 8)
        with self.assertRaises(ValueError):
            sf.downsample_3d_fft(a, 0, 2)
        with self.assertRaises(ValueError):
            sf.downsample_3d_fft(a, 2, 3)

    def test_noise_reproducibility_and_magnitude_scaling(self):
        mask = np.ones((32, 32, 32), dtype=bool)
        mask[:16] = False
        field = np.zeros(mask.shape, dtype=np.float32)
        a = sf.simulate_complex_noise(field, mask, SNR=100, seed=42)
        b = sf.simulate_complex_noise(field, mask, SNR=100, seed=42)
        c = sf.simulate_complex_noise(field, mask, SNR=100, seed=43)
        np.testing.assert_array_equal(a['S_noisy'], b['S_noisy'])
        self.assertFalse(np.array_equal(a['S_noisy'], c['S_noisy']))
        ratio = a['phase_error'][~mask].std() / a['phase_error'][mask].std()
        self.assertAlmostEqual(float(ratio), 2, delta=0.1)
        clean = sf.simulate_complex_noise(field, mask, SNR=np.inf)
        np.testing.assert_allclose(clean['LoFd_noisy_wrapped'], field)

    def test_zero_water_volume_is_finite(self):
        shape = (4, 4, 4)
        chi = np.ones(shape, dtype=np.float32)
        water = np.zeros((2, 2, 2), dtype=np.float32)
        fields = sf.simulate_local_fields(shape, [1, 1, 1], chi, water, chi, [[1, 0, 0]], 2, 2)
        self.assertTrue(np.isfinite(fields[0]).all())
        np.testing.assert_allclose(fields[0], 0)


class LCurveTests(unittest.TestCase):
    def curve(self):
        lam = np.logspace(-4, 4, 81)
        # Exact ridge solution of the one-dimensional problem A=1, y=1.
        sol = 1 / (1 + lam)
        res = lam / (1 + lam)
        return lam, res, sol

    def test_known_corner_and_original_indices(self):
        lam, res, sol = self.curve()
        for method in ('curvature', 'distance', 'angle'):
            chosen, idx, _ = find_lcurve_corner(res, sol, lam, method=method, smoothing=False, plot=False)
            self.assertAlmostEqual(np.log10(chosen), 0, delta=0.2)
            self.assertEqual(chosen, lam[idx])
            order = np.random.default_rng(7).permutation(len(lam))
            shuffled, shuffled_idx, _ = find_lcurve_corner(res[order], sol[order], lam[order], method=method, smoothing=False, plot=False)
            self.assertEqual(shuffled, chosen)
            self.assertEqual(lam[order][shuffled_idx], chosen)
        chosen, idx, curvature = find_l_curve_corner(lam, res, sol)
        self.assertEqual(curvature.shape, lam.shape)
        self.assertEqual(chosen, lam[idx])

    def test_invalid_and_degenerate_curves(self):
        lam, res, sol = self.curve()
        for arrays in ((res[:4], sol[:4], lam[:4]), (res, sol, np.ones_like(lam)),
                       (np.zeros_like(res), sol, lam), (res, np.ones_like(sol), lam),
                       (lam, 1 / lam, lam), (res[:-1], sol, lam)):
            with self.assertRaises(ValueError):
                find_lcurve_corner(*arrays, plot=False)
        with self.assertRaises(ValueError):
            moving_average_smoothing(res, 4)


if __name__ == '__main__':
    unittest.main()
