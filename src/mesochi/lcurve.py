#%% L-curve helpers independent of the reconstruction method
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

RC_LCURVE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
}


def moving_average_smoothing(data, window_size=5):
    """Moving average smoothing with smaller windows at the endpoints."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 1 or not np.all(np.isfinite(data)):
        raise ValueError("data must be a finite 1D array.")
    if not isinstance(window_size, (int, np.integer)) or window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")
    smoothed = np.zeros_like(data)
    half = window_size // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        smoothed[i] = np.mean(data[start:end])
    return smoothed


def _prepare_curve(residual_norms, solution_norms, lambda_values):
    residual_norms = np.asarray(residual_norms, dtype=float)
    solution_norms = np.asarray(solution_norms, dtype=float)
    lambda_values = np.asarray(lambda_values, dtype=float)
    arrays = (residual_norms, solution_norms, lambda_values)
    if any(a.ndim != 1 for a in arrays) or any(a.shape != lambda_values.shape for a in arrays):
        raise ValueError("Norms and lambda_values must be matching 1D arrays.")
    if len(lambda_values) < 5:
        raise ValueError("Use at least five candidate parameters for L-curve selection.")
    if any(not np.all(np.isfinite(a)) or np.any(a <= 0) for a in arrays):
        raise ValueError("Norms and candidate parameters must be positive and finite.")
    if len(np.unique(lambda_values)) != len(lambda_values):
        raise ValueError("Candidate parameters must be distinct.")
    order = np.argsort(lambda_values)
    x, y = np.log10(residual_norms[order]), np.log10(solution_norms[order])
    if np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
        raise ValueError("Degenerate L-curve: both norms must vary across the sweep.")
    # A straight line in log-log space has no identifiable corner.
    points = np.column_stack(((x - x.mean()) / np.ptp(x), (y - y.mean()) / np.ptp(y)))
    singular = np.linalg.svd(points, compute_uv=False)
    if singular[1] <= 1e-8 * singular[0]:
        raise ValueError("Degenerate L-curve: the log-log curve has no identifiable bend.")
    return residual_norms, solution_norms, lambda_values, order


#%% corner selection

def find_lcurve_corner(residual_norms, solution_norms, lambda_values,
                      method='curvature', smoothing=True, plot=True):
    """Return (selected_lambda, original_index, info) for an L-curve.

    solution_norms is the penalty measure: ||x||_2 for L2, or ||Gx||_1
    for the anisotropic TV objective used in reconstruction.py.
    The selected parameter is a candidate from the supplied sweep, not a
    guarantee of minimum reconstruction error. Inspect the curve as well.
    """
    res, sol, lam, order = _prepare_curve(residual_norms, solution_norms, lambda_values)
    log_res, log_sol = np.log10(res[order]), np.log10(sol[order])
    t = np.log10(lam[order])
    if method not in ('curvature', 'distance', 'angle'):
        raise ValueError("method must be 'curvature', 'distance', or 'angle'.")
    if np.any(np.diff(log_res) < -1e-8) or np.any(np.diff(log_sol) > 1e-8):
        print("Warning: norms are not monotonic with increasing lambda; inspect solver convergence.")
    info = {'method': method}
    first, last = 1, len(t) - 1  # exclude endpoint candidates

    if method == 'curvature':
        if smoothing and len(t) > 10:
            log_res_smooth = moving_average_smoothing(log_res, 5)
            log_sol_smooth = moving_average_smoothing(log_sol, 5)
        else:
            log_res_smooth, log_sol_smooth = log_res, log_sol
        spl_res = interpolate.CubicSpline(t, log_res_smooth)
        spl_sol = interpolate.CubicSpline(t, log_sol_smooth)
        t_fine = np.linspace(t.min(), t.max(), 500)
        dx, dy = spl_res.derivative()(t_fine), spl_sol.derivative()(t_fine)
        ddx, ddy = spl_res.derivative(2)(t_fine), spl_sol.derivative(2)(t_fine)
        numerator = np.abs(dx * ddy - ddx * dy)
        denominator = (dx**2 + dy**2)**1.5
        curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-14)
        start, stop = 50, 450  # exclude the outer 10 percent of the dense grid
        valid = curvature[start:stop]
        if not np.all(np.isfinite(valid)) or valid.max() <= 1e-10:
            raise ValueError("Degenerate L-curve: curvature does not identify a corner.")
        max_idx_fine = start + int(np.argmax(valid))
        sorted_idx = first + int(np.argmin(np.abs(t[first:last] - t_fine[max_idx_fine])))
        info.update(
            curvature=curvature, t_fine=t_fine,
            log_res_fine=spl_res(t_fine), log_sol_fine=spl_sol(t_fine),
            max_curvature=float(curvature[max_idx_fine]), corner_idx_fine=max_idx_fine,
        )
    elif method == 'distance':
        res_norm = (log_res - log_res.min()) / np.ptp(log_res)
        sol_norm = (log_sol - log_sol.min()) / np.ptp(log_sol)
        # Distance from the endpoint chord avoids selecting an extreme just
        # because the log norms span several orders of magnitude.
        dx = res_norm[-1] - res_norm[0]
        dy = sol_norm[-1] - sol_norm[0]
        chord_length = np.hypot(dx, dy)
        if chord_length < 1e-14:
            raise ValueError("Degenerate L-curve: endpoint chord has zero length.")
        distances = np.abs(dx * (sol_norm - sol_norm[0]) - dy * (res_norm - res_norm[0])) / chord_length
        sorted_idx = first + int(np.argmax(distances[first:last]))
        original_distances = np.empty_like(distances)
        original_distances[order] = distances
        info['distances'] = original_distances
    else:
        # Turn angle between consecutive segments (straight curve -> angle zero).
        points = np.column_stack((log_res, log_sol))
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths < 1e-14):
            raise ValueError("Degenerate L-curve: repeated adjacent norm pairs.")
        cos_angle = np.sum(segments[:-1] * segments[1:], axis=1) / (lengths[:-1] * lengths[1:])
        angles = np.arccos(np.clip(cos_angle, -1, 1))
        sorted_idx = 1 + int(np.argmax(angles))
        original_angles = np.zeros_like(lam)
        original_angles[order[1:-1]] = angles
        info['angles'] = original_angles

    corner_idx = int(order[sorted_idx])
    optimal_lambda = float(lam[corner_idx])
    if plot:
        plot_lcurve_analysis(res, sol, lam, corner_idx, optimal_lambda, info)
    return optimal_lambda, corner_idx, info


def find_l_curve_corner(lambda_values, residual_norms, solution_norms):
    """Compatibility wrapper returning (lambda, original_index, sampled curvature)."""
    lam, idx, info = find_lcurve_corner(
        residual_norms, solution_norms, lambda_values,
        method='curvature', smoothing=False, plot=False,
    )
    curvature = np.interp(np.log10(lambda_values), info['t_fine'], info['curvature'])
    return lam, idx, curvature


#%% plotting (both existing naming conventions)


def plot_lcurve_analysis(residual_norms, solution_norms, lambda_values, 
                        corner_idx, optimal_lambda, info, save_path=None, show=True,
                        penalty_label=None):
    """Plot L-curve analysis results"""
    plt.rcParams.update(RC_LCURVE)
    fig = plt.figure(figsize=(15, 10))
    
    order = np.argsort(lambda_values)
    residual_norms = np.asarray(residual_norms)
    solution_norms = np.asarray(solution_norms)
    lambda_values = np.asarray(lambda_values)

    # 1. L-curve plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.loglog(residual_norms[order], solution_norms[order], 'bo-', linewidth=2, markersize=6, label='L-curve')
    ax1.loglog(residual_norms[corner_idx], solution_norms[corner_idx], 
               'ro', markersize=12, label=f'Corner (λ={optimal_lambda:.2e})')
    ax1.set_xlabel('Residual Norm ||Ax - b||')
    ax1.set_ylabel(penalty_label or 'Solution Norm ||x||')
    ax1.set_title('L-curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark some λ values
    for i in [0, len(lambda_values)//2, len(lambda_values)-1]:
        ax1.annotate(f'λ={lambda_values[i]:.1e}', 
                    (residual_norms[i], solution_norms[i]),
                    fontsize=8, ha='center')
    
    # 2. Norms vs λ
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogx(lambda_values[order], residual_norms[order], 'b-', linewidth=2, label='Residual Norm')
    ax2.semilogx(lambda_values[order], solution_norms[order], 'r-', linewidth=2, label=penalty_label or 'Solution Norm')
    ax2.semilogx(optimal_lambda, residual_norms[corner_idx], 'bo', markersize=8)
    ax2.semilogx(optimal_lambda, solution_norms[corner_idx], 'ro', markersize=8)
    ax2.set_xlabel('Regularization Parameter λ')
    ax2.set_ylabel('Norm Value')
    ax2.set_title('Norms vs λ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Curvature plot (if using curvature method)
    if info['method'] == 'curvature':
        ax3 = plt.subplot(2, 3, 3)
        ax3.semilogx(10**info['t_fine'], info['curvature'],
                    'g-', linewidth=2, label='Curvature')
        ax3.semilogx(10**info['t_fine'][info['corner_idx_fine']], info['max_curvature'],
                    'ro', markersize=10, label='Interior maximum')
        ax3.set_xlabel('Regularization Parameter λ')
        ax3.set_ylabel('Curvature κ')
        ax3.set_title('L-curve Curvature')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. L-curve in log coordinates
    ax4 = plt.subplot(2, 3, 4)
    log_res = np.log10(residual_norms)
    log_sol = np.log10(solution_norms)
    
    ax4.plot(log_res[order], log_sol[order], 'bo-', linewidth=2, markersize=6)
    ax4.plot(log_res[corner_idx], log_sol[corner_idx], 'ro', markersize=12)
    ax4.set_xlabel('log(Residual Norm)')
    ax4.set_ylabel(f'log({penalty_label})' if penalty_label else 'log(Solution Norm)')
    ax4.set_title('L-curve (Log Coordinates)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Spline fitting results (if using curvature method)
    if info['method'] == 'curvature':
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(info['log_res_fine'], info['log_sol_fine'], 'g-', linewidth=2, label='Spline Fit')
        ax5.plot(log_res, log_sol, 'bo', markersize=4, alpha=0.5, label='Data Points')
        ax5.plot(log_res[corner_idx], log_sol[corner_idx], 'ro', markersize=12, label='Corner')
        ax5.set_xlabel('log(Residual Norm)')
        ax5.set_ylabel(f'log({penalty_label})' if penalty_label else 'log(Solution Norm)')
        ax5.set_title('Cubic Spline Fitting')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. λ value distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.semilogy(range(len(lambda_values)), lambda_values, 'mo-', linewidth=2)
    ax6.semilogy(corner_idx, optimal_lambda, 'ro', markersize=12, label=f'Optimal λ={optimal_lambda:.2e}')
    ax6.set_xlabel('Point Index')
    ax6.set_ylabel('λ Value')
    ax6.set_title('λ Value Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
        return None
    return fig


def plot_l_curve_analysis(residual_norms, solution_norms, lambda_values, best_idx, curvature, save_path=None, show=True):
    """
    Plot the L-curve and curvature function to visualize the corner selection.
    Parameters:
    residual_norms: ndarray
    solution_norms: ndarray
    lambda_values: list
        list of regularization parameters.
    best_idx: int 
    curvature: ndarray
    """

    plt.rcParams.update(RC_LCURVE)
    fig = plt.figure(figsize=(12, 5))
    residual_norms = np.asarray(residual_norms)
    solution_norms = np.asarray(solution_norms)
    lambda_values = np.asarray(lambda_values)
    curvature = np.asarray(curvature)
    order = np.argsort(lambda_values)
    
    # 左图: L-curve (log-log plot)
    plt.subplot(1, 2, 1)
    plt.loglog(residual_norms[order], solution_norms[order], 'b.-', markersize=5, label='L-curve')
    plt.loglog(residual_norms[best_idx], solution_norms[best_idx], 'r*', markersize=15, 
               label=rf'Optimal $\lambda$={lambda_values[best_idx]:.2e}')
    plt.xlabel('Residual Norm $||Ax - b||_2$')
    plt.ylabel('Solution Norm $||x||_2$')
    plt.title('L-Curve')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # 右图: 曲率函数 (semilogx plot)
    plt.subplot(1, 2, 2)
    plt.semilogx(lambda_values[order], curvature[order], 'g.-')
    plt.axvline(lambda_values[best_idx], color='r', linestyle='--', label='Max Curvature')
    plt.xlabel(r'Regularization Parameter $\lambda$')
    plt.ylabel('Curvature')
    plt.title('Curvature Function')
    plt.legend()
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
        return None
    return fig
