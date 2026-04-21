# solver/multigrid.py
"""
Geometric Multigrid V-cycle solver for the pressure Poisson equation:∇²p = rhs
with Neumann BCs on all boundaries.
"""

import numpy as np


def _restrict(r: np.ndarray) -> np.ndarray:
    """Full-weighting restriction: fine → coarse."""
    return 0.25 * (r[::2, ::2] + r[1::2, ::2] +
                   r[::2, 1::2] + r[1::2, 1::2])


def _prolongate(e_coarse: np.ndarray, shape_fine: tuple) -> np.ndarray:
    """Bilinear prolongation: coarse → fine."""
    Nx, Ny = shape_fine
    e = np.zeros((Nx, Ny))
    e[::2, ::2]   = e_coarse
    e[1::2, ::2]  = e_coarse
    e[::2, 1::2]  = e_coarse
    e[1::2, 1::2] = e_coarse
    return e


def _gauss_seidel(p: np.ndarray, rhs: np.ndarray,dx: float, dy: float, n_iter: int) -> np.ndarray:
    """Red-black Gauss-Seidel smoother."""
    dx2, dy2 = dx**2, dy**2
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)

    for _ in range(n_iter):
        for color in (0, 1):          # red / black
            i_s = np.arange(1, p.shape[0]-1)
            j_s = np.arange(1, p.shape[1]-1)
            I, J = np.meshgrid(i_s, j_s, indexing='ij')
            mask = ((I + J) % 2 == color)
            ii, jj = I[mask], J[mask]
            p[ii, jj] = (
                (p[ii+1, jj] + p[ii-1, jj]) / dx2 +
                (p[ii, jj+1] + p[ii, jj-1]) / dy2 -
                rhs[ii, jj]
            ) / denom

        # Neumann BC
        p[0, :]  = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0]  = p[:, 1]
        p[:, -1] = p[:, -2]

    return p


def _residual(p: np.ndarray, rhs: np.ndarray,
               dx: float, dy: float) -> np.ndarray:
    dx2, dy2 = dx**2, dy**2
    res = np.zeros_like(p)
    res[1:-1, 1:-1] = (
        (p[2:, 1:-1] - 2*p[1:-1, 1:-1] + p[:-2, 1:-1]) / dx2 +
        (p[1:-1, 2:] - 2*p[1:-1, 1:-1] + p[1:-1, :-2]) / dy2 -
        rhs[1:-1, 1:-1]
    )
    return res


def vcycle(p: np.ndarray, rhs: np.ndarray,
           dx: float, dy: float,
           levels: int, pre_smooth: int, post_smooth: int) -> np.ndarray:
    """
    Recursive V-cycle.
    Base case: solve directly on coarsest grid with many GS sweeps.
    """
    if levels == 1 or min(p.shape) <= 4:
        return _gauss_seidel(p, rhs, dx, dy, 50)

    # Pre-smooth
    p = _gauss_seidel(p, rhs, dx, dy, pre_smooth)

    # Compute residual and restrict
    res = _residual(p, rhs, dx, dy)
    res_c = _restrict(res)
    e_c = np.zeros_like(res_c)

    # Recurse on coarse grid
    e_c = vcycle(e_c, res_c, dx*2, dy*2, levels-1, pre_smooth, post_smooth)

    # Prolongate and correct
    p += _prolongate(e_c, p.shape)

    # Post-smooth
    p = _gauss_seidel(p, rhs, dx, dy, post_smooth)

    # Fix mean pressure (remove null-space for pure Neumann)
    p -= p.mean()
    return p


def solve_poisson(p: np.ndarray, rhs: np.ndarray,
                  dx: float, dy: float,
                  levels: int = 4, n_cycles: int = 2,
                  pre_smooth: int = 2, post_smooth: int = 2,
                  tol: float = 1e-6, max_iter: int = 50) -> tuple:
    """
    Outer loop: apply n_cycles V-cycles until convergence.
    Returns (p, residual_history).
    """
    res_hist = []
    for _ in range(max_iter):
        for _ in range(n_cycles):
            p = vcycle(p, rhs, dx, dy, levels, pre_smooth, post_smooth)
        r = np.linalg.norm(_residual(p, rhs, dx, dy))
        res_hist.append(r)
        if r < tol:
            break
    return p, res_hist
