import numba
import numpy as np
from numpy.typing import NDArray
import src.parameters as cfg


@numba.jit(nopython=True)
def get_indicator_function(u: float) -> float:
    if u > cfg.T_0:
        return 1.0
    return 1.0 / (cfg.EPS * cfg.EPS)


@numba.jit(nopython=True)
def get_thermal_expansion_coef(u: float) -> float:
    if u < cfg.T_0:
        return 0.0
    return -9.85e-8 * u * u + 1.4872e-5 * u - 5.2770e-5


@numba.jit(nopython=True)
def get_kinematic_visc(u: float) -> float:
    if u < cfg.T_0:
        return 0.0
    return 5.56e-10 * u * u - 4.95e-8 * u + 1.767e-6


@numba.jit(nopython=True)
def calculate_velocity_field(sf: NDArray[np.float64], dx: float, dy: float):
    inv_dy = 1.0 / dy
    inv_dx = 1.0 / dx

    v_x = np.zeros_like(sf)
    v_y = np.zeros_like(sf)

    # Interior points: central difference
    v_x[1:-1, :] = 0.5 * inv_dy * (sf[2:, :] - sf[:-2, :])
    v_y[:, 1:-1] = -0.5 * inv_dx * (sf[:, 2:] - sf[:, :-2])

    # Boundary points: forward/backward difference
    v_x[0, :] = (sf[1, :] - sf[0, :]) * inv_dy
    v_x[-1, :] = (sf[-1, :] - sf[-2, :]) * inv_dy
    v_y[:, 0] = -(sf[:, 1] - sf[:, 0]) * inv_dx
    v_y[:, -1] = -(sf[:, -1] - sf[:, -2]) * inv_dx

    return v_x, v_y
