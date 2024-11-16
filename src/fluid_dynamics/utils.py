import numba
import src.parameters as cfg

@numba.jit(nopython=True)
def get_indicator_function(u: float) -> float:
    if u > cfg.T_0:
        return 1.0
    return 1.0 / (cfg.EPS * cfg.EPS)

@numba.jit(nopython=True)
def get_thermal_expansion_coef(u: float) -> float:
    if u < 0:
        return 0.0
    return -0.0000000985 * u * u + 0.00001482 * u - 0.00005277

@numba.jit(nopython=True)
def get_kinematic_visc(u: float) -> float:
    if u < 0:
        return 0.0
    return 906.36 * u * u - 57191.82 * u + 1785718.18
