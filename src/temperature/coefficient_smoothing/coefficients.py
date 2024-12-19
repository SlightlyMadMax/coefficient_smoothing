import math
import numba


@numba.jit(nopython=True)
def delta_function(u_j_i: float, u_0: float, _delta: float) -> float:
    """
    Smoothed approximation of the delta function, centered at u_0.

    :param u_j_i: The temperature value at the (j, i) node of the coordinate grid.
    :param u_0: The point where the delta function is centered.
    :param _delta: The smoothing parameter.
    :return: The value of the smoothed delta function at the point u_j_i.
    """
    return math.exp(-(u_j_i - u_0) * (u_j_i - u_0) / (2.0 * _delta * _delta)) / (
        (2.0 * math.pi) ** 0.5 * _delta
    )


@numba.jit(nopython=True)
def c_smoothed(
    u_j_i: float,
    u_pt: float,
    u_ref: float,
    c_solid: float,
    c_liquid: float,
    l_solid: float,
    _delta: float,
) -> float:
    """
    Smoothed effective volumetric heat capacity.

    :param u_j_i: The temperature value at the (j, i) node of the coordinate grid.
    :param u_pt: The phase transition temperature.
    :param u_ref: The reference temperature.
    :param c_solid: The volumetric heat capacity of the solid phase.
    :param c_liquid: The volumetric heat capacity of the liquid phase.
    :param l_solid: The volumetric latent heat of fusion of the solid phase.
    :param _delta: The smoothing parameter.
    :return: The value of the smoothed effective volumetric heat capacity at the temperature u_j_i.
    """
    u_pt = u_pt - u_ref

    if _delta <= 0:
        return c_solid if u_j_i < u_pt else c_liquid

    return (
        c_solid
        + (c_liquid - c_solid)
        * (1.0 + math.erf((u_j_i - u_pt) / (2**0.5 * _delta)))
        * 0.5
        + l_solid * delta_function(u_j_i=u_j_i, u_0=u_pt, _delta=_delta)
    )


@numba.jit(nopython=True)
def k_smoothed(
    u_j_i: float,
    u_pt: float,
    u_ref: float,
    k_solid: float,
    k_liquid: float,
    _delta: float,
) -> float:
    """
    Smoothed heat conductivity coefficient.

    :param u_j_i: The temperature value at the (j, i) node of the coordinate grid.
    :param u_pt: The phase transition temperature.
    :param u_ref: The reference temperature.
    :param k_solid: The heat conductivity of the solid phase.
    :param k_liquid: The heat conductivity of the liquid phase.
    :param _delta: The smoothing parameter.
    :return: The value of the smoothed heat conductivity coefficient at the temperature u_j_i.
    """
    u_pt = u_pt - u_ref

    if _delta <= 0.0:
        return k_solid if u_j_i < u_pt else k_liquid

    return (
        k_solid
        + (k_liquid - k_solid)
        * (1.0 + math.erf((u_j_i - u_pt) / (2**0.5 * _delta)))
        * 0.5
    )


@numba.jit(nopython=True)
def delta_parabolic(u_j_i: float, u_pt: float, _delta: float) -> float:
    if abs(u_j_i - u_pt) <= _delta:
        return 0.75 * (1.0 - u_j_i * u_j_i / (_delta * _delta)) / _delta
    return 0.0


@numba.jit(nopython=True)
def delta_const(u_j_i: float, u_pt: float, _delta: float) -> float:
    if abs(u_j_i - u_pt) <= _delta:
        return 0.5 / _delta
    return 0.0


@numba.jit(nopython=True)
def c_simple(
    u_j_i: float,
    u_pt: float,
    u_ref: float,
    c_solid: float,
    c_liquid: float,
    l_solid: float,
    _delta: float,
) -> float:
    u_pt = u_pt - u_ref

    if abs(u_j_i - u_pt) <= _delta:
        return 0.5 * (c_liquid + c_solid) + l_solid * delta_const(u_j_i, _delta)
    elif u_j_i < u_pt - _delta:
        return c_solid
    else:
        return c_liquid


@numba.jit(nopython=True)
def k_simple(
    u_j_i: float,
    u_pt: float,
    u_ref: float,
    k_solid: float,
    k_liquid: float,
    _delta: float,
) -> float:
    u_pt = u_pt - u_ref

    if abs(u_j_i - u_pt) <= _delta:
        return k_liquid + 0.5 * (k_liquid - k_solid) * (u_j_i - u_pt - _delta) / _delta
    elif u_j_i < u_pt - _delta:
        return k_solid
    else:
        return k_liquid
