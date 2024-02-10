import math
import numba

from src.parameters import T_0, C_ICE_VOL, C_WATER_VOL, L_VOL, K_ICE, K_WATER


@numba.jit(nopython=True)
def delta_function(T_j_i: float, _delta: float):
    """
    Сглаженная аппроксимация дельта-функции, сосредоточенная в T_0.
    :param T_j_i: Значение температуры в (i, j) узле координатной сетки.
    :param _delta: Параметр сглаживания.
    :return: Значение сглаженной дельта-функции в точке T_j_i.
    """
    return math.exp(-(T_j_i - T_0)*(T_j_i - T_0)/(2.0*_delta*_delta)) / ((2.0*math.pi)**0.5 * _delta)


@numba.jit(nopython=True)
def c_smoothed(T_j_i: float, _delta: float):
    """
    Сглаженная эффективная объемная теплоемкость.
    :param T_j_i: Значение температуры в (i, j) узле координатной сетки.
    :param _delta: Параметр сглаживания.
    :return: Значение сглаженной эффективной объемной теплоемкости при температуре T_j_i.
    """
    if _delta <= 0:
        return C_ICE_VOL if T_j_i < T_0 else C_WATER_VOL

    return C_ICE_VOL + (C_WATER_VOL - C_ICE_VOL) * (1.0 + math.erf((T_j_i - T_0)/(2**0.5*_delta))) * 0.5 + \
           L_VOL * delta_function(T_j_i, _delta)


@numba.jit(nopython=True)
def k_smoothed(T_j_i: float, _delta: float):
    """
    Сглаженный коэффициент теплопроводности.
    :param T_j_i: Значение температуры в (i, j) узле координатной сетки.
    :param _delta: Параметр сглаживания.
    :return: Значение сглаженного коэффициента теплопроводности при температуре T_j_i.
    """
    if _delta <= 0.0:
        return K_ICE if T_j_i < T_0 else K_WATER

    return K_ICE + (K_WATER - K_ICE) * (1.0 + math.erf((T_j_i - T_0)/(2**0.5*_delta))) * 0.5


@numba.jit(nopython=True)
def delta_parabolic(T_j_i: float, _delta: float):
    if abs(T_j_i - T_0) <= _delta:
        return 0.75 * (1.0 - T_j_i * T_j_i / (_delta * _delta)) / _delta
    return 0.0


@numba.jit(nopython=True)
def delta_const(T_j_i: float, _delta: float):
    if abs(T_j_i - T_0) <= _delta:
        return 0.5 / _delta
    return 0.0


@numba.jit(nopython=True)
def c_simple(T_j_i: float, _delta: float):
    if abs(T_j_i - T_0) <= _delta:
        return 0.5 * (C_WATER_VOL + C_ICE_VOL) + L_VOL * delta_const(T_j_i, _delta)
    elif T_j_i < T_0 - _delta:
        return C_ICE_VOL
    else:
        return C_WATER_VOL


@numba.jit(nopython=True)
def k_simple(T_j_i: float, _delta: float):
    if abs(T_j_i - T_0) <= _delta:
        return K_WATER + 0.5 * (K_WATER - K_ICE) * (T_j_i - T_0 - _delta) / _delta
    elif T_j_i < T_0 - _delta:
        return K_ICE
    else:
        return K_WATER
