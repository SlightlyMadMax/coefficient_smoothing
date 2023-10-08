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
    return K_ICE + (K_WATER - K_ICE) * (1.0 + math.erf((T_j_i - T_0)/(2**0.5*_delta))) * 0.5

