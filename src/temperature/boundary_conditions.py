import numba
import numpy as np
from numpy.typing import NDArray
from math import sin, cos, pi

import src.parameters as cfg


@numba.jit(nopython=True)
def get_left_bc_1(time: float, n_y: int) -> NDArray[np.float64]:
    return cfg.T_WATER_MAX * np.ones(n_y)


@numba.jit(nopython=True)
def get_top_bc_1(time: float, n_x: int) -> NDArray[np.float64]:
    return cfg.T_WATER_MAX * np.ones(n_x)


@numba.jit(nopython=True)
def get_right_bc_1(time: float, n_y: int) -> NDArray[np.float64]:
    return 3.0 * np.ones(n_y)


@numba.jit(nopython=True)
def get_bottom_bc_1(time: float, n_x: int) -> NDArray[np.float64]:
    return cfg.T_WATER_MAX * np.ones(n_x)


@numba.jit(nopython=True)
def get_top_bc_2(time: float) -> float:
    return 0.0


@numba.jit(nopython=True)
def solar_heat(t: float):
    """
    Функция для вычисления потока солнечной радиации.

    :param t: Время в секундах.
    :return: Величина солнечного потока радиации на горизонтальную поверхность при заданных параметрах
    в момент времени t. [Вт/м^2]
    """
    # вычисляем склонение солнца
    decl = cfg.DECL * sin(2 * pi * t / (365 * 24.0 * 3600.0) - pi / 2)

    return cfg.Q_SOL * (
        sin(cfg.LAT) * sin(decl)
        + cos(cfg.LAT) * cos(decl) * cos(cfg.RAD_SPEED * t + 12.0 * 3600.0)
    )


@numba.jit(nopython=True)
def air_temperature(t: float):
    """
    Функция изменения температуры воздуха.

    :param t: Время в секундах
    :return: Температура воздуха в заданный момент времени
    """
    return (
        cfg.T_air
        + cfg.T_amp_day * sin(2 * pi * t / (24.0 * 3600.0) + pi / 2)
        + cfg.T_amp_year * sin(2 * pi * t / (365 * 24.0 * 3600.0) + pi / 2)
    )


@numba.jit(nopython=True)
def conv_coef(wind_speed: float):
    return wind_speed**0.5 * (7.0 + 7.2 / wind_speed**2)


@numba.jit(nopython=True)
def get_top_bc_3(time: float) -> (float, float):
    # Определяем температуру воздуха у поверхности
    T_air_t = air_temperature(time)

    # Определяем тепловой поток солнечной энергии
    Q_sol = solar_heat(time)

    # Коэффициент теплообмена с воздухом
    c = conv_coef(cfg.WIND_SPEED)

    # Приведенный коэффициент теплообмена
    c_r = c * (1.0 + 0.0195 * cfg.A) + 0.205 * (T_air_t / 100.0) ** 3

    # Давление насыщенного водяного пара
    p = cfg.A * T_air_t + cfg.B

    # Солнечная радиация с учетом облачности
    h_c = Q_sol * (1.0 - 0.38 * cfg.CLOUDINESS * (1.0 + cfg.CLOUDINESS))

    # Приведенная температура окружающей среды
    T_r = (
        c * (T_air_t - 0.0195 * (cfg.B - p * cfg.REL_HUMIDITY))
        + 19.9 * (T_air_t / 100.0) ** 4
        + h_c
    ) / c_r

    psi = T_r * c_r / cfg.K_ICE
    phi = -c_r / cfg.K_ICE

    # psi = (cfg.CONV_COEF * T_air_t + Q_sol) / cfg.K_ICE
    # phi = - cfg.CONV_COEF / cfg.K_ICE
    return psi, phi
