import math

import numpy as np
import numba
from numpy import ndarray
from typing import Optional

from src.parameters import T_ICE_MIN, T_WATER_MAX, T_0, WATER_H
from src.geometry import DomainGeometry


@numba.jit(nopython=True)
def get_max_delta(T: ndarray) -> float:
    """
    Функция для поиска параметра сглаживания по обоим осям.
    :param T: Двумерный массив температур на текущем временном слое.
    :return: Максимальный температурный интервал содержащий границу ф.п.
    """
    n_y, n_x = T.shape
    delta = 0.0
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            if (T[j + 1, i] - T_0) * (T[j, i] - T_0) < 0.0:
                temp = abs(T[j + 1, i] - T[j, i])
                delta = temp if temp > delta else delta
                break
            if (T[j, i + 1] - T_0) * (T[j, i] - T_0) < 0.0:
                temp = abs(T[j, i + 1] - T[j, i])
                delta = temp if temp > delta else delta
                break
    return delta


def init_temperature(geom: DomainGeometry, F: Optional[list | ndarray] = None):
    T = np.empty((geom.n_y, geom.n_x))

    if F is None:
        # Линейное изменение температуры от T_ICE_MIN на нижней границе y = 0 до T_WATER_MAX на верхней границе
        T[0, :] = T_ICE_MIN
        T[geom.n_y-1, :] = T_WATER_MAX
        for j in range(1, geom.n_y):
            T[j, :] = T_ICE_MIN + j * (T_WATER_MAX - T_ICE_MIN) / geom.n_y
    else:
        # Линейное изменение температуры с учетом начального положения границы ф.п.
        for i in range(geom.n_x):
            for j in range(geom.n_y):
                if j * geom.dy < F[i]:
                    # T[j, i] = T_ICE_MIN
                    T[j, i] = T_ICE_MIN + j * geom.dy * (T_0 - T_ICE_MIN) / (geom.height - WATER_H)
                elif j * geom.dy > F[i]:
                    T[j, i] = T_WATER_MAX
                else:
                    T[j, i] = T_0

    return T


def init_temperature_angle(geom: DomainGeometry):
    T = np.empty((geom.n_y, geom.n_x))
    T[:, :] = T_WATER_MAX
    T[0, :] = T_ICE_MIN
    T[:, 0] = T_ICE_MIN

    return T


def init_temperature_test(geom: DomainGeometry):
    T = np.empty((geom.n_y, geom.n_x))
    T[:, :] = T_WATER_MAX
    T[0, :] = T_ICE_MIN

    return T


def init_temperature_circle(geom: DomainGeometry, water_temp: float, ice_temp: float) -> ndarray:
    T = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        for j in range(geom.n_y):
            if (i * geom.dx - geom.width / 2.0)**2 + (j * geom.dy - geom.height / 2.0)**2 < 0.0625:
                T[j, i] = water_temp
            else:
                T[j, i] = ice_temp

    return T


def init_temperature_pacman(geom: DomainGeometry):
    T = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        for j in range(geom.n_y):
            if (i * geom.dx - geom.width / 2.0)**2 + (j * geom.dy - geom.height / 2.0)**2 < 0.0625:
                if i * geom.dx <= j * geom.dy <= - i * geom.dx + 1:
                    T[j, i] = T_ICE_MIN
                elif (i * geom.dx - 0.6) ** 2 + (j * geom.dy - 0.6) ** 2 < 0.0025:
                    T[j, i] = T_ICE_MIN
                else:
                    T[j, i] = T_WATER_MAX
            else:
                T[j, i] = T_ICE_MIN

    return T


def init_temperature_double_circle(geom: DomainGeometry):
    T = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        for j in range(geom.n_y):
            if (i * geom.dx - geom.width / 2.0)**2 + (j * geom.dy - 0.75)**2 < 0.04:
                T[j, i] = T_WATER_MAX
            elif (i * geom.dx - geom.width / 2.0)**2 + (j * geom.dy - 0.25)**2 < 0.04:
                T[j, i] = T_WATER_MAX
            else:
                T[j, i] = T_ICE_MIN * (1.0 - j / geom.n_y)

    return T


def init_temperature_square(geom: DomainGeometry):
    T = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        for j in range(geom.n_y):
            if abs(i * geom.dx - geom.width / 2.0) < 0.25 and abs(j * geom.dy - geom.height / 2.0) < 0.25:
                T[j, i] = T_WATER_MAX
            else:
                T[j, i] = T_ICE_MIN

    return T


def init_temperature_2f_test(geom: DomainGeometry, water_temp: float, ice_temp: float, F: ndarray) -> ndarray:
    T = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        for j in range(geom.n_y):
            if j * geom.dy < F[i]:
                T[j][i] = ice_temp
            else:
                T[j][i] = water_temp

    return T


def init_temperature_lake(geom: DomainGeometry, water_temp: float, ice_temp: float):
    water_th_grid = np.load("../data/lake.npz")["water"]
    ice_th_grid = np.load("../data/lake.npz")["ice"]

    grid_x = water_th_grid[0]
    grid_step = grid_x[1] - grid_x[0]
    print(f"Grid step: {grid_step}")

    lake_width = grid_x[-1]
    print(f"Lake width: {lake_width}")

    new_x = [i * geom.dx for i in range(int(lake_width / geom.dx + 1))]
    print(new_x[-1], len(new_x))

    water_th_interp = np.interp(
        new_x,
        grid_x,
        water_th_grid[1]
    )

    ice_th_interp = np.interp(
        new_x,
        grid_x,
        ice_th_grid[1]
    )

    print(f"Max lake thickness {max(water_th_interp)}")

    T = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        x = i * geom.dx
        ice_th_at_x, water_th_at_x = 0.0, 0.0

        if (geom.width - lake_width) / 2.0 <= x <= (geom.width + lake_width) / 2.0:
            water_th_at_x = water_th_interp[i + int((len(new_x) - geom.n_x) / 2)]
            ice_th_at_x = ice_th_interp[i + int((len(new_x) - geom.n_x) / 2)]

        for j in range(geom.n_y):
            y = geom.height - j * geom.dy
            if water_th_at_x > 0.0 and ice_th_at_x <= y <= ice_th_at_x + water_th_at_x:
                T[j, i] = water_temp
            else:
                T[j, i] = ice_temp * (1.0 - j / geom.n_y)
    return T
