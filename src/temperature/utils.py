import numba
import numpy as np
from enum import Enum
from numpy.typing import NDArray

import src.parameters as cfg


class TemperatureUnit(Enum):
    CELSIUS = 1, "Celsius"
    KELVIN = 2, "Kelvin"


@numba.jit(nopython=True)
def is_frozen(T: NDArray[np.float64]) -> bool:
    """
    Определяет, произошло ли замерзание всей воды (отсутствие границы фазового перехода).

    :param T: двумерный массив температур
    :return: True -- если граница ф.п. отсутствует, иначе False
    """
    N_X, N_Y = T.shape
    for i in range(N_X - 1):
        for j in range(N_Y - 1):
            if (T[j + 1, i] - cfg.T_0) * (T[j, i] - cfg.T_0) < 0.0 or (
                T[j, i + 1] - cfg.T_0
            ) * (T[j, i] - cfg.T_0) < 0.0:
                return False
    return True


@numba.jit(nopython=True)
def get_crev_depth(T: NDArray[np.float64]) -> float:
    """
    Определяет глубину трещины.

    :param T: двумерный массив температур
    :return: максимальная глубина трещины
    """
    N_X, N_Y = T.shape
    i = int(N_X / 2)
    for j in range(N_Y - 1):
        if (T[j + 1, i] - cfg.T_0) * (T[j, i] - cfg.T_0) < 0.0 or (
            T[j, i + 1] - cfg.T_0
        ) * (T[j, i] - cfg.T_0) < 0.0:
            return cfg.HEIGHT - cfg.WATER_H - j * cfg.dy


@numba.jit(nopython=True)
def get_water_thickness(T: NDArray[np.float64], dy: float) -> float:
    n_y, n_x = T.shape
    bottom, top = 0.0, 0.0
    for j in range(n_y - 1):
        if T[j + 1, n_x // 2] > cfg.T_0 and T[j, n_x // 2] <= cfg.T_0:
            bottom = j * dy
            continue
        if T[j + 1, n_x // 2] <= cfg.T_0 and T[j, n_x // 2] > cfg.T_0:
            top = j * dy
            break
    return top - bottom
