import numpy as np
import math
from numpy import ndarray

import src.parameters as cfg
from src.geometry import DomainGeometry


def init_boundary(geom: DomainGeometry):
    """
    Функция для задания начального положения границы фазового перехода.
    :param n_x: Число узлов по оси X.
    :return: Вектор координат границы фазового перехода.
    """
    F = np.empty(geom.n_x)

    # Трещина-гауссиана
    F[:] = [geom.height - cfg.WATER_H - cfg.CREV_DEPTH * math.exp(-(i * geom.dx - 0.5) ** 2 / 0.005) for i in range(geom.n_x)]

    return F


def get_phase_trans_boundary(T: ndarray, geom: DomainGeometry):
    X = []
    Y = []

    for j in range(1, geom.n_y - 1):
        for i in range(1, geom.n_x - 1):
            if (T[j, i] - cfg.T_0) * (T[j + 1, i] - cfg.T_0) < 0.0:
                y_0 = abs((T[j, i] * (j + 1) * geom.dy - T[j + 1, i] * j * geom.dy) / (T[j, i] - T[j + 1, i]))
                Y.append(y_0)
                X.append(i*geom.dx)
            elif (T[j, i] - cfg.T_0) * (T[j, i + 1] - cfg.T_0) < 0.0:
                x_0 = abs((T[j, i] * (i + 1) * geom.dx - T[j, i + 1] * i * geom.dx) / (T[j, i] - T[j, i + 1]))
                X.append(x_0)
                Y.append(j*geom.dy)

    return X, Y
