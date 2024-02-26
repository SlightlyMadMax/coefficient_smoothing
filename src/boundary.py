import numpy as np
import math
import numba

from src.parameters import N_X, N_Y, dx, dy, T_0, HEIGHT, WATER_H, CREV_DEPTH


def init_boundary(n_x: int = N_X):
    """
    Функция для задания начального положения границы фазового перехода.
    :param n_x: Число узлов по оси X.
    :return: Вектор координат границы фазового перехода.
    """
    F = np.empty(n_x)

    # Трещина-гауссиана
    F[:] = [HEIGHT - WATER_H - CREV_DEPTH * math.exp(-(i * dx - 0.5) ** 2 / 0.005) for i in range(n_x)]

    return F


@numba.jit(nopython=True)
def get_phase_trans_boundary(T, n_x: int = N_X, n_y: int = N_Y):
    """
    Функция для определения границы фазового перехода по изотерме T(x, y) = T_0
    :param T: Двумерный массив температур на текущем временном слое.
    :param n_x: Число узлов по оси X.
    :param n_y: Число узлов по оси Y.
    :return: Кортеж векторов (X, Y) с точками границы фазового перехода на текущем временном слое.
    """
    X = []
    Y = []

    for j in range(1, n_y - 1):
        for i in range(1, n_x - 1):
            if (T[j, i] - T_0) * (T[j + 1, i] - T_0) < 0.0:
                y_0 = abs((T[j, i] * (j + 1) * dy - T[j + 1, i] * j * dy) / (T[j, i] - T[j + 1, i]))
                Y.append(y_0)
                X.append(i*dx)
            elif (T[j, i] - T_0) * (T[j, i + 1] - T_0) < 0.0:
                x_0 = abs((T[j, i] * (i + 1) * dx - T[j, i + 1] * i * dx) / (T[j, i] - T[j, i + 1]))
                X.append(x_0)
                Y.append(j*dy)

    return X, Y
