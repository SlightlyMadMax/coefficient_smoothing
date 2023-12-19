import numpy as np
import math
import numba

from numba.typed import Dict
from src.parameters import N_X, N_Y, dx, dy, T_0, T_ICE_MIN, T_WATER_MAX, HEIGHT, WATER_H, CREV_DEPTH


def init_bc():
    boundary_conditions = Dict()

    bc_bottom = Dict()
    bc_bottom["type"] = 1.0
    bc_bottom["temp"] = T_ICE_MIN

    bc_top = Dict()
    bc_top["type"] = 1.0
    bc_top["temp"] = T_WATER_MAX

    bc_sides = Dict()
    bc_sides["type"] = 2.0

    boundary_conditions["left"] = bc_sides
    boundary_conditions["right"] = bc_sides
    boundary_conditions["bottom"] = bc_bottom
    boundary_conditions["top"] = bc_top

    return boundary_conditions


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
def get_phase_trans_boundary(T, n_x: int = N_X):
    """
    Функция для определения границы фазового перехода по изотерме T(x, y) = T_0
    :param T: Двумерный массив температур на текущем временном слое.
    :param n_x: Число узлов по оси X.
    :return: Кортеж векторов (X, Y) с точками границы фазового перехода на текущем временном слое.
    """
    X = []
    Y = []

    for j in range(1, N_Y - 1):
        for i in range(1, n_x - 1):
            if (T[j, i] - T_0) * (T[j + 1, i] - T_0) < 0.0:
                if abs(T[j, i] - T_0) <= abs(T[j + 1, i] - T_0):
                    Y.append(j * dy)
                else:
                    Y.append((j + 1) * dy)
                X.append(i*dx)
            elif (T[j, i] - T_0) * (T[j, i + 1] - T_0) < 0.0:
                if abs(T[j, i] - T_0) <= abs(T[j, i + 1] - T_0):
                    X.append(i * dx)
                else:
                    X.append((i + 1) * dx)
                Y.append(j*dy)

    return X, Y
