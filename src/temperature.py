import numpy as np
import numba

from src.parameters import N_X, N_Y, T_ICE_MIN, T_WATER_MAX, dy, T_0, delta


@numba.jit(nopython=True)
def get_delta_y(T) -> float:
    """
    Функция для поиска параметра сглаживания по оси y.
    :param T: Двумерный массив температур на текущем временном слое.
    :return: Максимальный температурный интервал вдоль оси y, содержащий границу ф.п.
    """
    delta_y = 0.0

    for i in range(N_X):
        for j in range(0, N_Y - 1):
            if (T[j + 1, i] - T_0) * (T[j, i] - T_0) < 0.0:
                temp = abs(T[j + 1, i] - T[j, i])
                delta_y = temp if temp > delta_y else delta_y
                break

    return delta_y or delta


@numba.jit(nopython=True)
def get_delta_x(T) -> float:
    """
    Функция для поиска параметра сглаживания по оси x.
    :param T: Двумерный массив температур на текущем временном слое.
    :return: Максимальный температурный интервал вдоль оси x, содержащий границу ф.п.
    """
    delta_x = 0.0

    for j in range(N_Y):
        for i in range(0, N_X - 1):
            if (T[j, i + 1] - T_0) * (T[j, i] - T_0) < 0.0:
                temp = abs(T[j, i + 1] - T[j, i])
                delta_x = temp if temp > delta_x else delta_x
                break

    return delta_x or delta


def init_temperature(F = None):
    """
    Функция для задания начального распределения температуры.
    :param F: Необязательный параметр, вектор координат начального положения границы фазового перехода.
    :return: Двумерный массив температур в начальный момент времени.
    """
    T = np.empty((N_Y, N_X))

    if F is None:
        # Линейное изменение температуры от T_ICE_MIN на нижней границе y = 0 до T_WATER_MAX на верхней границе
        T[0, :] = T_ICE_MIN
        T[N_Y-1, :] = T_WATER_MAX
        for j in range(1, N_Y):
            T[j, :] = T_ICE_MIN + j * (T_WATER_MAX - T_ICE_MIN) / N_Y
    else:
        # Линейное изменение температуры с учетом начального положения границы ф.п.
        for i in range(N_X):
            for j in range(N_Y):
                if j * dy < F[i]:
                    T[j, i] = T_ICE_MIN + j * (T_0 - T_ICE_MIN) / N_Y
                elif j * dy > F[i]:
                    T[j, i] = T_WATER_MAX
                else:
                    T[j, i] = T_0

    return T


def init_temperature_angle():
    T = np.empty((N_Y, N_X))
    T[:, :] = T_WATER_MAX
    T[0, :] = T_ICE_MIN
    T[:, 0] = T_ICE_MIN

    return T


def init_temperature_1d_1f_test():
    T = np.empty((N_Y, N_X))
    T[:, :] = T_WATER_MAX
    T[0, :] = T_ICE_MIN

    return T
