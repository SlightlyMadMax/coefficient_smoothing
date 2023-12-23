import numpy as np
import numba

from math import sin, cos, pi
from src.parameters import N_X, N_Y, T_ICE_MIN, T_WATER_MAX, T_0, Q_SOL, LAT, DECL, RAD_SPEED, T_air, T_amp, \
    HEIGHT, WIDTH, WATER_H, dx, dy


@numba.jit(nopython=True)
def solar_heat(t: float):
    """
    Функция для вычисления потока солнечной радиации.
    :param t: Время в секундах.
    :return: Величина солнечного потока радиации на горизонтальную поверхность при заданных параметрах
    в момент времени t. [Вт/м^2]
    """
    return Q_SOL * (
            sin(LAT) * sin(DECL) +
            cos(LAT) * cos(DECL) * cos(RAD_SPEED * t + 12.0 * 3600.0)
    )


@numba.jit(nopython=True)
def air_temperature(t: float):
    """
    Функция изменения температуры воздуха.
    :param t: Время в секундах
    :return: Температура воздуха в заданный момент времени
    """
    return T_air + T_amp * sin(2 * pi * t / (24.0 * 3600.0) - pi / 2)


@numba.jit(nopython=True)
def get_max_delta(T) -> float:
    """
    Функция для поиска параметра сглаживания по оси y.
    :param T: Двумерный массив температур на текущем временном слое.
    :return: Максимальный температурный интервал содержащий границу ф.п.
    """
    delta = 0.0

    for i in range(N_X - 1):
        for j in range(N_Y - 1):
            if (T[j + 1, i] - T_0) * (T[j, i] - T_0) < 0.0:
                temp = abs(T[j + 1, i] - T[j, i])
                delta = temp if temp > delta else delta
                break
            if (T[j, i + 1] - T_0) * (T[j, i] - T_0) < 0.0:
                temp = abs(T[j, i + 1] - T[j, i])
                delta = temp if temp > delta else delta
                break
    return delta


@numba.jit(nopython=True)
def get_max_delta_y(T) -> float:
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

    return delta_y


@numba.jit(nopython=True)
def get_max_delta_x(T) -> float:
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

    return delta_x


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
                    # T[j, i] = T_ICE_MIN
                    T[j, i] = T_ICE_MIN + j * dy * (T_0 - T_ICE_MIN) / (HEIGHT - WATER_H)
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


def init_temperature_test():
    T = np.empty((N_Y, N_X))
    T[:, :] = T_WATER_MAX
    T[0, :] = T_ICE_MIN

    return T


def init_temperature_circle():
    T = np.empty((N_Y, N_X))

    for i in range(N_X):
        for j in range(N_Y):
            if (i * dx - WIDTH / 2.0)**2 + (j * dy - WIDTH / 2.0)**2 < 0.0625:
                T[j, i] = T_WATER_MAX
            else:
                T[j, i] = T_ICE_MIN

    return T


def init_temperature_pacman():
    T = np.empty((N_Y, N_X))

    for i in range(N_X):
        for j in range(N_Y):
            if (i * dx - WIDTH / 2.0)**2 + (j * dy - WIDTH / 2.0)**2 < 0.0625:
                if i*dx <= j * dy <= - i * dx + 1:
                    T[j, i] = T_ICE_MIN
                elif (i * dx - 0.6) ** 2 + (j * dy - 0.6) ** 2 < 0.0025:
                    T[j, i] = T_ICE_MIN
                else:
                    T[j, i] = T_WATER_MAX
            else:
                T[j, i] = T_ICE_MIN

    return T


def init_temperature_double_circle():
    T = np.empty((N_Y, N_X))

    for i in range(N_X):
        for j in range(N_Y):
            if (i * dx - WIDTH / 2.0)**2 + (j * dy - 0.75)**2 < 0.04:
                T[j, i] = T_WATER_MAX
            elif (i * dx - WIDTH / 2.0)**2 + (j * dy - 0.25)**2 < 0.04:
                T[j, i] = T_WATER_MAX
            else:
                T[j, i] = T_ICE_MIN * (1.0 - j / N_Y)

    return T


def init_temperature_square():
    T = np.empty((N_Y, N_X))

    for i in range(N_X):
        for j in range(N_Y):
            if abs(i * dx - WIDTH / 2.0) < 0.25 and abs(j * dy - WIDTH / 2.0) < 0.25:
                T[j, i] = T_WATER_MAX
            else:
                T[j, i] = T_ICE_MIN

    return T
