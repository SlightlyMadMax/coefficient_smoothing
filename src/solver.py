import numpy as np
import numba
from numba.typed import Dict

from src.coefficients import c_smoothed, k_smoothed
from src.parameters import N_Y, N_X, inv_dx, inv_dy, dt, dy, T_ICE_MIN, T_WATER_MAX, delta, K_WATER, CONV_COEF
from src.temperature import get_delta_y, get_delta_x, solar_heat, air_temperature


@numba.jit(nopython=True)
def solve(T, boundary_conditions: Dict, time: float, _delta: float = delta, fixed_delta: bool = True):
    """
    Функция для нахождения решения задачи Стефана в обобщенной формулировке с помощью локально-одномерной
    линеаризованной разностной схемы.
    :param T: Двумерный массив температур на текущем временном слое.
    :param boundary_conditions: Словарь, в котором указан тип граничного условия для каждой границы.
    :param time: Время в секундах на следующем шаге сетки. Используется для определения граничных условий.
    :param _delta: Параметр сглаживания.
    :param fixed_delta: Определяет зафиксирован ли параметр сглаживания.
    Если задано значение False – параметр _delta будет определяться адаптивно на каждом шаге.
    :return: Двумерный массив температур на новом временном слое.
    """
    # Массив для хранения значений температуры на промежуточном слое
    tempT = np.empty((N_Y, N_X))

    # Векторы для хранения значений прогоночных коэффициентов (см. Кольцова Э. М. и др.
    # "Численные методы решения уравнений математической физики и химии" 2021, стр. 43)
    alpha = np.empty((N_X - 1), )
    beta = np.empty((N_X - 1), )

    if boundary_conditions["left"] == 1:
        alpha[0] = 0
        beta[0] = T_ICE_MIN
    else:
        alpha[0] = 1
        beta[0] = 0

    if not fixed_delta:
        _delta = get_delta_x(T)

    # Прогонка по X
    for j in range(1, N_Y - 1):
        for i in range(1, N_X - 1):
            inv_c = 1.0 / c_smoothed(T[j, i], _delta)

            # Коэффициент при T_(i-1,j)^(n+1/2)
            a_i = -dt * k_smoothed(0.5 * (T[j, i + 1] + T[j, i]), _delta) * inv_c * inv_dx * inv_dx
            # Коэффициент при T_(i,j)^(n+1/2)
            b_i = (
                    1.0 +
                    dt * (k_smoothed(0.5 * (T[j, i + 1] + T[j, i]), _delta) + k_smoothed(0.5 * (T[j, i] + T[j, i - 1]), _delta)) *
                    inv_c * inv_dx * inv_dx
            )
            # Коэффициент при T_(i+1,j)^(n+1/2)
            c_i = -dt * k_smoothed(0.5 * (T[j, i] + T[j, i - 1]), _delta) * inv_c * inv_dx * inv_dx

            # Расчет прогоночных коэффициентов
            alpha[i] = -a_i / (b_i + c_i * alpha[i - 1])
            beta[i] = (T[j, i] - c_i * beta[i - 1]) / (b_i + c_i * alpha[i - 1])

        if boundary_conditions["right"] == 1:
            tempT[j, N_X - 1] = T_ICE_MIN
        else:
            tempT[j, N_X - 1] = beta[N_X - 2]/(1.0 - alpha[N_X - 2])

        # Вычисление температуры на промежуточном временном слое
        for i in range(N_X - 2, -1, -1):
            tempT[j, i] = alpha[i] * tempT[j, i + 1] + beta[i]

    # Массив для хранения значений температуры на новом временном слое
    new_T = np.empty((N_Y, N_X))

    alpha = np.empty((N_Y - 1), )
    beta = np.empty((N_Y - 1), )

    alpha[0] = 0  # из левого граничного условия первого рода (по Y)
    beta[0] = T_ICE_MIN  # из левого граничного условия первого рода (по Y)

    if not fixed_delta:
        _delta = get_delta_y(tempT)

    # Прогонка по Y
    for i in range(1, N_X - 1):
        for j in range(1, N_Y - 1):
            inv_c = 1.0 / c_smoothed(tempT[j, i], _delta)

            # Коэффициент при T_(i,j-1)^n
            a_j = -dt * k_smoothed(0.5 * (tempT[j + 1, i] + tempT[j, i]), _delta) * inv_c * inv_dy * inv_dy
            # Коэффициент при T_(i,j)^n
            b_j = (
                    1.0 +
                    dt * (k_smoothed(0.5 * (tempT[j + 1, i] + tempT[j, i]), _delta) +
                          k_smoothed(0.5 * (tempT[j, i] + tempT[j - 1, i]), _delta)) *
                    inv_c * inv_dy * inv_dy
            )
            # Коэффициент при T_(i,j+1)^n
            c_j = -dt * k_smoothed(0.5 * (tempT[j, i] + tempT[j - 1, i]), _delta) * inv_c * inv_dy * inv_dy

            # Расчет прогоночных коэффициентов
            alpha[j] = -a_j / (b_j + c_j * alpha[j - 1])
            beta[j] = (tempT[j, i] - c_j * beta[j - 1]) / (b_j + c_j * alpha[j - 1])

        if boundary_conditions["upper"] == 1:
            new_T[N_Y - 1, i] = T_ICE_MIN
        elif boundary_conditions["upper"] == 2:
            new_T[N_Y - 1, i] = beta[N_Y - 2]/(1.0 - alpha[N_Y - 2])
        else:
            # Определяем температуру воздуха у поверхности
            T_air_t = air_temperature(time)
            # Определяем тепловой поток солнечной энергии
            Q_sol = solar_heat(time)
            new_T[N_Y - 1, i] = (dy * (Q_sol - CONV_COEF * T_air_t) / K_WATER +
                                 beta[N_Y - 2])/(1 - alpha[N_Y - 2] - dy * CONV_COEF / K_WATER)

        # Вычисление температуры на новом временном слое
        for j in range(N_Y - 2, -1, -1):
            new_T[j, i] = alpha[j] * new_T[j + 1, i] + beta[j]

    if boundary_conditions["left"] == 1:
        new_T[:, 0] = T_ICE_MIN
    else:
        new_T[:, 0] = new_T[:, 1]

    if boundary_conditions["right"] == 1:
        new_T[:, N_X - 1] = T_ICE_MIN
    else:
        new_T[:, N_X - 1] = new_T[:, N_X - 2]

    return new_T


@numba.jit(nopython=True)
def solve_alt_dir(T, _delta: float):
    """
    Функция для нахождения решения задачи Стефана в обобщенной формулировке с линеаризованной классической схемы
    переменных направлений (Писмена-Рэкфорда).
    :param T: Двумерный массив температур на текущем временном слое.
    :param _delta: Параметр сглаживания
    :return: Двумерный массив температур на новом временном слое.
    """
    tempT = np.empty((N_Y, N_X))

    alpha = np.empty((N_X - 1), )
    beta = np.empty((N_X - 1), )

    alpha[0] = 1  # из левого граничного условия второго рода
    beta[0] = 0  # из левого граничного условия второго рода

    for j in range(1, N_Y - 1):
        for i in range(1, N_X - 1):
            inv_c = 1.0 / c_smoothed(T[j, i], _delta)
            a_i = -dt * 0.5 * k_smoothed(0.5 * (T[j, i + 1] + T[j, i]), _delta) * inv_c * inv_dx * inv_dx
            b_i = (
                    1.0 +
                    dt * (k_smoothed(0.5 * (T[j, i + 1] + T[j, i]), _delta) + k_smoothed(0.5 * (T[j, i] + T[j, i - 1]), _delta)) *
                    inv_c * inv_dx * inv_dx * 0.5
            )
            c_i = -dt * 0.5 * k_smoothed(0.5 * (T[j, i] + T[j, i - 1]), _delta) * inv_c * inv_dx * inv_dx

            rhs_i = T[j, i] + dt * 0.5 * inv_c * inv_dy * inv_dy * \
                    (k_smoothed(0.5 * (T[j + 1, i] + T[j, i]), _delta) * (T[j + 1, i] - T[j, i]) -
                     k_smoothed(0.5 * (T[j, i] + T[j - 1, i]), _delta) * (T[j, i] - T[j - 1, i]))

            alpha[i] = -a_i / (b_i + c_i * alpha[i - 1])
            beta[i] = (rhs_i - c_i * beta[i - 1]) / (b_i + c_i * alpha[i - 1])

        tempT[j, N_X - 1] = beta[N_X - 2]/(1.0 - alpha[N_X - 2])  # из правого граничного условия второго рода

        for i in range(N_X - 2, -1, -1):
            tempT[j, i] = alpha[i] * tempT[j, i + 1] + beta[i]

    tempT[0, :] = T_ICE_MIN
    tempT[N_Y-1, :] = T_WATER_MAX

    new_T = np.empty((N_Y, N_X))

    alpha = np.empty((N_Y - 1), )
    beta = np.empty((N_Y - 1), )

    alpha[0] = 0  # из левого граничного условия первого рода
    beta[0] = T_ICE_MIN  # из левого граничного условия первого рода

    for i in range(1, N_X - 1):
        for j in range(1, N_Y - 1):
            inv_c = 1.0 / c_smoothed(tempT[j, i], _delta)
            a_j = -dt * 0.5 * k_smoothed(0.5 * (tempT[j + 1, i] + tempT[j, i]), _delta) * inv_c * inv_dy * inv_dy
            b_j = (
                    1.0 +
                    dt * (k_smoothed(0.5 * (tempT[j + 1, i] + tempT[j, i]), _delta) + k_smoothed(0.5 * (tempT[j, i] + tempT[j - 1, i]), _delta)) *
                    inv_c * inv_dy * inv_dy * 0.5
            )
            c_j = -dt * 0.5 * k_smoothed(0.5 * (tempT[j, i] + tempT[j - 1, i]), _delta) * inv_c * inv_dy * inv_dy

            rhs_j = tempT[j, i] + dt * 0.5 * inv_c * inv_dx * inv_dx * \
                    (k_smoothed(0.5 * (tempT[j, i + 1] + tempT[j, i]), _delta) * (tempT[j, i + 1] - tempT[j, i]) -
                     k_smoothed(0.5 * (tempT[j, i] + tempT[j, i - 1]), _delta) * (tempT[j, i] - tempT[j, i - 1]))

            alpha[j] = -a_j / (b_j + c_j * alpha[j - 1])
            beta[j] = (rhs_j - c_j * beta[j - 1]) / (b_j + c_j * alpha[j - 1])

        new_T[N_Y - 1, i] = beta[N_Y - 2]/(1.0 - alpha[N_Y - 2])  # из правого граничного условия второго рода

        for j in range(N_Y - 2, -1, -1):
            new_T[j, i] = alpha[j] * new_T[j + 1, i] + beta[j]

    new_T[:, 0] = new_T[:, 1]
    new_T[:, N_X - 1] = new_T[:, N_X - 2]

    return new_T
