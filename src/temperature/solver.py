import numpy as np
from numpy import ndarray
import numba

from src.temperature.coefficient_smoothing.coefficients import c_smoothed, k_smoothed
import src.parameters as cfg
from src.temperature import boundary_conditions as bc
from src.temperature.coefficient_smoothing.delta import get_max_delta


@numba.jit(nopython=True)
def solve(
    T: ndarray,
    temp_T: ndarray,
    new_T: ndarray,
    alpha: ndarray,
    beta: ndarray,
    top_cond_type: int,
    right_cond_type: int,
    bottom_cond_type: int,
    left_cond_type: int,
    dx: float,
    dy: float,
    dt: float,
    time: float = 0.0,
    fixed_delta: bool = True,
) -> ndarray:
    """
    Функция для нахождения решения задачи Стефана в обобщенной формулировке с помощью локально-одномерной
    линеаризованной разностной схемы.

    :param T: двумерный массив температур на текущем временном слое.
    :param temp_T: массив для хранения значений температуры на промежуточном слое
    :param new_T: массив для хранения температуры на новом временном слое
    :param alpha: массив для хранения коэффициентов обратного хода прогонки
    :param beta: массив для хранения коэффициентов обратного хода прогонки
    :param top_cond_type: тип граничного условия на верхней границе
    :param right_cond_type: тип граничного условия на правой границе
    :param bottom_cond_type: тип граничного условия на нижней границе
    :param left_cond_type: тип граничного условия на левой границе
    :param dx: шаг пространственной сетки по координате x
    :param dy: шаг пространственной сетки по координате y
    :param dt: шаг по времени
    :param time: время в секундах на следующем шаге сетки. Используется для определения граничных условий.
    :param fixed_delta: определяет зафиксирован ли параметр сглаживания. Если задано значение False – параметр _delta
    будет определяться адаптивно на каждом шаге.
    :return: двумерный массив температур на новом временном слое.
    """
    n_y, n_x = T.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    _delta = cfg.delta
    if not fixed_delta:
        _delta = get_max_delta(T)

    lbc = bc.get_left_bc_1(time, n_y)
    rbc = bc.get_right_bc_1(time, n_y)
    bbc = bc.get_bottom_bc_1(time, n_x)
    tbc = bc.get_top_bc_1(time, n_x)
    phi = bc.get_top_bc_2(time)
    psi, ksi = bc.get_top_bc_3(time)

    for j in range(1, n_y - 1):
        if left_cond_type == cfg.DIRICHLET:
            alpha[0] = 0.0
            beta[0] = lbc[j]
        else:
            alpha[0] = 1.0
            beta[0] = 0.0
        for i in range(1, n_x - 1):
            inv_c = 1.0 / c_smoothed(T[j, i], _delta)

            # Коэффициент при T_(i+1,j)^(n+1/2)
            a_i = (
                -dt
                * k_smoothed(0.5 * (T[j, i + 1] + T[j, i]), _delta)
                * inv_c
                * inv_dx2
            )
            # Коэффициент при T_(i,j)^(n+1/2)
            b_i = (
                1.0
                + dt
                * (
                    k_smoothed(0.5 * (T[j, i + 1] + T[j, i]), _delta)
                    + k_smoothed(0.5 * (T[j, i] + T[j, i - 1]), _delta)
                )
                * inv_c
                * inv_dx2
            )
            # Коэффициент при T_(i-1,j)^(n+1/2)
            c_i = (
                -dt
                * k_smoothed(0.5 * (T[j, i] + T[j, i - 1]), _delta)
                * inv_c
                * inv_dx2
            )

            # Расчет прогоночных коэффициентов
            alpha[i] = -a_i / (b_i + c_i * alpha[i - 1])
            beta[i] = (T[j, i] - c_i * beta[i - 1]) / (b_i + c_i * alpha[i - 1])

        if right_cond_type == 1:
            temp_T[j, n_x - 1] = rbc[j]
        else:  # NEUMANN
            temp_T[j, n_x - 1] = beta[n_x - 2] / (1.0 - alpha[n_x - 2])

        # Вычисление температуры на промежуточном временном слое
        for i in range(n_x - 2, -1, -1):
            temp_T[j, i] = alpha[i] * temp_T[j, i + 1] + beta[i]

    temp_T[0, :] = bbc

    if top_cond_type == cfg.DIRICHLET:
        temp_T[n_y - 1, :] = tbc
    elif top_cond_type == cfg.NEUMANN:
        temp_T[n_y - 1, :] = (dy * phi + beta[n_y - 2]) / (1.0 - alpha[n_y - 2])
    else:  # ROBIN
        temp_T[n_y - 1, :] = (dy * psi + beta[n_y - 2]) / (1 - alpha[n_y - 2] - dy * ksi)

    # Second sweep (y-direction)
    if not fixed_delta:
        _delta = get_max_delta(temp_T)

    # Прогонка по Y
    for i in range(1, n_x - 1):
        if bottom_cond_type == cfg.DIRICHLET:
            alpha[0] = 0.0
            beta[0] = bbc[i]
        else:
            alpha[0] = 1.0
            beta[0] = 0.0
        for j in range(1, n_y - 1):
            inv_c = 1.0 / c_smoothed(temp_T[j, i], _delta)

            # Коэффициент при T_(i,j-1)^n
            a_j = (
                -dt
                * k_smoothed(0.5 * (temp_T[j + 1, i] + temp_T[j, i]), _delta)
                * inv_c
                * inv_dy2
            )
            # Коэффициент при T_(i,j)^n
            b_j = (
                1.0
                + dt
                * (
                    k_smoothed(0.5 * (temp_T[j + 1, i] + temp_T[j, i]), _delta)
                    + k_smoothed(0.5 * (temp_T[j, i] + temp_T[j - 1, i]), _delta)
                )
                * inv_c
                * inv_dy2
            )
            # Коэффициент при T_(i,j+1)^n
            c_j = (
                -dt
                * k_smoothed(0.5 * (temp_T[j, i] + temp_T[j - 1, i]), _delta)
                * inv_c
                * inv_dy2
            )

            # Расчет прогоночных коэффициентов
            alpha[j] = -a_j / (b_j + c_j * alpha[j - 1])
            beta[j] = (temp_T[j, i] - c_j * beta[j - 1]) / (b_j + c_j * alpha[j - 1])

        if top_cond_type == cfg.DIRICHLET:
            new_T[n_y - 1, i] = tbc[i]
        elif top_cond_type == cfg.NEUMANN:
            new_T[n_y - 1, i] = (dy * phi + beta[n_y - 2]) / (1.0 - alpha[n_y - 2])
        else:  # ROBIN
            new_T[n_y - 1, i] = (dy * psi + beta[n_y - 2]) / (
                1 - alpha[n_y - 2] - dy * ksi
            )

        # Вычисление температуры на новом временном слое
        for j in range(n_y - 2, -1, -1):
            new_T[j, i] = alpha[j] * new_T[j + 1, i] + beta[j]

    if left_cond_type == cfg.DIRICHLET:
        new_T[:, 0] = lbc
    else:  # NEUMANN
        new_T[:, 0] = new_T[:, 1]

    if right_cond_type == cfg.DIRICHLET:
        new_T[:, n_x - 1] = rbc
    else:  # NEUMANN
        new_T[:, n_x - 1] = new_T[:, n_x - 2]

    return new_T


@numba.jit(nopython=True)
def solve_alt_dir(T: ndarray, dx: float, dy: float, dt: float, _delta: float):
    """
    Функция для нахождения решения задачи Стефана в обобщенной формулировке с линеаризованной классической схемы
    переменных направлений (Писмена-Рэкфорда).

    :param T: Двумерный массив температур на текущем временном слое.
    :param _delta: Параметр сглаживания
    :return: Двумерный массив температур на новом временном слое.
    """
    n_y, n_x = T.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    tempT = np.empty((n_y, n_x))

    alpha = np.empty(
        (n_x - 1),
    )
    beta = np.empty(
        (n_x - 1),
    )

    alpha[0] = 1  # из левого граничного условия второго рода
    beta[0] = 0  # из левого граничного условия второго рода

    for j in range(1, n_y - 1):
        for i in range(1, n_x - 1):
            inv_c = 1.0 / c_smoothed(T[j, i], _delta)
            a_i = (
                -dt
                * 0.5
                * k_smoothed(0.5 * (T[j, i + 1] + T[j, i]), _delta)
                * inv_c
                * inv_dx
                * inv_dx
            )
            b_i = (
                1.0
                + dt
                * (
                    k_smoothed(0.5 * (T[j, i + 1] + T[j, i]), _delta)
                    + k_smoothed(0.5 * (T[j, i] + T[j, i - 1]), _delta)
                )
                * inv_c
                * inv_dx
                * inv_dx
                * 0.5
            )
            c_i = (
                -dt
                * 0.5
                * k_smoothed(0.5 * (T[j, i] + T[j, i - 1]), _delta)
                * inv_c
                * inv_dx
                * inv_dx
            )

            rhs_i = T[j, i] + dt * 0.5 * inv_c * inv_dy * inv_dy * (
                k_smoothed(0.5 * (T[j + 1, i] + T[j, i]), _delta)
                * (T[j + 1, i] - T[j, i])
                - k_smoothed(0.5 * (T[j, i] + T[j - 1, i]), _delta)
                * (T[j, i] - T[j - 1, i])
            )

            alpha[i] = -a_i / (b_i + c_i * alpha[i - 1])
            beta[i] = (rhs_i - c_i * beta[i - 1]) / (b_i + c_i * alpha[i - 1])

        tempT[j, n_x - 1] = beta[n_x - 2] / (
            1.0 - alpha[n_x - 2]
        )  # из правого граничного условия второго рода

        for i in range(n_x - 2, -1, -1):
            tempT[j, i] = alpha[i] * tempT[j, i + 1] + beta[i]

    tempT[0, :] = cfg.T_ICE_MIN
    tempT[n_y - 1, :] = cfg.T_WATER_MAX

    new_T = np.empty((n_y, n_x))

    alpha = np.empty(
        (n_y - 1),
    )
    beta = np.empty(
        (n_y - 1),
    )

    alpha[0] = 0  # из левого граничного условия первого рода
    beta[0] = cfg.T_ICE_MIN  # из левого граничного условия первого рода

    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            inv_c = 1.0 / c_smoothed(tempT[j, i], _delta)
            a_j = (
                -dt
                * 0.5
                * k_smoothed(0.5 * (tempT[j + 1, i] + tempT[j, i]), _delta)
                * inv_c
                * inv_dy
                * inv_dy
            )
            b_j = (
                1.0
                + dt
                * (
                    k_smoothed(0.5 * (tempT[j + 1, i] + tempT[j, i]), _delta)
                    + k_smoothed(0.5 * (tempT[j, i] + tempT[j - 1, i]), _delta)
                )
                * inv_c
                * inv_dy
                * inv_dy
                * 0.5
            )
            c_j = (
                -dt
                * 0.5
                * k_smoothed(0.5 * (tempT[j, i] + tempT[j - 1, i]), _delta)
                * inv_c
                * inv_dy
                * inv_dy
            )

            rhs_j = tempT[j, i] + dt * 0.5 * inv_c * inv_dx * inv_dx * (
                k_smoothed(0.5 * (tempT[j, i + 1] + tempT[j, i]), _delta)
                * (tempT[j, i + 1] - tempT[j, i])
                - k_smoothed(0.5 * (tempT[j, i] + tempT[j, i - 1]), _delta)
                * (tempT[j, i] - tempT[j, i - 1])
            )

            alpha[j] = -a_j / (b_j + c_j * alpha[j - 1])
            beta[j] = (rhs_j - c_j * beta[j - 1]) / (b_j + c_j * alpha[j - 1])

        new_T[n_y - 1, i] = beta[n_y - 2] / (
            1.0 - alpha[n_y - 2]
        )  # из правого граничного условия второго рода

        for j in range(n_y - 2, -1, -1):
            new_T[j, i] = alpha[j] * new_T[j + 1, i] + beta[j]

    new_T[:, 0] = new_T[:, 1]
    new_T[:, n_x - 1] = new_T[:, n_x - 2]

    return new_T
