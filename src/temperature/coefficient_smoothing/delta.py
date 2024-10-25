import numba

from numpy import ndarray

import src.parameters as cfg


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
            if (T[j + 1, i] - cfg.T_0) * (T[j, i] - cfg.T_0) < 0.0:
                temp = abs(T[j + 1, i] - T[j, i])
                delta = temp if temp > delta else delta
                break
            if (T[j, i + 1] - cfg.T_0) * (T[j, i] - cfg.T_0) < 0.0:
                temp = abs(T[j, i + 1] - T[j, i])
                delta = temp if temp > delta else delta
                break
    return delta