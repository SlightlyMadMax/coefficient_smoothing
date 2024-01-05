import numba

from src.parameters import T_0, dy, HEIGHT, WATER_H


@numba.jit(nopython=True)
def is_frozen(T) -> bool:
    """
    Определяет, произошло ли замерзание всей воды (отсутствие границы фазового перехода).
    :param T: двумерный массив температур
    :return: True -- если граница ф.п. отсутствует, иначе False
    """
    N_X, N_Y = T.shape
    for i in range(N_X - 1):
        for j in range(N_Y - 1):
            if (T[j + 1, i] - T_0) * (T[j, i] - T_0) < 0.0 or (T[j, i + 1] - T_0) * (T[j, i] - T_0) < 0.0:
                return False
    return True


@numba.jit(nopython=True)
def get_crev_depth(T) -> float:
    """
    Определяет глубину трещины.
    :param T: двумерный массив температур
    :return: максимальная глубина трещины
    """
    N_X, N_Y = T.shape
    i = int(N_X/2)
    for j in range(N_Y - 1):
        if (T[j + 1, i] - T_0) * (T[j, i] - T_0) < 0.0 or (T[j, i + 1] - T_0) * (T[j, i] - T_0) < 0.0:
            return HEIGHT - WATER_H - j*dy
