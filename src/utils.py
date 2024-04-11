import numba

import src.parameters as cfg


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
            if (T[j + 1, i] - cfg.T_0) * (T[j, i] - cfg.T_0) < 0.0 or (T[j, i + 1] - cfg.T_0) * (T[j, i] - cfg.T_0) < 0.0:
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
        if (T[j + 1, i] - cfg.T_0) * (T[j, i] - cfg.T_0) < 0.0 or (T[j, i + 1] - cfg.T_0) * (T[j, i] - cfg.T_0) < 0.0:
            return cfg.HEIGHT - cfg.WATER_H - j*cfg.dy
