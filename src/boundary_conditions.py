import numba
import numpy as np
from math import sin, cos, pi

from numba import types
from numpy import ndarray
from typing import Callable, Optional
from enum import Enum
from numba.typed import Dict
from src.parameters import T_ICE_MIN, T_WATER_MAX, K_WATER, CONV_COEF, Q_SOL, LAT, DECL, RAD_SPEED, T_air, T_amp


@numba.jit(nopython=True)
def get_left_bc_1(time: float, n_y: int) -> ndarray:
    return T_ICE_MIN * np.ones(n_y)


@numba.jit(nopython=True)
def get_top_bc_1(time: float, n_x: int) -> ndarray:
    return T_ICE_MIN * np.ones(n_x)


@numba.jit(nopython=True)
def get_right_bc_1(time: float, n_y: int) -> ndarray:
    return T_ICE_MIN * np.ones(n_y)


@numba.jit(nopython=True)
def get_bottom_bc_1(time: float, n_x: int) -> ndarray:
    return T_ICE_MIN * np.ones(n_x)


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
def get_top_bc_3(time: float) -> (float, float):
    # Определяем температуру воздуха у поверхности
    T_air_t = air_temperature(time)
    # Определяем тепловой поток солнечной энергии
    Q_sol = solar_heat(time)
    psi = Q_sol - CONV_COEF * T_air_t / K_WATER
    phi = CONV_COEF / K_WATER
    return psi, phi


class ConditionType(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3


class BoundaryCondition:
    def __init__(self, cond_type: ConditionType, init_value: ndarray,
                 eval_func: Optional[Callable[[float], ndarray]] = None, is_constant: bool = True):
        self._init_value = init_value
        self._cond_type = cond_type
        self._eval_func = eval_func
        self._is_constant = is_constant

    @property
    def cond_type(self):
        return self._cond_type

    @property
    def is_constant(self):
        return self._is_constant

    @property
    def init_value(self):
        return self._init_value

    def get_value_at_t(self, t: float):
        if not self._is_constant and self._eval_func is not None:
            return self._eval_func(t)
        else:
            return self._init_value

    def to_typed_dict_repr(self) -> Dict:
        if not self.is_constant:
            raise Exception("Cannot convert time-dependent condition to Dict.")
        bc = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:],
        )
        bc["type"] = np.asarray([float(self.cond_type.value)])
        bc["value"] = self.init_value
        return bc

    def __str__(self):
        return f"Condition type: {self.cond_type.name}. Is time-dependant: {not self.is_constant}."


class BoundaryConditions:
    def __init__(self, left: BoundaryCondition, right: BoundaryCondition,
                 bottom: BoundaryCondition, top: BoundaryCondition):
        self._left = left
        self._right = right
        self._bottom = bottom
        self._top = top

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def bottom(self):
        return self._bottom

    @property
    def top(self):
        return self._top

    def to_typed_dict_repr(self) -> Dict:
        bcs = Dict()
        bcs["top"] = self.top.to_typed_dict_repr()
        bcs["right"] = self.right.to_typed_dict_repr()
        bcs["bottom"] = self.bottom.to_typed_dict_repr()
        bcs["left"] = self.left.to_typed_dict_repr()
        return bcs

    def __str__(self):
        return (f"Top (y = HEIGHT). {self.top}\nRight (x = WIDTH). {self.right}"
                f"\nBottom (y = 0). {self.bottom}\nLeft (x = 0). {self.left}")


def init_bc() -> Dict:
    """
    Инициализация граничных условий.
    :return: словарь (совместимого с numba типа Dict), содержащий описание условий (тип, температура) для левой, правой, верхней и нижней границ области
    """
    boundary_conditions = Dict()

    bc_bottom = Dict()
    bc_bottom["type"] = 1.0
    bc_bottom["temp"] = T_ICE_MIN

    bc_top = Dict()
    bc_top["type"] = 1.0
    bc_top["temp"] = T_ICE_MIN

    bc_sides = Dict()
    bc_sides["type"] = 1.0
    bc_sides["temp"] = T_ICE_MIN

    boundary_conditions["left"] = bc_sides
    boundary_conditions["right"] = bc_sides
    boundary_conditions["bottom"] = bc_bottom
    boundary_conditions["top"] = bc_top

    return boundary_conditions
