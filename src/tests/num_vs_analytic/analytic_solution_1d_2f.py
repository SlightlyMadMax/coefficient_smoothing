import math
import numpy as np

from scipy.optimize import fsolve
from scipy.special import erf

from src.parameters import K_ICE, K_WATER, C_ICE_VOL, C_WATER_VOL, L_VOL, N_X, N_Y, dy
from src.plotting import plot_temperature

g = -5.0
u_0 = 5.0

a_ice = (K_ICE / C_ICE_VOL) ** 0.5
a_water = (K_WATER / C_WATER_VOL) ** 0.5

s_0 = float(input("Enter the initial position of free boundary (in meters): "))


def trans_eq(_gamma: float):
    lhs = K_ICE * g * math.exp(-(_gamma / (2.0 * a_ice)) ** 2) / (a_ice * erf(_gamma / (2.0 * a_ice)))
    rhs = -K_WATER * u_0 * math.exp(-(_gamma / (2.0 * a_water)) ** 2) / \
          (a_water * (1.0 - erf(_gamma / (2.0 * a_water)))) - \
          _gamma * L_VOL * math.pi ** 0.5 / 2
    return lhs - rhs


gamma = fsolve(trans_eq, 0.0002)[0]

print(f"GAMMA: gamma")

t_0 = (s_0 / gamma)**2

print(f"STARTING TIME (S = {s_0} m): {int(t_0/3600)} hours")


def get_ice_temp(y: float, _s_0: float, _t_0: float):
    return g * (erf(_s_0 / (2.0 * a_ice * _t_0 ** 0.5)) - erf(y / (2.0 * a_ice * _t_0 ** 0.5))) / \
        erf(_s_0 / (2.0 * a_ice * _t_0 ** 0.5))


def get_water_temp(y: float, _s_0: float, _t_0: float):
    return u_0 * (erf(y / (2.0 * a_water * _t_0 ** 0.5)) - erf(_s_0 / (2.0 * a_water * _t_0 ** 0.5))) / \
        (1.0 - erf(_s_0 / (2.0 * a_water * _t_0 ** 0.5)))


analytic_T = np.empty((N_Y, N_X))

for j in range(N_Y):
    analytic_T[j, :] = get_ice_temp(j * dy, s_0, t_0) if j * dy <= s_0 else get_water_temp(j * dy, s_0, t_0)


np.savez_compressed("./data/analytic_1d_2f_at_0_3", T=analytic_T)

plot_temperature(analytic_T, t_0, 0, True)


def get_analytic_solution(_s_0: float):
    """
    Функция для нахождения аналитического решения модельной одномерной двухфазной задачи при заданных параметрах.
    :param _s_0: Начальное положение границы.
    :return: Распределение температуры для модельной задачи в момент времени, соответсвующий положению свободной
    границы s_0.
    """
    result = np.empty((N_Y, N_X))

    _gamma = fsolve(trans_eq, 0.0002)[0]
    _t_0 = (_s_0 / _gamma) ** 2

    for j in range(N_Y):
        result[j, :] = get_ice_temp(j * dy, _s_0, _t_0) if j * dy <= _s_0 else get_water_temp(j * dy, _s_0, _t_0)

    return result
