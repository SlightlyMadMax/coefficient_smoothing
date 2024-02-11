import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import fsolve
from scipy.special import erf

from src.parameters import K_ICE, K_WATER, C_ICE_VOL, C_WATER_VOL, L_VOL, dy

g = -5.0
u_0 = 5.0

a_ice = (K_ICE / C_ICE_VOL) ** 0.5
a_water = (K_WATER / C_WATER_VOL) ** 0.5


def trans_eq(_gamma: float):
    lhs = K_ICE * g * math.exp(-(_gamma / (2.0 * a_ice)) ** 2) / (a_ice * erf(_gamma / (2.0 * a_ice)))
    rhs = -K_WATER * u_0 * math.exp(-(_gamma / (2.0 * a_water)) ** 2) / \
          (a_water * (1.0 - erf(_gamma / (2.0 * a_water)))) - \
          _gamma * L_VOL * math.pi ** 0.5 / 2
    return lhs - rhs


_s_0 = 0.3

num_03 = np.load("results/delta_0_3_s_0_3/1d_2f_boundary.npz")["boundary"]
num_adap = np.load("results/delta_adaptive_s_0_3/1d_2f_boundary.npz")["boundary"]

gamma = fsolve(trans_eq, 0.0002)[0]

print(f"GAMMA: {gamma}")

n = len(num_03)
print(f"Modeling time: {n} days.")

t_0 = (_s_0 / gamma) ** 2

print(int(t_0/3600))

time = [i * 60. * 60. * 24.0 + t_0 for i in range(n)]

exact = [gamma * time[i] ** 0.5 for i in range(n)]

relative_error_1 = [abs(exact[i] - num_03[i]) * 100 / exact[i] for i in range(n)]
relative_error_2 = [abs(exact[i] - num_adap[i]) * 100 / exact[i] for i in range(n)]

fig = plt.figure()

ax = plt.axes()
plt.plot(
    time,
    relative_error_1,
    linewidth=1,
    color='r',
    label="$\Delta$ = 0.3"
)
plt.plot(
    time,
    relative_error_2,
    linewidth=1,
    color='k',
    label="Адаптивный подбор $\Delta$"
)
ax.set_title("Относительная погрешность")
ax.set_xlabel("Время, с")
ax.set_ylabel("Относительная погрешность, %")
ax.legend()
plt.savefig("./adap_vs_best_fixed.png")
plt.show()
