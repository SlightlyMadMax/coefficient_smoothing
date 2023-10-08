import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.special import erf

from src.parameters import K_ICE, K_WATER, C_ICE_VOL, C_WATER_VOL, L_VOL

s_0 = 0.3
g = -5.0
u_0 = 5.0

a_ice = (K_ICE / C_ICE_VOL) ** 0.5
a_water = (K_WATER / C_WATER_VOL) ** 0.5

dir_name = input("DIR NAME: ")

method = input("FINITE DIFFERENCE METHOD: ")

delta = input("DELTA: ")

def trans_eq(_gamma: float):
    lhs = K_ICE * g * math.exp(-(_gamma / (2.0 * a_ice)) ** 2) / (a_ice * erf(_gamma / (2.0 * a_ice)))
    rhs = -K_WATER * u_0 * math.exp(-(_gamma / (2.0 * a_water)) ** 2) / \
          (a_water * (1.0 - erf(_gamma / (2.0 * a_water)))) - \
          _gamma * L_VOL * math.pi ** 0.5 / 2
    return lhs - rhs


gamma = fsolve(trans_eq, 0.0002)[0]

print(f"GAMMA: {gamma}")

num = np.load(f"./results/{dir_name}/1d_2f_boundary.npz")['boundary']

n = len(num)
print(f"Modeling time: {n} days.")

t_0 = (s_0 / gamma) ** 2

time = [i * 60. * 60. * 24.0 + t_0 for i in range(n)]

exact = [gamma * time[i] ** 0.5 for i in range(n)]

relative_error = [abs(exact[i] - num[i]) * 100 / exact[i] for i in range(n)]

abs_error = [abs(exact[i] - num[i]) for i in range(n)]

fig = plt.figure()
ax = plt.axes()

plt.plot(time, relative_error, linewidth=1, color='r', label=f'Relative error, delta={delta}\n{method}')
ax.set_xlabel("time, s")
ax.set_ylabel("relative error, %")
ax.legend()
plt.savefig(f"./results/{dir_name}/1d_2f_boundary_rel_error.png")
plt.show()
plt.clf()

ax = plt.axes()
plt.plot(time, abs_error, linewidth=1, color='r', label=f'Abs error, delta={delta}')
ax.set_xlabel("time, s")
ax.set_ylabel("abs error, m")
ax.legend()
plt.savefig(f"./results/{dir_name}/1d_2f_boundary_abs_error.png")
plt.show()
plt.clf()

ax = plt.axes()
plt.plot(time, exact, linewidth=1, color='r', label='Analytic solution')
plt.plot(time, num, linewidth=1, color='k', label='Numerical solution')
ax.set_xlabel("time, s")
ax.set_ylabel("free boundary position, m")
ax.legend()
plt.savefig(f"./results/{dir_name}/1d_2f_boundary.png")
plt.show()
