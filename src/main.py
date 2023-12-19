import time
import os
import numpy as np
import numba
import matplotlib.pyplot as plt

from src.boundary import init_boundary, get_phase_trans_boundary, init_bc
from src.temperature import (init_temperature, air_temperature, init_temperature_circle, init_temperature_square, init_temperature_pacman)
from src.plotting import plot_temperature
from src.solver import solve
from src.parameters import N_T, dt, T_0, T_ICE_MIN, dy, N_Y, HEIGHT, WATER_H


@numba.jit(nopython=True)
def is_frozen(T) -> bool:
    N_X, N_Y = T.shape
    for i in range(N_X - 1):
        for j in range(N_Y - 1):
            if (T[j + 1, i] - T_0) * (T[j, i] - T_0) < 0.0 or (T[j, i + 1] - T_0) * (T[j, i] - T_0) < 0.0:
                return False
    return True


@numba.jit(nopython=True)
def get_crev_depth(T) -> float:
    N_X, N_Y = T.shape
    i = int(N_X/2)
    for j in range(N_Y - 1):
        if (T[j + 1, i] - T_0) * (T[j, i] - T_0) < 0.0 or (T[j, i + 1] - T_0) * (T[j, i] - T_0) < 0.0:
            return HEIGHT - WATER_H - j*dy


if __name__ == '__main__':
    # dir_name = input("Enter a directory name where data will be stored: ")
    # dir_name = f"../data/{dir_name}"
    #
    # try:
    #     os.mkdir(dir_name)
    # except FileExistsError:
    #     pass

    F = init_boundary()
    T = init_temperature(F=F)

    plot_temperature(T, time=0, graph_id=0, plot_boundary=True, show_graph=True)

    boundary_conditions = init_bc()

    start_time = time.process_time()

    depths = []

    for n in range(1, N_T):
        t = n * dt
        T = solve(T, boundary_conditions, t, fixed_delta=False)
        if n % 30 == 0:
            # print(f"T_air_t = {round(air_temperature(t), 2)}")
            # print(f"Max delta: {np.amax(get_delta_matrix(T))}")
            # print(f"Max temp: {np.amax(T)}")
            # print(f"Min temp: {np.amin(T)}")
            print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")
            # np.savez_compressed(f"{dir_name}/T_at_{n}", T=T)
            d = get_crev_depth(T)
            print(d)
            depths.append(d)
            plot_temperature(T, time=t, graph_id=n, plot_boundary=True, show_graph=False)
        # if is_frozen(T):
        #     print(f"УСЕ! {n}")
        #     break

    print(depths)
    times = [i*0.5 for i in range(1, len(depths)+1)]
    plt.plot(times, depths)
    plt.savefig(f"../graphs/delta/depth_{N_Y}.png")
    plot_temperature(T, time=N_T * dt, graph_id=int(N_T / 60), plot_boundary=True, show_graph=False)
