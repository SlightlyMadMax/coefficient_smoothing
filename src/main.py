import time
import os
import numpy as np
import matplotlib.pyplot as plt

from src.boundary import init_boundary, get_phase_trans_boundary, init_bc
from src.temperature import (init_temperature, get_max_delta, init_temperature_circle)
from src.plotting import plot_temperature, animate
from src.solver import solve
from src.parameters import N_T, dt, N_Y
from src.utils import get_crev_depth


if __name__ == '__main__':
    # dir_name = input("Enter a directory name where data will be stored: ")
    # dir_name = f"../data/{dir_name}"
    #
    # try:
    #     os.mkdir(dir_name)
    # except FileExistsError:
    #     pass

    # F = init_boundary()

    T = init_temperature_circle()
    print(get_max_delta(T))
    plot_temperature(T, time=0, graph_id=0, plot_boundary=True, show_graph=True)

    boundary_conditions = init_bc()

    start_time = time.process_time()

    depths = []
    deltas = []

    T_full = [T]
    times = [0]
    for n in range(1, N_T):
        t = n * dt
        T = solve(T, boundary_conditions, t, fixed_delta=False)
        if n % 60 == 0:
            T_full.append(T)
            times.append(t)
            print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")

    animate(T_full, times, t_step=60, filename="test_animation")
    plot_temperature(T, time=N_T * dt, graph_id=int(N_T / 60), plot_boundary=True, show_graph=True)
