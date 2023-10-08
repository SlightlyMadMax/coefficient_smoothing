import numpy as np
import os

from datetime import datetime

from src.solver import solve
from src.parameters import N_Y, N_X, N_T, T_0, dy, dt
from src.plotting import plot_temperature


def run_test():
    dir_name = f"./tests/num_vs_analytic/results/1d_2f_{datetime.now().strftime('%m%d%y_%H%M')}"

    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    # T = init_temperature_1d_1f_test()
    T = np.load("./tests/num_vs_analytic/data/analytic_1d_2f_at_0_3.npz")["T"]

    plot_temperature(T, time=0, plot_boundary=True, graph_id=0, show_graph=True)

    boundary = [0.3]
    temp_list = []

    for n in range(1, N_T):
        T = solve(T, fixed_delta=False)
        i = int(N_X/2)

        if n % 24 == 0:
            print(f"ДЕНЬ: {int(n/24)}")
            for j in range(N_Y - 1):
                if (T[j, i] - T_0) * (T[j + 1, i] - T_0) < 0.0:
                    if abs(T[j, i] - T_0) <= abs(T[j + 1, i] - T_0):
                        boundary.append(j * dy)
                    else:
                        boundary.append((j+1) * dy)
                    temp_list.append(T[:, i])
                    break

    print(len(boundary))
    np.savez_compressed(f"{dir_name}/1d_2f_boundary", boundary=boundary)
    # np.savez_compressed(f"./results/1d_2f_{dir_name}/1d_2f_temp_list", temp_list=temp_list)

    plot_temperature(T, time=dt * (N_T - 1), plot_boundary=True, graph_id=1, show_graph=True)
