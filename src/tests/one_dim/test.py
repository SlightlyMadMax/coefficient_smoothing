import numpy as np
import os
import shutil

from compare_boundary import compare_num_with_analytic
from src.geometry import DomainGeometry
from src.parameters import N_Y, N_X, N_T, T_0, WIDTH, HEIGHT, FULL_TIME
from src.plotting import plot_temperature
from src.solvers.heat_transfer import HeatTransferSolver
from src.tests.one_dim.analytic_solution_1d_2f import get_analytic_solution


if __name__ == "__main__":
    # dir_name = input("Enter a directory name where data will be stored: ")
    dir_name = "erf_delta_adaptive_3"
    dir_name = f"./results/{dir_name}"

    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    geometry = DomainGeometry(
        width=WIDTH, height=HEIGHT, end_time=FULL_TIME, n_x=N_X, n_y=N_Y, n_t=N_T
    )

    print(geometry)

    heat_transfer_solver = HeatTransferSolver(
        geometry=geometry,
        top_cond_type=1,
        right_cond_type=2,
        bottom_cond_type=1,
        left_cond_type=2,
        fixed_delta=False,
    )

    # s_0 = float(input("Enter the initial position of free boundary (in meters): "))

    s_0 = 0.3

    # _delta = input("Enter the smoothing parameter delta or just press 'Enter' to use an adaptive one: ")

    _delta = ""

    if _delta == "":
        _delta = None
        fixed_delta = False
    else:
        _delta = float(_delta)
        fixed_delta = True

    T = get_analytic_solution(_s_0=s_0)

    boundary = [s_0]
    times = [0.0]
    i = int(N_X / 2)


    for n in range(1, N_T):
        t = n * geometry.dt
        T = heat_transfer_solver.solve(u=T, time=t, iters=4)
        if n % 24 == 0:
            times.append(t)
            print(f"ДЕНЬ: {int(n / 24)}")
            # plot_temperature(T, geometry, time=t, graph_id=123, plot_boundary=True, show_graph=True, directory=".")
            for j in range(N_Y - 1):
                if (T[j, i] - T_0) * (T[j + 1, i] - T_0) < 0.0:
                    y_0 = (
                        j * geometry.dy
                        + ((T_0 - T[j, i]) / (T[j + 1, i] - T[j, i])) * geometry.dy
                    )
                    print(y_0)
                    boundary.append(y_0)
                    break

    # np.savez_compressed(f"{dir_name}/1d_2f_boundary", boundary=boundary)

    compare_num_with_analytic(
        num=boundary, _s_0=s_0, _delta=_delta, show_graphs=True, dir_name=dir_name
    )
    # shutil.copy("../../parameters.py", dir_name)

