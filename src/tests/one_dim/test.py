import numpy as np
import os
import shutil

from src.stefan_solver import solve
from src.parameters import N_Y, N_X, N_T, T_0, WIDTH, HEIGHT, FULL_TIME
from compare_boundary import compare_num_with_analytic
from src.tests.one_dim.analytic_solution_1d_2f import get_analytic_solution
from src.geometry import DomainGeometry
from src.temperature import get_max_delta

from matplotlib import pyplot as plt


def run_test():
    # dir_name = input("Enter a directory name where data will be stored: ")
    dir_name = "erf_delta_adaptive_2"
    dir_name = f"./results/{dir_name}"

    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    geometry = DomainGeometry(
        width=WIDTH,
        height=HEIGHT,
        end_time=FULL_TIME,
        n_x=N_X,
        n_y=N_Y,
        n_t=N_T
    )

    print(geometry)

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
    # T = init_temperature_test()
    # plot_temperature(T, 0, 0, True, True)
    boundary = [s_0]
    deltas = []
    times = [0.0]
    i = int(N_X / 2)

    deltas.append(get_max_delta(T))

    for n in range(1, N_T):
        t = n * geometry.dt
        T = solve(
            T=T,
            top_cond_type=1,
            right_cond_type=2,
            bottom_cond_type=1,
            left_cond_type=2,
            dx=geometry.dx,
            dy=geometry.dy,
            dt=geometry.dt,
            time=t,
            fixed_delta=fixed_delta
        )
        if n % 24 == 0:
            # plot_temperature(T, 0, 0, True, True)
            deltas.append(get_max_delta(T))
            times.append(t)
            print(f"ДЕНЬ: {int(n/24)}")
            for j in range(N_Y - 1):
                if (T[j, i] - T_0) * (T[j + 1, i] - T_0) < 0.0:
                    y_0 = abs((T[j, i] * (j + 1) * geometry.dy - T[j + 1, i] * j * geometry.dy) / (T[j, i] - T[j + 1, i]))
                    boundary.append(y_0)
                    break

    # plt.plot(
    #     times,
    #     deltas,
    #     linewidth=1,
    #     color='r',
    # )
    # plt.show()

    np.savez_compressed(f"{dir_name}/1d_2f_boundary", boundary=boundary)

    compare_num_with_analytic(
        num=boundary,
        _s_0=s_0,
        _delta=_delta,
        show_graphs=True,
        dir_name=dir_name
    )
    shutil.copy("../../parameters.py", dir_name)


run_test()
