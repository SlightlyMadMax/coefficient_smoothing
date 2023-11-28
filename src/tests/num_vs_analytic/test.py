import numpy as np
import os
import shutil

from numba.typed import Dict

from src.solver import solve
from src.parameters import N_Y, N_X, N_T, T_0, dy
from compare_boundary import compare_num_with_analytic
from src.tests.num_vs_analytic.plotting import plot_temperature
from src.tests.num_vs_analytic.analytic_solution_1d_2f import get_analytic_solution
from src.temperature import init_temperature_test


def run_test():
    # dir_name = input("Enter a directory name where data will be stored: ")
    dir_name = "hehe"
    dir_name = f"./results/{dir_name}"

    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

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

    i = int(N_X / 2)

    boundary_conditions = Dict()

    boundary_conditions["left"] = 2
    boundary_conditions["right"] = 2
    boundary_conditions["bottom"] = 1
    boundary_conditions["upper"] = 1

    for n in range(1, N_T):
        T = solve(T, boundary_conditions=boundary_conditions, _delta=_delta, fixed_delta=fixed_delta)
        if n % 24 == 0:
            # plot_temperature(T, 0, 0, True, True)
            print(f"ДЕНЬ: {int(n/24)}")
            for j in range(N_Y - 1):
                if (T[j, i] - T_0) * (T[j + 1, i] - T_0) < 0.0:
                    if abs(T[j, i] - T_0) <= abs(T[j + 1, i] - T_0):
                        boundary.append(j * dy)
                    else:
                        boundary.append((j+1) * dy)
                    break

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
