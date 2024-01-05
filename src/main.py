import time
import os
import numpy as np
import matplotlib.pyplot as plt

from src.boundary import init_boundary, get_phase_trans_boundary, init_bc
from src.temperature import (init_temperature, get_max_delta, air_temperature, solar_heat)
from src.plotting import plot_temperature
from src.solver import solve
from src.parameters import N_T, dt, N_Y
from src.utils import get_crev_depth, is_frozen


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
    print(get_max_delta(T))
    plot_temperature(T, time=0, graph_id=0, plot_boundary=True, show_graph=True)

    boundary_conditions = init_bc()

    start_time = time.process_time()

    depths = []
    deltas = []
    for n in range(1, N_T):
        t = n * dt
        T = solve(T, boundary_conditions, t, fixed_delta=False)
        if n % 2 == 0:
            depth = get_crev_depth(T)
            delta = get_max_delta(T)
            depths.append(depth)
            deltas.append(delta)
        if n % 60 == 0:
            print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")
            # b = get_phase_trans_boundary(T=T)
            # np.savez_compressed(f"../data/max_delta_testing/b_{n}", b=b)
            # print(f"T_air_t = {round(air_temperature(t), 2)}")
            # print(f"Max temp: {np.amax(T)}")
            # print(f"Min temp: {np.amin(T)}")
            # print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")
            # np.savez_compressed(f"{dir_name}/T_at_{n}", T=T)
            # plot_temperature(T, time=t, graph_id=n, plot_boundary=False, show_graph=True)
        # if is_frozen(T):
        #     print(f"УСЕ! {n}")
        #     break
    np.savez_compressed("../data/max_delta_testing/adaptive_delta_1500_72h_1m_sun", deltas=deltas)
    np.savez_compressed("../data/max_delta_testing/depths_1500_72h_1m_sun", depths=depths)
    times = [i/30 for i in range(1, len(deltas)+1)]
    ax = plt.axes()
    plt.plot(times, deltas, linewidth=1, color='b', label="1500x1500")
    plt.title("Зависимость параметра $\Delta$ от времени\nпри адаптивном подборе")
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("$\Delta$, м")
    ax.legend(title="Число узлов сетки")
    plt.savefig(f"../graphs/delta/comparison_fixed_indep_3.png")

    plot_temperature(T, time=N_T * dt, graph_id=int(N_T / 60), plot_boundary=True, show_graph=False)
