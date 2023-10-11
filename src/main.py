import time
import os
import numpy as np


from src.boundary import init_boundary
from src.temperature import init_temperature, air_temperature, get_delta_y, get_delta_x
from src.plotting import plot_temperature
from src.solver import solve
from src.parameters import N_T, dt


if __name__ == '__main__':
    dir_name = input("Enter a directory name where data will be stored: ")
    dir_name = f"../data/{dir_name}"

    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    F = init_boundary()
    T = init_temperature(F=F)

    plot_temperature(T, time=0, graph_id=0, plot_boundary=True, show_graph=True)

    start_time = time.process_time()

    for n in range(1, N_T):
        t = n * dt
        T = solve(T, time=t, fixed_delta=True)
        if n % 60 == 0:
            print(f"T_air_t = {round(air_temperature(t), 2)}")
            print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {int(n / 60)} ч, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")
            np.savez_compressed(f"{dir_name}/T_at_{n}", T=T)
            plot_temperature(T, time=t, graph_id=int(n / 60), plot_boundary=True, show_graph=True)

    plot_temperature(T, time=N_T * 3600, graph_id=int(N_T / 60), plot_boundary=False, show_graph=True)
