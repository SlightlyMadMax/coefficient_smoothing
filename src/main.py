import time
import os
import numpy as np

from datetime import datetime

from src.boundary import init_boundary
from src.temperature import init_temperature
from src.plotting import plot_temperature
from src.solver import solve
from src.parameters import N_T
from src.tests.num_vs_analytic.test import run_test

if __name__ == '__main__':
    # run_test()
    dir_name = f"../data/{datetime.now().strftime('%m%d%y_%H%M')}"

    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    F = init_boundary()
    T = init_temperature(F=F)

    plot_temperature(T, time=0, graph_id=0, plot_boundary=True, show_graph=True)

    start_time = time.process_time()

    for n in range(1, N_T):
        T = solve(T, fixed_delta=False)
        if n % 60 == 0:
            print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {int(n / 60)} ч, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")
            np.savez_compressed(f"../data/{dir_name}/T_at_{n}", T=T)
            plot_temperature(T, time=n * 60, graph_id=int(n / 60), plot_boundary=True, show_graph=True)

    plot_temperature(T, time=N_T * 3600, graph_id=int(N_T / 60), plot_boundary=False, show_graph=True)
