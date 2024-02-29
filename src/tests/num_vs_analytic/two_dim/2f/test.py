import time

import numba
import numpy as np
from numpy import ndarray

from src.geometry import DomainGeometry
from src.solver import solve
from src.plotting import plot_temperature, animate
from src.temperature import init_temperature_2f_test
from src.boundary import get_phase_trans_boundary


T_WATER = 5.0
T_ICE = -5.0
end_time = 60.0 * 60.0 * 24.0 * 300.0
n_t = 24*300 + 1


geom = DomainGeometry(
    width=1.0,
    height=1.0,
    end_time=end_time,
    n_x=500,
    n_y=500,
    n_t=n_t
)

T = init_temperature_2f_test(geom=geom, water_temp=T_WATER, ice_temp=T_ICE, b=0.5)

plot_temperature(
    T=T,
    geom=geom,
    time=0.0,
    graph_id=0,
    plot_boundary=True,
    show_graph=True,
    min_temp=T_ICE,
    max_temp=T_WATER,
    directory="./results/"
)

T_full = [T]
times = [0]

start_time = time.process_time()

for i in range(1, n_t+1):
    t = i * geom.dt
    T = solve(T,
              top_cond_type=1,
              right_cond_type=2,
              bottom_cond_type=1,
              left_cond_type=2,
              dx=geom.dx,
              dy=geom.dy,
              dt=geom.dt,
              time=t,
              fixed_delta=False
              )
    if i % 24 == 0:
        T_full.append(T)
        times.append(t)
        print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {i} ч, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")


print("СОЗДАНИЕ АНИМАЦИИ...")
animate(
    T_full=T_full,
    geom=geom,
    times=times,
    t_step=60*60*24,
    directory="./results/",
    filename="test_animation",
    min_temp=T_ICE,
    max_temp=T_WATER
)

X, Y = get_phase_trans_boundary(T, geom)
print(Y[250][250])
