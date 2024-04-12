import time

import numba
import numpy as np

from src.temperature import get_max_delta, init_temperature_circle, init_temperature_lake, init_temperature
from src.plotting import plot_temperature, animate
from src.solver import solve
from src.geometry import DomainGeometry
from src.boundary import init_boundary


@numba.jit(nopython=True)
def get_water_thickness(T, dy: float):
    n_y, n_x = T.shape
    bottom, top = 0.0, 0.0
    for j in range(n_y-1):
        if T[j+1, n_x // 2] > 0.0 and T[j, n_x // 2] <= 0.0:
            bottom = j * dy
            continue
        if T[j+1, n_x // 2] <= 0.0 and T[j, n_x // 2] > 0.0:
            top = j * dy
            break
    return top - bottom


if __name__ == '__main__':
    geometry = DomainGeometry(
        width=200.0,
        height=48.0,
        end_time=60.0*60.0*24.0*365.0*50.0,
        n_x=2001,
        n_y=500,
        n_t=8*365*50
    )

    print(geometry)

    # F = init_boundary(geometry)

    # T = init_temperature(geometry, F)

    T = init_temperature_lake(geometry, water_temp=2.0, ice_temp=-8.5)

    print(f"Delta for initial temperature distribution: {get_max_delta(T)}")

    plot_temperature(
        T=T,
        geom=geometry,
        time=0.0,
        graph_id=0,
        plot_boundary=True,
        show_graph=False,
        min_temp=-32.0,
        max_temp=2.0,
        invert_yaxis=True,
    )

    vertical_temp_slice = []
    horizontal_temp_slice = []

    wt = get_water_thickness(T, geometry.dy)
    print(wt)

    start_time = time.process_time()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        T = solve(T,
                  top_cond_type=3,
                  right_cond_type=2,
                  bottom_cond_type=1,
                  left_cond_type=2,
                  dx=geometry.dx,
                  dy=geometry.dy,
                  dt=geometry.dt,
                  time=t,
                  fixed_delta=False
                  )
        if n % 800 == 0:
            plot_temperature(
                T=T,
                geom=geometry,
                time=t,
                graph_id=n,
                plot_boundary=True,
                show_graph=False,
                min_temp=-32.0,
                max_temp=2.0,
                invert_yaxis=True,
            )
            print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")
            print(f"Максимальная температура: {max(T)}")
            vertical_temp_slice.append(T[:, geometry.n_x // 2])
            horizontal_temp_slice.append(T[int((geometry.height - wt) / geometry.dy), :])

    np.savez_compressed(
        "../data/lake_temp_slices.npz",
        vertical_temp_slice=vertical_temp_slice,
        horizontal_temp_slice=horizontal_temp_slice
    )

    # np.savez_compressed("../data/lake_test_data.npz", T_full=T_full, times=times)

    # print("СОЗДАНИЕ АНИМАЦИИ...")
    # animate(
    #     T_full=T_full,
    #     geom=geometry,
    #     times=times,
    #     t_step=1440,
    #     filename="lake_test_data",
    #     min_temp=-8.5,
    #     max_temp=2.0
    # )
    plot_temperature(
        T=T,
        geom=geometry,
        time=geometry.n_t * geometry.dt,
        graph_id=int(geometry.n_t / 60),
        plot_boundary=True,
        show_graph=True,
        min_temp=-32.0,
        max_temp=2.0
    )
