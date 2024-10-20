import time

import numba
import numpy as np

from src.temperature import get_max_delta, init_temperature_circle, init_temperature_lake, init_temperature, init_temperature_test
from src.plotting import plot_temperature, animate
from src.solver import solve
from src.geometry import DomainGeometry
from src.boundary import init_boundary
import src.parameters as cfg


@numba.jit(nopython=True)
def get_water_thickness(T, dy: float):
    n_y, n_x = T.shape
    bottom, top = 0.0, 0.0
    for j in range(n_y-1):
        if T[j+1, n_x // 2] > cfg.T_0 and T[j, n_x // 2] <= cfg.T_0:
            bottom = j * dy
            continue
        if T[j+1, n_x // 2] <= cfg.T_0 and T[j, n_x // 2] > cfg.T_0:
            top = j * dy
            break
    return top - bottom


if __name__ == '__main__':
    geometry = DomainGeometry(
        width=20.0,
        height=20.0,
        end_time=60.0*60.0*24.0*365.0*3.0,
        n_x=200,
        n_y=200,
        n_t=8*365*3
    )

    print(geometry)

    # F = init_boundary(geometry)

    # T = init_temperature(geometry, F)

    T = init_temperature_test(geometry)

    from matplotlib import pyplot as plt

    plt.figure("Figure 1")
    plt.plot([i * geometry.dy for i in range(geometry.n_y)], T[:, geometry.n_x // 2] - cfg.T_0)
    plt.savefig("graphs/slice1.png")

    print(f"Delta for initial temperature distribution: {get_max_delta(T)}")

    # plot_temperature(
    #     T=T-cfg.T_0,
    #     geom=geometry,
    #     time=0.0,
    #     graph_id=0,
    #     plot_boundary=True,
    #     show_graph=True,
    #     min_temp=-32.0,
    #     max_temp=2.0,
    #     invert_yaxis=True,
    # )

    vertical_temp_slice = []
    horizontal_temp_slice = []

    wt = get_water_thickness(T, geometry.dy)
    print(wt)

    vertical_temp_slice.append(T[:, geometry.n_x // 2])
    horizontal_temp_slice.append(T[int((geometry.height - wt) / geometry.dy), :])

    start_time = time.process_time()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        T = solve(T,
                  top_cond_type=2,
                  right_cond_type=2,
                  bottom_cond_type=1,
                  left_cond_type=2,
                  dx=geometry.dx,
                  dy=geometry.dy,
                  dt=geometry.dt,
                  time=t,
                  fixed_delta=False
                  )

        # from boundary_conditions import solar_heat, air_temperature, get_top_bc_3
        # print(f"Solar heat: {solar_heat(t)}. Air temperature: {air_temperature(t) - cfg.T_0}, top bc = {get_top_bc_3(t)}, temperature of ice at the top: {T[-1, geometry.n_x//2] - cfg.T_0}")

        if n % 800 == 0:
            # plot_temperature(
            #     T=T-cfg.T_0,
            #     geom=geometry,
            #     time=t,
            #     graph_id=n,
            #     plot_boundary=True,
            #     show_graph=True,
            #     min_temp=-32.0,
            #     max_temp=2.0,
            #     invert_yaxis=True,
            # )
            print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")
            print(f"Максимальная температура: {np.max(T)}")
            print(f"Минимальная температура: {np.min(T)}")

            plt.plot([i * geometry.dy for i in range(geometry.n_y)], T[:, geometry.n_x // 2] - cfg.T_0)

            vertical_temp_slice.append(T[:, geometry.n_x // 2])
            horizontal_temp_slice.append(T[int((geometry.height - wt) / geometry.dy), :])

    plt.savefig("graphs/slice2.png")
    plt.show()

    np.savez_compressed(
        "data/lake_temp_slices.npz",
        vertical_temp_slice=vertical_temp_slice,
        horizontal_temp_slice=horizontal_temp_slice
    )

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
        T=T-cfg.T_0,
        geom=geometry,
        time=geometry.n_t * geometry.dt,
        graph_id=int(geometry.n_t / 60),
        plot_boundary=True,
        show_graph=False,
        min_temp=-32.0,
        max_temp=2.0
    )
