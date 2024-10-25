import time

import numpy as np

from src.temperature.coefficient_smoothing.delta import get_max_delta
from src.plotting import plot_temperature, animate
from src.temperature.solver import solve
from src.geometry import DomainGeometry
import src.parameters as cfg
from src.temperature.init_values import init_temperature_shape, TemperatureShape

if __name__ == "__main__":
    geometry = DomainGeometry(
        width=1.0,
        height=1.0,
        end_time=60.0 * 60.0 * 24.0 * 7.0,
        n_x=500,
        n_y=500,
        n_t=60 * 24 * 3,
    )

    print(geometry)

    # F = init_boundary(geometry)

    # T = init_temperature(geometry, F)

    ice_temp = 264.15
    water_temp = 274.15

    T = init_temperature_shape(
        geom=geometry,
        shape=TemperatureShape.SQUARE,
        water_temp=water_temp,
        ice_temp=ice_temp,
    )

    print(f"Delta for initial temperature distribution: {get_max_delta(T)}")

    plot_temperature(
        T=T,
        geom=geometry,
        time=0.0,
        graph_id=0,
        plot_boundary=True,
        show_graph=True,
        min_temp=ice_temp - cfg.T_0,
        max_temp=water_temp - cfg.T_0,
        invert_yaxis=False,
    )

    temp_T = np.empty((geometry.n_y, geometry.n_x))
    new_T = np.empty((geometry.n_y, geometry.n_x))
    alpha = np.empty(geometry.n_x - 1)
    beta = np.empty(geometry.n_x - 1)

    T_full = [T]
    times = [0.0]
    start_time = time.process_time()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        T = solve(
            T=T,
            temp_T=temp_T,
            new_T=new_T,
            alpha=alpha,
            beta=beta,
            top_cond_type=cfg.DIRICHLET,
            right_cond_type=cfg.DIRICHLET,
            bottom_cond_type=cfg.DIRICHLET,
            left_cond_type=cfg.DIRICHLET,
            dx=geometry.dx,
            dy=geometry.dy,
            dt=geometry.dt,
            time=t,
            fixed_delta=False,
        )

        if n % 60 == 0:
            plot_temperature(
                T=T,
                geom=geometry,
                time=t,
                graph_id=n,
                plot_boundary=True,
                show_graph=False,
                min_temp=ice_temp - cfg.T_0,
                max_temp=water_temp - cfg.T_0,
                invert_yaxis=False,
            )
            T_full.append(T.copy())
            times.append(t)
            print(
                f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}"
            )
            print(f"Максимальная температура: {round(np.max(T) - cfg.T_0, 2)}")
            print(f"Минимальная температура: {round(np.min(T) - cfg.T_0, 2)}")

    print("СОЗДАНИЕ АНИМАЦИИ...")
    animate(
        T_full=T_full,
        geom=geometry,
        times=times,
        t_step=3600,
        filename="new_test_animation",
        min_temp=ice_temp - cfg.T_0,
        max_temp=water_temp - cfg.T_0,
    )
    plot_temperature(
        T=T,
        geom=geometry,
        time=geometry.n_t * geometry.dt,
        graph_id=int(geometry.n_t / 60),
        plot_boundary=True,
        show_graph=False,
        min_temp=ice_temp - cfg.T_0,
        max_temp=water_temp - cfg.T_0,
    )
