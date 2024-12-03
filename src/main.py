import time

import numpy as np

import src.parameters as cfg
from src.solvers.convection import NavierStokesSolver
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_vorticity,
)
from src.geometry import DomainGeometry
from src.temperature.init_values import init_temperature_shape, TemperatureShape
from src.temperature.coefficient_smoothing.delta import get_max_delta
from src.solvers.heat_transfer import LocOneDimSolver
from src.plotting import plot_temperature, animate
from src.temperature.utils import TemperatureUnit

if __name__ == "__main__":
    geometry = DomainGeometry(
        width=1.0,
        height=1.0,
        end_time=60.0 * 60.0 * 24.0,
        n_x=500,
        n_y=500,
        n_t=60 * 60 * 24,
    )

    print(geometry)

    # F = init_boundary(geometry)

    # T = init_temperature(geometry, F)

    T = init_temperature_shape(
        geom=geometry,
        shape=TemperatureShape.UNIFORM_W,
        water_temp=cfg.T_WATER_MAX,
        ice_temp=cfg.T_ICE_MIN,
    )

    print(f"Delta for initial temperature distribution: {get_max_delta(T)}")
    plot_temperature(
        T=T,
        geom=geometry,
        time=0.0,
        graph_id=0,
        plot_boundary=False,
        show_graph=True,
        min_temp=cfg.T_WATER_MAX,
        max_temp=10.0,
        invert_yaxis=False,
        actual_temp_units=TemperatureUnit.CELSIUS,
        display_temp_units=TemperatureUnit.CELSIUS,
    )

    sf = initialize_stream_function(geom=geometry)
    w = initialize_vorticity(geom=geometry)

    T_full = [T]
    times = [0.0]
    heat_transfer_solver = LocOneDimSolver(
        geometry=geometry,
        top_cond_type=cfg.DIRICHLET,
        right_cond_type=cfg.DIRICHLET,
        bottom_cond_type=cfg.DIRICHLET,
        left_cond_type=cfg.DIRICHLET,
        fixed_delta=False,
    )
    navier_solver = NavierStokesSolver(
        geometry=geometry,
        top_cond_type=cfg.DIRICHLET,
        right_cond_type=cfg.DIRICHLET,
        bottom_cond_type=cfg.DIRICHLET,
        left_cond_type=cfg.DIRICHLET,
    )

    start_time = time.process_time()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        T = heat_transfer_solver.solve(u=T, sf=sf, time=t, iters=1)
        sf, w = navier_solver.solve(w=w, sf=sf, u=T)

        if n % 60 == 0:
            plot_temperature(
                T=T,
                geom=geometry,
                time=t,
                graph_id=n,
                plot_boundary=True,
                show_graph=False,
                min_temp=cfg.T_WATER_MAX,
                max_temp=10.0,
                invert_yaxis=False,
                actual_temp_units=TemperatureUnit.CELSIUS,
                display_temp_units=TemperatureUnit.CELSIUS,
            )
            # T_full.append(T.copy())
            # times.append(t)
            print(
                f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}"
            )
            print(f"Максимальная температура: {round(np.max(T), 2)}")
            print(f"Минимальная температура: {round(np.min(T), 2)}")

    print("СОЗДАНИЕ АНИМАЦИИ...")
    animate(
        T_full=T_full,
        geom=geometry,
        times=times,
        t_step=3600,
        filename="new_test_animation_2",
        min_temp=cfg.T_WATER_MAX,
        max_temp=10.0,
        actual_temp_units=TemperatureUnit.CELSIUS,
        display_temp_units=TemperatureUnit.CELSIUS,
    )
    plot_temperature(
        T=T,
        geom=geometry,
        time=geometry.n_t * geometry.dt,
        graph_id=int(geometry.n_t / 60),
        plot_boundary=True,
        show_graph=False,
        min_temp=cfg.T_WATER_MAX,
        max_temp=10.0,
        actual_temp_units=TemperatureUnit.CELSIUS,
        display_temp_units=TemperatureUnit.CELSIUS,
    )
