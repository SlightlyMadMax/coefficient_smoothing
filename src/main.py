import time

import numpy as np

import src.parameters as cfg
from src.boundary_conditions import BoundaryCondition, BoundaryConditionType
from src.fluid_dynamics.plotting import plot_velocity_field
from src.fluid_dynamics.utils import calculate_velocity_field
from src.solvers.convection import NavierStokesSolver
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_vorticity,
)
from src.geometry import DomainGeometry
from src.temperature.init_values import init_temperature_shape, TemperatureShape
from src.temperature.coefficient_smoothing.delta import get_max_delta
from src.temperature.plotting import plot_temperature, animate
from src.temperature.utils import TemperatureUnit
from src.solvers.heat_transfer import LocOneDimSolver

if __name__ == "__main__":
    geometry = DomainGeometry(
        width=1.0,
        height=1.0,
        end_time=60.0 * 60.0 * 24.0 * 7.0,
        n_x=500,
        n_y=500,
        n_t=60 * 60 * 24 * 3,
    )

    print(geometry)

    u = init_temperature_shape(
        geom=geometry,
        shape=TemperatureShape.PACMAN,
        water_temp=cfg.T_WATER_MAX,
        ice_temp=cfg.T_ICE_MIN,
    )

    print(f"Delta for initial temperature distribution: {get_max_delta(u)}")

    plot_temperature(
        T=u,
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

    top_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: cfg.T_ICE_MIN * np.ones(geometry.n_x),
    )
    right_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_y,
        value_func=lambda t, n: cfg.T_ICE_MIN * np.ones(geometry.n_y),
    )
    bottom_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: cfg.T_ICE_MIN * np.ones(geometry.n_x),
    )
    left_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_y,
        value_func=lambda t, n: cfg.T_ICE_MIN * np.ones(geometry.n_y),
    )

    sf = initialize_stream_function(geom=geometry)
    w = initialize_vorticity(geom=geometry)

    u_full = [u]
    times = [0.0]
    heat_transfer_solver = LocOneDimSolver(
        geometry=geometry,
        top_bc=top_bc,
        right_bc=right_bc,
        bottom_bc=bottom_bc,
        left_bc=left_bc,
        fixed_delta=False,
    )
    # navier_solver = NavierStokesSolver(
    #     geometry=geometry,
    #     top_cond_type=cfg.DIRICHLET,
    #     right_cond_type=cfg.DIRICHLET,
    #     bottom_cond_type=cfg.DIRICHLET,
    #     left_cond_type=cfg.DIRICHLET,
    # )

    start_time = time.process_time()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt

        u = heat_transfer_solver.solve(u=u, sf=sf, time=t, iters=1)
        # sf, w = navier_solver.solve(w=w, sf=sf, u=u)

        if n % 60 == 0:
            plot_temperature(
                T=u,
                geom=geometry,
                time=t,
                graph_id=n,
                plot_boundary=True,
                show_graph=True,
                min_temp=cfg.T_WATER_MAX,
                max_temp=10.0,
                invert_yaxis=False,
                actual_temp_units=TemperatureUnit.CELSIUS,
                display_temp_units=TemperatureUnit.CELSIUS,
            )
            # v_x, v_y = calculate_velocity_field(sf=sf, dx=geometry.dx, dy=geometry.dy)
            # plot_velocity_field(
            #     v_x=v_x,
            #     v_y=v_y,
            #     geometry=geometry,
            #     graph_id=n,
            #     show_graph=True,
            # )
            # u_full.append(u.copy())
            # times.append(t)
            print(
                f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}"
            )
            print(f"Максимальная температура: {round(np.max(u), 2)}")
            print(f"Минимальная температура: {round(np.min(u), 2)}")
            # print(f"Максимальное значение функции тока: {round(np.max(sf), 6)}")
            # print(f"Минимальное значение функции тока: {round(np.min(sf), 6)}")
            # print(f"Максимальное значение вихря скорости: {round(np.max(w), 2)}")
            # print(f"Минимальное значение вихря скорости: {round(np.min(w), 2)}")

    print("СОЗДАНИЕ АНИМАЦИИ...")
    animate(
        T_full=u_full,
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
        T=u,
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
