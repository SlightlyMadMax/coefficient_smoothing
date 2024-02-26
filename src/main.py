import time

from src.temperature import get_max_delta, init_temperature_circle
from src.plotting import plot_temperature, animate
from src.solver import solve
from src.parameters import WIDTH, HEIGHT, FULL_TIME, N_X, N_Y, N_T, T_ICE_MIN, T_WATER_MAX
from src.geometry import DomainGeometry


if __name__ == '__main__':
    geometry = DomainGeometry(
        width=WIDTH,
        height=HEIGHT,
        end_time=FULL_TIME,
        n_x=N_X,
        n_y=N_Y,
        n_t=N_T
    )

    print(geometry)

    # F = init_boundary()

    T = init_temperature_circle(geom=geometry, water_temp=T_WATER_MAX, ice_temp=T_ICE_MIN)

    print(f"Delta for initial temperature distribution: {get_max_delta(T)}")

    plot_temperature(
        T=T,
        geom=geometry,
        time=0.0,
        graph_id=0,
        plot_boundary=True,
        show_graph=True,
        min_temp=T_ICE_MIN,
        max_temp=T_WATER_MAX
    )

    start_time = time.process_time()

    T_full = [T]
    times = [0]
    for n in range(1, N_T):
        t = n * geometry.dt
        T = solve(T,
                  top_cond_type=1,
                  right_cond_type=1,
                  bottom_cond_type=1,
                  left_cond_type=1,
                  dx=geometry.dx,
                  dy=geometry.dy,
                  dt=geometry.dt,
                  time=t,
                  fixed_delta=False
                  )
        if n % 60 == 0:
            T_full.append(T)
            times.append(t)
            print(f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {n} М, ВРЕМЯ ВЫПОЛНЕНИЯ: {time.process_time() - start_time}")

    print("СОЗДАНИЕ АНИМАЦИИ...")
    animate(
        T_full=T_full,
        geom=geometry,
        times=times,
        t_step=60,
        filename="test_animation",
        min_temp=T_ICE_MIN,
        max_temp=T_WATER_MAX
    )
    plot_temperature(
        T=T,
        geom=geometry,
        time=geometry.n_t * geometry.dt,
        graph_id=int(geometry.n_t / 60),
        plot_boundary=True,
        show_graph=True,
        min_temp=T_ICE_MIN,
        max_temp=T_WATER_MAX
    )
