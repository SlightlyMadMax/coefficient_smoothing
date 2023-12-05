import matplotlib.pyplot as plt
import numpy as np

from src.parameters import N_Y, N_X, HEIGHT, WIDTH, dt, dx, dy, T_WATER_MAX, T_ICE_MIN
from src.boundary import get_phase_trans_boundary


def plot_temperature(T, time: float, graph_id: int, plot_boundary: bool = False, show_graph: bool = True):
    x = np.linspace(0, WIDTH, N_X)
    y = np.linspace(0, HEIGHT, N_Y)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes()

    # plt.plot(X, Y, marker=".", markersize=0.5, color='k', linestyle='none')  # сетка

    plt.contourf(X, Y, T, 50, cmap="viridis", vmin=T_ICE_MIN, vmax=T_WATER_MAX)
    plt.colorbar()
    plt.clim(T_ICE_MIN, T_WATER_MAX)

    if plot_boundary:
        X_b, Y_b = get_phase_trans_boundary(T)
        plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color='r', label='Граница ф.п.')
        ax.legend()

    # plt.ylim((4.0, 5.0))
    # plt.xlim((4.0, 5.0))

    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(f"t = {int(time/3600)} ч.\n dx = {round(dx, 3)} m, dy = {round(dy, 3)} m, dt = {round(dt, 2)} с")
    plt.savefig(f"../graphs/temperature/T_{graph_id}.png")

    if show_graph:
        plt.show()
    else:
        plt.close()
