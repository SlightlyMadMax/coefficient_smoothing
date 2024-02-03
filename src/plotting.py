import matplotlib.pyplot as plt
import numpy as np

from src.parameters import N_Y, N_X, HEIGHT, WIDTH, dt, dx, dy, T_WATER_MAX, T_ICE_MIN, N_T
from src.boundary import get_phase_trans_boundary
from matplotlib import animation


def plot_temperature(T, time: float, graph_id: int, plot_boundary: bool = False, show_graph: bool = True):
    x = np.linspace(0, WIDTH, N_X)
    y = np.linspace(0, HEIGHT, N_Y)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, WIDTH), ylim=(0, HEIGHT), xlabel="x, м", ylabel="y, м")

    # plt.plot(X, Y, marker=".", markersize=0.5, color='k', linestyle='none')  # сетка

    plt.contourf(X, Y, T, 50, cmap="viridis", vmin=T_ICE_MIN, vmax=T_WATER_MAX)
    plt.colorbar()
    plt.clim(T_ICE_MIN, T_WATER_MAX)

    if plot_boundary:
        X_b, Y_b = get_phase_trans_boundary(T)
        plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color='r', label='Граница ф.п.')
        ax.legend()

    ax.set_title(f"t = {int(time/60)} м.\n dx = {round(dx, 3)} m, dy = {round(dy, 3)} m, dt = {round(dt, 2)} с")
    plt.savefig(f"graphs/temperature/T_{graph_id}.png")

    if show_graph:
        plt.show()
    else:
        plt.close()


def animate(T_full, times, t_step, filename):
    x = np.linspace(0, WIDTH, N_X)
    y = np.linspace(0, HEIGHT, N_Y)
    X, Y = np.meshgrid(x, y)

    plt.rcParams["animation.ffmpeg_path"] = r"C:\Users\ZZZ\ffmpeg\ffmpeg.exe"
    plt.rcParams["animation.convert_path"] = r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe"

    fig = plt.figure()
    ax = plt.axes(xlim=(0, WIDTH), ylim=(0, HEIGHT), xlabel="x, м", ylabel="y, м")
    B = []

    for T in T_full:
        B.append(get_phase_trans_boundary(T))

    # define the first frame
    cont = plt.contourf(X, Y, T_full[0], 50, cmap="viridis", vmin=T_ICE_MIN, vmax=T_WATER_MAX)
    plt.title("t = 0 min")
    plt.colorbar()
    plt.clim(T_ICE_MIN, T_WATER_MAX)
    plt.scatter(B[0][0], B[0][1], s=1, linewidths=0.1, color='r', label='Граница ф.п.')
    plt.legend()

    def update(i):
        cont = plt.contourf(X, Y, T_full[i], 50, cmap="viridis", vmin=T_ICE_MIN, vmax=T_WATER_MAX)
        plt.scatter(B[i][0], B[i][1], s=1, linewidths=0.1, color='r', label='Граница ф.п.')
        plt.title(f"t = {round(i * t_step)} min")
        return cont

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=100, blit=False, repeat=True)
    anim.save(f"graphs/animations/{filename}.gif", dpi=100, writer="imagemagick")
