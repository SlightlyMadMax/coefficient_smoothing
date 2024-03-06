import os
import matplotlib.pyplot as plt

from typing import Optional
from matplotlib import animation
from numpy import ndarray

from src.boundary import get_phase_trans_boundary
from src.geometry import DomainGeometry


def plot_temperature(T: ndarray,
                     geom: DomainGeometry,
                     time: float,
                     graph_id: int,
                     plot_boundary: bool = False,
                     show_graph: bool = True,
                     show_grid: bool = False,
                     directory: str = "../graphs/temperature/",
                     min_temp: Optional[float] = None,
                     max_temp: Optional[float] = None,
                     equal_aspect: Optional[bool] = True
                     ):
    X, Y = geom.mesh_grid

    ax = plt.axes(xlim=(0, geom.width), ylim=(0, geom.height), xlabel="x, м", ylabel="y, м")

    if show_grid:
        plt.plot(X, Y, marker=".", markersize=0.5, color='k', linestyle='none')

    plt.contourf(X, Y, T, 50, cmap="viridis", vmin=min_temp, vmax=max_temp)
    plt.clim(min_temp, max_temp)
    plt.colorbar()

    if plot_boundary:
        X_b, Y_b = get_phase_trans_boundary(T=T, geom=geom)
        plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color='r', label='Граница ф.п.')
        ax.legend()

    ax.set_title(f"t = {int(time/60)} м.\n dx = {round(geom.dx, 3)} m, "
                 f"dy = {round(geom.dy, 3)} m, dt = {round(geom.dt)} с")

    if equal_aspect:
        ax.set_aspect("equal")

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}T_{graph_id}.png")

    if show_graph:
        plt.show()
    else:
        plt.close()


def animate(T_full: list | ndarray,
            geom: DomainGeometry,
            times: list | ndarray,
            t_step: int,
            filename: str,
            directory: str = "../graphs/animations/",
            min_temp: Optional[float] = None,
            max_temp: Optional[float] = None,
            equal_aspect: Optional[bool] = True
            ):
    plt.rcParams["animation.ffmpeg_path"] = r"C:\Users\ZZZ\ffmpeg\ffmpeg.exe"
    plt.rcParams["animation.convert_path"] = r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe"

    fig = plt.figure()
    ax = plt.axes(xlim=(0, geom.width), ylim=(0, geom.height), xlabel="x, м", ylabel="y, м")

    if equal_aspect:
        ax.set_aspect("equal")

    B = []
    for T in T_full:
        B.append(get_phase_trans_boundary(T, geom=geom))

    X, Y = geom.mesh_grid

    # define the first frame
    cont = plt.contourf(X, Y, T_full[0], 50, cmap="viridis", vmin=min_temp, vmax=max_temp)
    plt.title("t = 0 min")
    plt.colorbar()
    plt.clim(min_temp, max_temp)
    plt.scatter(B[0][0], B[0][1], s=1, linewidths=0.1, color='r', label='Граница ф.п.')
    plt.legend()

    def update(i):
        cont = plt.contourf(X, Y, T_full[i], 50, cmap="viridis")
        plt.scatter(B[i][0], B[i][1], s=1, linewidths=0.1, color='r', label='Граница ф.п.')
        plt.title(f"t = {round(i * t_step)} min")
        return cont

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=100, blit=False, repeat=True)

    if not os.path.exists(directory):
        os.makedirs(directory)

    anim.save(f"{directory}{filename}.gif", dpi=100, writer="imagemagick")
