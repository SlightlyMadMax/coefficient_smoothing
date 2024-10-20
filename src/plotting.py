import os
import matplotlib.pyplot as plt

from typing import Optional
from matplotlib import animation
from numpy import ndarray

from src.boundary import get_phase_trans_boundary
from src.geometry import DomainGeometry
from src.utils import TemperatureUnit
import parameters as cfg


def _get_temp_in_display_units(
    T: ndarray,
    actual_temp_units: TemperatureUnit = TemperatureUnit.KELVIN,
    display_temp_units: TemperatureUnit = TemperatureUnit.CELSIUS,
):
    """
    Перевод температуры в желаемые единицы измерения
    :param T: массив со значениями температур
    :param actual_temp_units: исходные единица измерения
    :param display_temp_units: желаемая единица измерения
    :return: массив температур в желаемых единицах измерения
    """
    if actual_temp_units == display_temp_units:
        return T
    return T - cfg.T_0 if display_temp_units == TemperatureUnit.CELSIUS else T + cfg.T_0


def plot_temperature(
    T: ndarray,
    geom: DomainGeometry,
    time: float,
    graph_id: int,
    actual_temp_units: TemperatureUnit = TemperatureUnit.KELVIN,
    display_temp_units: TemperatureUnit = TemperatureUnit.CELSIUS,
    plot_boundary: bool = False,
    show_graph: bool = True,
    show_grid: bool = False,
    directory: str = "../graphs/temperature/",
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    equal_aspect: Optional[bool] = True,
    invert_xaxis: Optional[bool] = False,
    invert_yaxis: Optional[bool] = False,
):
    X, Y = geom.mesh_grid

    if not show_graph:
        import matplotlib

        matplotlib.use("Agg")

    plt.rcParams["figure.figsize"] = (16, 9)

    ax = plt.axes(
        xlim=(0, geom.width), ylim=(0, geom.height), xlabel="x, м", ylabel="y, м"
    )

    if show_grid:
        plt.plot(X, Y, marker=".", markersize=0.5, color="k", linestyle="none")

    plt.contourf(
        X,
        Y,
        _get_temp_in_display_units(T, actual_temp_units, display_temp_units),
        50,
        cmap="viridis",
        vmin=min_temp,
        vmax=max_temp,
    )
    plt.clim(min_temp, max_temp)
    plt.colorbar()

    if plot_boundary:
        X_b, Y_b = get_phase_trans_boundary(
            T=T if actual_temp_units == TemperatureUnit.KELVIN else T + cfg.T_0,
            geom=geom,
        )
        plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color="r", label="Граница ф.п.")
        ax.legend()

    ax.set_title(
        f"t = {int(time/(24*60))} ч.\n dx = {round(geom.dx, 3)} м, "
        f"dy = {round(geom.dy, 3)} м, dt = {round(geom.dt/3600)} ч"
    )

    if invert_xaxis:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(reversed(labels))

    if invert_yaxis:
        labels = [item.get_text() for item in ax.get_yticklabels()]
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(reversed(labels))

    if equal_aspect:
        ax.set_aspect("equal")

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}T_{graph_id}.png")

    if show_graph:
        plt.show()
    else:
        plt.close()


def animate(
    T_full,
    geom: DomainGeometry,
    times,
    t_step: int,
    filename: str,
    actual_temp_units: TemperatureUnit = TemperatureUnit.KELVIN,
    display_temp_units: TemperatureUnit = TemperatureUnit.CELSIUS,
    directory: str = "../graphs/animations/",
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    equal_aspect: Optional[bool] = True,
):
    # plt.rcParams["animation.ffmpeg_path"] = r"C:\Users\ZZZ\ffmpeg\ffmpeg.exe"
    plt.rcParams["animation.convert_path"] = (
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
    )

    fig = plt.figure()
    ax = plt.axes(
        xlim=(0, geom.width), ylim=(0, geom.height), xlabel="x, м", ylabel="y, м"
    )

    if equal_aspect:
        ax.set_aspect("equal")

    B = []
    for T in T_full:
        B.append(
            get_phase_trans_boundary(
                T=T if actual_temp_units == TemperatureUnit.KELVIN else T + cfg.T_0,
                geom=geom,
            )
        )

    X, Y = geom.mesh_grid

    # define the first frame
    cont = plt.contourf(
        X,
        Y,
        _get_temp_in_display_units(T_full[0], actual_temp_units, display_temp_units),
        50,
        cmap="viridis",
        vmin=min_temp,
        vmax=max_temp,
    )
    plt.title("t = 0 min")
    plt.colorbar()
    plt.clim(min_temp, max_temp)
    plt.scatter(B[0][0], B[0][1], s=1, linewidths=0.1, color="r", label="Граница ф.п.")
    plt.legend()

    def update(i):
        cont = plt.contourf(
            X,
            Y,
            _get_temp_in_display_units(
                T_full[i], actual_temp_units, display_temp_units
            ),
            50,
            cmap="viridis",
        )
        plt.scatter(
            B[i][0], B[i][1], s=1, linewidths=0.1, color="r", label="Граница ф.п."
        )
        plt.title(f"t = {round(i * t_step)} min")
        return cont

    anim = animation.FuncAnimation(
        fig, update, frames=len(times), interval=100, blit=False, repeat=True
    )

    if not os.path.exists(directory):
        os.makedirs(directory)

    anim.save(f"{directory}{filename}.gif", dpi=100, writer="imagemagick")
