import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from src.geometry import DomainGeometry


def plot_velocity_field(
    v_x: NDArray[np.float64],
    v_y: NDArray[np.float64],
    geometry: DomainGeometry,
    graph_id: int,
    show_graph: bool = True,
    directory: str = "../graphs/velocity/",
    equal_aspect: Optional[bool] = True,
):
    n_y, n_x = v_x.shape
    dx, dy = geometry.dx, geometry.dy
    x = np.linspace(0, (n_x - 1) * dx, n_x)
    y = np.linspace(0, (n_y - 1) * dy, n_y)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, v_x, v_y, angles="xy", scale_units="xy", scale=1, color="blue")

    plt.xlabel("x, м")
    plt.ylabel("y, м")
    plt.title("Поле скоростей")

    if equal_aspect:
        plt.axis("equal")

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}v_{graph_id}.png")

    if show_graph:
        plt.show()
    else:
        plt.close()
