import numpy as np

from enum import Enum
from numpy import ndarray
from typing import Tuple

import src.parameters as cfg
from src.geometry import DomainGeometry


class TemperatureShape(Enum):
    LINEAR = "linear"
    CIRCLE = "circle"
    DOUBLE_CIRCLE = "double_circle"
    PACMAN = "pacman"
    SQUARE = "square"


def init_temperature(geom: DomainGeometry, F: ndarray) -> ndarray:
    """
    Initializes the temperature field based on the given interface F.

    :param geom: Object containing geometry information.
    :param F: 1D array representing the interface position for the phase transition.
    :return: 2D array of temperatures initialized based on the interface.
    """
    T = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        for j in range(geom.n_y):
            if j * geom.dy < F[i]:
                T[j, i] = cfg.T_ICE_MIN + j * geom.dy * (cfg.T_0 - cfg.T_ICE_MIN) / (
                    geom.height - cfg.WATER_H
                )
            elif j * geom.dy > F[i]:
                T[j, i] = cfg.T_WATER_MAX
            else:
                T[j, i] = cfg.T_0

    return T


def init_temperature_shape(
    geom: DomainGeometry,
    shape: TemperatureShape,
    water_temp: float = cfg.T_WATER_MAX,
    ice_temp: float = cfg.T_ICE_MIN,
    radius: float = 0.25,
    small_radius: float = 0.1,
    square_size: float = 0.5,
    eye_radius: float = 0.05,
    eye_offset: float = 0.6,
) -> ndarray:
    """
    Initializes the temperature field based on a specified shape.

    :param geom: Object containing geometry information.
    :param shape: The shape of the temperature distribution (circle, double circle, square, or pacman).
    :param water_temp: The temperature assigned to water regions (default: T_WATER_MAX).
    :param ice_temp: The temperature assigned to ice regions (default: T_ICE_MIN).
    :param radius: The radius used for circular shapes (default: 0.25).
    :param small_radius: A smaller radius for additional features in shapes (default: 0.1).
    :param square_size: The size of the square region (default: 0.5).
    :param eye_radius: The radius of the eye in the Pacman shape (default: 0.05).
    :param eye_offset: The offset for positioning the eye in the Pacman shape (default: 0.6).
    :return: A 2D array of temperatures initialized based on the specified shape.
    """
    T = np.full((geom.n_y, geom.n_x), ice_temp)

    X, Y = geom.mesh_grid

    if shape == TemperatureShape.LINEAR:
        # Linear temperature gradient from bottom (ice) to top (water)
        T[:, :] = np.linspace(ice_temp, water_temp, geom.n_y).reshape(1, -1)

    elif shape == TemperatureShape.CIRCLE:
        # Single circle centered at domain center with radius threshold
        mask = (X - geom.width / 2) ** 2 + (Y - geom.height / 2) ** 2 < radius**2
        T[mask] = water_temp

    elif shape == TemperatureShape.DOUBLE_CIRCLE:
        # Two circles centered vertically with specified radius
        mask1 = (X - geom.width / 2) ** 2 + (
            Y - 0.75 * geom.height
        ) ** 2 < small_radius**2
        mask2 = (X - geom.width / 2) ** 2 + (
            Y - 0.25 * geom.height
        ) ** 2 < small_radius**2
        T[mask1 | mask2] = water_temp

    elif shape == TemperatureShape.PACMAN:
        for i in range(geom.n_x):
            for j in range(geom.n_y):
                if (i * geom.dx - geom.width / 2.0) ** 2 + (
                    j * geom.dy - geom.height / 2.0
                ) ** 2 < radius**2:
                    if i * geom.dx <= j * geom.dy <= -i * geom.dx + 1:
                        T[j, i] = ice_temp  # Pacman's mouth
                    elif (i * geom.dx - eye_offset) ** 2 + (
                        j * geom.dy - eye_offset
                    ) ** 2 < eye_radius**2:
                        T[j, i] = ice_temp  # Pacman's eye
                    else:
                        T[j, i] = water_temp  # Pacman's body
                else:
                    T[j, i] = ice_temp

    elif shape == TemperatureShape.SQUARE:
        # Square shape centered in the domain with side length "square_size"
        half_size = square_size / 2
        mask = (np.abs(X - geom.width / 2) < half_size) & (
            np.abs(Y - geom.height / 2) < half_size
        )
        T[mask] = water_temp

    return T


def init_temperature_lake(
    geom: DomainGeometry,
    lake_data: Tuple[ndarray, ndarray],
    water_temp: float,
    ice_temp: float,
) -> ndarray:
    """
    Initialize temperature for a lake profile using preloaded thickness data.

    :param geom: Object containing geometry information.
    :param lake_data: Preloaded water and ice thickness grids.
    :param water_temp: Temperature for water.
    :param ice_temp: Temperature for ice.
    :return: 2D temperature field array.
    """
    water_th_grid, ice_th_grid = lake_data

    grid_x = water_th_grid[0]
    grid_step = grid_x[1] - grid_x[0]
    print(f"Grid step: {grid_step}")

    lake_width = grid_x[-1]
    print(f"Lake width: {lake_width}")

    new_x = [i * geom.dx for i in range(int(lake_width / geom.dx + 1))]
    print(new_x[-1], len(new_x))

    water_th_interp = np.interp(new_x, grid_x, water_th_grid[1])

    ice_th_interp = np.interp(new_x, grid_x, ice_th_grid[1])

    print(f"Max lake thickness {max(water_th_interp)}")

    T = np.empty((geom.n_y, geom.n_x))

    for i in range(geom.n_x):
        x = i * geom.dx
        ice_th_at_x, water_th_at_x = 0.0, 0.0

        if (geom.width - lake_width) / 2.0 <= x <= (geom.width + lake_width) / 2.0:
            water_th_at_x = water_th_interp[i + int((len(new_x) - geom.n_x) / 2)]
            ice_th_at_x = ice_th_interp[i + int((len(new_x) - geom.n_x) / 2)]

        for j in range(geom.n_y):
            y = geom.height - j * geom.dy
            if water_th_at_x > 0.0 and ice_th_at_x <= y <= ice_th_at_x + water_th_at_x:
                T[j, i] = water_temp
            else:
                T[j, i] = ice_temp + (j / geom.n_y) * (cfg.T_0 - ice_temp)
    return T
