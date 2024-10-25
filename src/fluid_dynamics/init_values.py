import numpy as np
from typing import Tuple

from src.geometry import DomainGeometry


def initialize_velocity_field(
    geom: DomainGeometry, initial_velocity: Tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """
    Initializes a 2D array to store a discretized velocity field.

    :param geom: Object containing geometry information.
    :param N_x: Number of grid points along the x-axis.
    :param N_y: Number of grid points along the y-axis.
    :param initial_velocity: Initial velocity (V_x, V_y) to assign to each point.
    :return: 3D array of shape (N_x, N_y, 2) representing the velocity field.
    """
    # Create a 3D array where each element at (i, j) is a vector (V_x, V_y)
    velocity_field = np.full((geom.n_y, geom.n_x, 2), initial_velocity, dtype=float)
    return velocity_field
