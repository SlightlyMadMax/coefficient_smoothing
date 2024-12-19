import numpy as np

from src.geometry import DomainGeometry


def initialize_stream_function(geom: DomainGeometry) -> np.ndarray:
    return np.zeros((geom.n_y, geom.n_x),)


def initialize_vorticity(geom: DomainGeometry) -> np.ndarray:
    return np.zeros((geom.n_y, geom.n_x),)
