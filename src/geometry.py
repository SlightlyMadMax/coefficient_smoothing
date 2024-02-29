import numpy as np
from decimal import Decimal


class DomainGeometry:
    def __init__(self, width: float, height: float,
                 end_time: float, n_x: int, n_y: int, n_t: int):
        self._width = width
        self._height = height
        self._end_time = end_time
        self._n_x = n_x
        self._n_y = n_y
        self._n_t = n_t
        self._dx = width / (n_x - 1)
        self._dy = height / (n_y - 1)
        self._dt = end_time / (n_t - 1)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def end_time(self):
        return self._end_time

    @property
    def n_x(self):
        return self._n_x

    @property
    def n_y(self):
        return self._n_y

    @property
    def n_t(self):
        return self._n_t

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def dt(self):
        return self._dt

    @property
    def mesh_grid(self):
        x = np.linspace(0, self.width, self.n_x)
        y = np.linspace(0, self.height, self.n_y)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def __str__(self):
        return (f"Width: {self.width}. Height: {self.height}. End time: {int(self.end_time)} s.\n"
                f"x step = {Decimal(self.dx):.2E} m. y step = {Decimal(self.dy):.2E} m. Time step = {round(self.dt)} s")
