import numba
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from src.geometry import DomainGeometry


class BaseSolver(ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        top_cond_type: int,
        right_cond_type: int,
        bottom_cond_type: int,
        left_cond_type: int,
    ):
        self.geometry = geometry
        self.top_cond_type = top_cond_type
        self.right_cond_type = right_cond_type
        self.bottom_cond_type = bottom_cond_type
        self.left_cond_type = left_cond_type

    @abstractmethod
    def solve(self, *kwargs) -> NDArray[np.float64]: ...


class SweepSolver2D(BaseSolver, ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        top_cond_type: int,
        right_cond_type: int,
        bottom_cond_type: int,
        left_cond_type: int,
    ):
        super().__init__(
            geometry=geometry,
            top_cond_type=top_cond_type,
            right_cond_type=right_cond_type,
            bottom_cond_type=bottom_cond_type,
            left_cond_type=left_cond_type,
        )
        self._a_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._b_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._c_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._a_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._b_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._c_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))

    @staticmethod
    @abstractmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(*kwargs) -> NDArray[np.float64]: ...

    @staticmethod
    @abstractmethod
    @numba.jit(nopython=True)
    def _compute_sweep_y(*kwargs) -> NDArray[np.float64]: ...
