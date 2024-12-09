import numba
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from src.boundary_conditions import BoundaryCondition
from src.geometry import DomainGeometry


class BaseSolver(ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
    ):
        self.geometry = geometry
        self.top_bc = top_bc
        self.right_bc = right_bc
        self.bottom_bc = bottom_bc
        self.left_bc = left_bc

    @abstractmethod
    def solve(self, *kwargs) -> NDArray[np.float64]: ...


class SweepSolver2D(BaseSolver, ABC):
    def __init__(
        self,
        geometry: DomainGeometry,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
    ):
        super().__init__(
            geometry=geometry,
            top_bc=top_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            left_bc=left_bc,
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
