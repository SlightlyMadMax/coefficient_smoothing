import numba
import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition, BoundaryConditionType
from src.geometry import DomainGeometry
from src.solver import BaseSolver
from src.fluid_dynamics.utils import (
    get_kinematic_visc as visc,
    get_thermal_expansion_coef as th_exp,
    get_indicator_function as c_ind,
)
from src import constants as cfg


class ExplicitNavierStokesSolver(BaseSolver):
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
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._sf: NDArray[np.float64] = np.empty((self.geometry.n_y, self.geometry.n_x))

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_vorticity(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        result: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy = 1.0 / dy
        inv_dy2 = 1.0 / (dy * dy)

        # result[0, :] = -0.5 * inv_dy2 * (8.0 * sf[1, :] - sf[2, :])
        # result[n_y - 1, :] = -0.5 * inv_dy2 * (8.0 * sf[n_y - 2, :] - sf[n_y - 3, :])
        # result[:, 0] = -0.5 * inv_dx2 * (8.0 * sf[:, 1] - sf[:, 2])
        # result[:, n_x - 1] = -0.5 * inv_dx2 * (8.0 * sf[:, n_x - 2] - sf[:, n_x - 3])
        result[0, :] = -2.0 * inv_dy2 * sf[1, :]
        result[n_y - 1, :] = -2.0 * inv_dy2 * sf[n_y - 2, :]
        result[:, 0] = -2.0 * inv_dx2 * sf[:, 1]
        result[:, n_x - 1] = -2.0 * inv_dx2 * sf[:, n_x - 2]

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                result[j, i] = w[j, i] + dt * (
                    -cfg.G
                    * th_exp(u[j, i])
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    + visc(u[j, i])
                    * inv_dx2
                    * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + visc(u[j, i])
                    * inv_dy2
                    * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j + 1, i - 1] - sf[j - 1, i - 1])
                    * w[j, i - 1]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j - 1, i + 1] - sf[j + 1, i + 1])
                    * w[j, i + 1]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j - 1, i - 1] - sf[j - 1, i + 1])
                    * w[j - 1, i]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j + 1, i + 1] - sf[j + 1, i - 1])
                    * w[j + 1, i]
                    # + visc(u[j, i]) * c_ind(u[j, i]) * sf[j, i]
                )

        return result

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_stream_function(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        dx: float,
        dy: float,
        right_value: NDArray[np.float64] = None,
        left_value: NDArray[np.float64] = None,
        top_value: NDArray[np.float64] = None,
        bottom_value: NDArray[np.float64] = None,
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        beta = dx / dy
        factor = 0.5 / (1.0 + beta * beta)

        result = np.copy(sf)

        result[0, :] = top_value
        result[n_y - 1, :] = bottom_value
        result[:, 0] = left_value
        result[:, n_x - 1] = right_value

        temp = np.copy(result)

        for iteration in range(100):
            for i in range(1, n_x - 1):
                for j in range(1, n_y - 1):
                    result[j, i] = factor * (
                        temp[j, i + 1]
                        + result[j, i - 1]
                        + beta * beta * temp[j + 1, i]
                        + beta * beta * result[j - 1, i]
                        + dx * dx * w[j, i]
                    )
            diff = np.linalg.norm(temp - result)
            if diff < 1e-6:
                break
            temp = np.copy(result)
        return result

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float,
    ) -> (NDArray[np.float64], NDArray[np.float64]):
        temp_sf = np.copy(sf)
        for iteration in range(5):
            self._compute_vorticity(
                w=w,
                sf=temp_sf,
                u=u,
                result=self._new_w,
                dx=self.geometry.dx,
                dy=self.geometry.dy,
                dt=self.geometry.dt,
            )
            self._sf = self._compute_stream_function(
                w=self._new_w,
                sf=sf,
                dx=self.geometry.dx,
                dy=self.geometry.dy,
                right_value=(
                    self.right_bc.get_value(t=time)
                    if self.right_bc.boundary_type == BoundaryConditionType.DIRICHLET
                    else None
                ),
                left_value=(
                    self.left_bc.get_value(t=time)
                    if self.left_bc.boundary_type == BoundaryConditionType.DIRICHLET
                    else None
                ),
                top_value=(
                    self.top_bc.get_value(t=time)
                    if self.top_bc.boundary_type == BoundaryConditionType.DIRICHLET
                    else None
                ),
                bottom_value=(
                    self.bottom_bc.get_value(t=time)
                    if self.bottom_bc.boundary_type == BoundaryConditionType.DIRICHLET
                    else None
                ),
            )
            temp_sf = 0.5 * (self._sf + temp_sf)

        return self._sf, self._new_w
