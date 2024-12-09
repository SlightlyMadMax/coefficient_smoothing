import numba
import numpy as np
from numpy.typing import NDArray

from src.geometry import DomainGeometry
from src.solver import SweepSolver2D
from src.fluid_dynamics.utils import (
    get_kinematic_visc as visc,
    get_thermal_expansion_coef as th_exp,
    get_indicator_function as c_ind,
)
from src.utils import solve_tridiagonal
from src import parameters as cfg


class NavierStokesSolver(SweepSolver2D):
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
        self._temp_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._new_w: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._sf: NDArray[np.float64] = np.empty((self.geometry.n_y, self.geometry.n_x))

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        result: NDArray[np.float64],
        a_x: NDArray[np.float64],
        b_x: NDArray[np.float64],
        c_x: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        right_cond_type: int,
        left_cond_type: int,
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy = 1.0 / dy
        inv_dy2 = 1.0 / (dy * dy)

        f = np.empty(n_x)

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                a_x[i] = (
                    0.5
                    * dt
                    * inv_dx
                    * (
                        (sf[j + 1, i + 1] - sf[j - 1, i + 1]) * 0.25 * inv_dy
                        - visc(u[j, i]) * inv_dx
                    )
                )

                b_x[i] = 1.0 + visc(u[j, i]) * dt * inv_dx2

                c_x[i] = (
                    0.5
                    * dt
                    * inv_dx
                    * (
                        (sf[j - 1, i - 1] - sf[j + 1, i - 1]) * 0.25 * inv_dy
                        - visc(u[j, i]) * inv_dx
                    )
                )

                f[i] = w[j, i] + 0.5 * dt * (
                    -cfg.G
                    * th_exp(u[j, i])
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    + visc(u[j, i])
                    * inv_dy2
                    * (w[j + 1, i] - 2.0 * w[j, i] + w[j - 1, i])
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j - 1, i + 1] - sf[j - 1, i - 1])
                    * w[j - 1, i]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j + 1, i - 1] - sf[j + 1, i + 1])
                    * w[j + 1, i]
                    + visc(u[j, i]) * c_ind(u[j, i]) * sf[j, i]
                )

            result[j, :] = solve_tridiagonal(
                a=a_x,
                b=b_x,
                c=c_x,
                f=f,
                left_type=left_cond_type,
                left_value=-0.5 * inv_dx2 * (8.0 * sf[j, 1] - sf[j, 2]),
                right_type=right_cond_type,
                right_value=-0.5 * inv_dx2 * (8.0 * sf[j, n_x - 2] - sf[j, n_x - 3]),
                h=dx,
            )

        result[0, :] = -0.5 * inv_dy2 * (8.0 * sf[1, :] - sf[2, :])
        result[n_y - 1, :] = -0.5 * inv_dy2 * (8.0 * sf[n_y - 2, :] - sf[n_y - 3, :])

        return result

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_y(
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        result: NDArray[np.float64],
        a_y: NDArray[np.float64],
        b_y: NDArray[np.float64],
        c_y: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        top_cond_type: int,
        bottom_cond_type: int,
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        inv_dx = 1.0 / dx
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy = 1.0 / dy
        inv_dy2 = 1.0 / (dy * dy)

        f = np.empty(n_y)

        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                a_y[j] = (
                    0.5
                    * dt
                    * inv_dy
                    * (
                        (sf[j + 1, i - 1] - sf[j + 1, i + 1]) * 0.25 * inv_dx
                        - visc(u[j, i]) * inv_dy
                    )
                )

                b_y[j] = 1.0 + visc(u[j, i]) * dt * inv_dy2

                c_y[j] = (
                    0.5
                    * dt
                    * inv_dy
                    * (
                        (sf[j - 1, i + 1] - sf[j - 1, i - 1]) * 0.25 * inv_dx
                        - visc(u[j, i]) * inv_dy
                    )
                )

                f[j] = w[j, i] + 0.5 * dt * (
                    -cfg.G
                    * th_exp(u[j, i])
                    * 0.5
                    * inv_dx
                    * (u[j, i + 1] - u[j, i - 1])
                    + visc(u[j, i])
                    * inv_dx2
                    * (w[j, i + 1] - 2.0 * w[j, i] + w[j, i - 1])
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j - 1, i - 1] - sf[j + 1, i - 1])
                    * w[j, i - 1]
                    + 0.25
                    * inv_dy
                    * inv_dx
                    * (sf[j + 1, i + 1] - sf[j - 1, i + 1])
                    * w[j, i + 1]
                    + visc(u[j, i]) * c_ind(u[j, i]) * sf[j, i]
                )

            result[:, i] = solve_tridiagonal(
                a=a_y,
                b=b_y,
                c=c_y,
                f=f,
                left_type=bottom_cond_type,
                left_value=-0.5 * inv_dy2 * (8.0 * sf[1, i] - sf[2, i]),
                right_type=top_cond_type,
                right_value=-0.5 * inv_dy2 * (8.0 * sf[n_y - 2, i] - sf[n_y - 3, i]),
                h=dy,
            )

        result[:, 0] = -0.5 * inv_dx2 * (8.0 * sf[:, 1] - sf[:, 2])
        result[:, n_x - 1] = -0.5 * inv_dx2 * (8.0 * sf[:, n_x - 2] - sf[:, n_x - 3])

        return result

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_stream_function(
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        dx: float,
        dy: float,
    ) -> NDArray[np.float64]:
        n_y, n_x = w.shape
        beta = dx / dy
        factor = 0.5 / (1.0 + beta * beta)

        result = np.copy(sf)

        result[0, :] = 0.0
        result[n_y - 1, :] = 0.0
        result[:, 0] = 0.0
        result[:, n_x - 1] = 0.0

        temp = np.copy(result)

        for iteration in range(50):
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
    ) -> (NDArray[np.float64], NDArray[np.float64]):
        self._temp_w = self._compute_sweep_x(
            w=w,
            sf=sf,
            u=u,
            result=self._temp_w,
            a_x=self._a_x,
            b_x=self._b_x,
            c_x=self._c_x,
            dx=self.geometry.dx,
            dy=self.geometry.dy,
            dt=self.geometry.dt,
            right_cond_type=self.right_cond_type,
            left_cond_type=self.left_cond_type,
        )
        self._new_w = self._compute_sweep_y(
            w=self._temp_w,
            sf=sf,
            u=u,
            result=self._new_w,
            a_y=self._a_y,
            b_y=self._b_y,
            c_y=self._c_y,
            dx=self.geometry.dx,
            dy=self.geometry.dy,
            dt=self.geometry.dt,
            top_cond_type=self.top_cond_type,
            bottom_cond_type=self.bottom_cond_type,
        )
        self._sf = self._compute_stream_function(
            w=self._new_w,
            sf=sf,
            dx=self.geometry.dx,
            dy=self.geometry.dy,
        )

        return self._sf, self._new_w
