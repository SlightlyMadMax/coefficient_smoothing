from abc import abstractmethod

import numba
import numpy as np
from numpy.typing import NDArray

from src.fluid_dynamics.fluid_solver import BaseSolver
from src.geometry import DomainGeometry
import src.parameters as cfg
from src.temperature.coefficient_smoothing.coefficients import c_smoothed, k_smoothed
from src.temperature.coefficient_smoothing.delta import get_max_delta
from src.temperature import boundary_conditions as bc
from src.utils import solve_tridiagonal


class HeatTransferSolver(BaseSolver):
    def __init__(
        self,
        geometry: DomainGeometry,
        top_cond_type: int,
        right_cond_type: int,
        bottom_cond_type: int,
        left_cond_type: int,
        fixed_delta: bool = False,
    ):
        super().__init__(
            geometry=geometry,
            top_cond_type=top_cond_type,
            right_cond_type=right_cond_type,
            bottom_cond_type=bottom_cond_type,
            left_cond_type=left_cond_type,
        )
        self.fixed_delta = fixed_delta
        # Pre-allocate arrays
        self._temp_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._iter_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._new_u: NDArray[np.float64] = np.empty(
            (self.geometry.n_y, self.geometry.n_x)
        )
        self._a_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._b_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._c_x: NDArray[np.float64] = np.empty((self.geometry.n_x - 1))
        self._a_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._b_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))
        self._c_y: NDArray[np.float64] = np.empty((self.geometry.n_y - 1))

    @abstractmethod
    def solve(
        self, u: NDArray[np.float64], time: float = 0.0, iters: int = 1
    ) -> NDArray[np.float64]: ...


class LocOneDimSolver(HeatTransferSolver):
    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(
        u: NDArray[np.float64],
        iter_u: NDArray[np.float64],
        result: NDArray[np.float64],
        a_x: NDArray[np.float64],
        b_x: NDArray[np.float64],
        c_x: NDArray[np.float64],
        dx: float,
        dt: float,
        right_cond_type: int,
        left_cond_type: int,
        delta: float,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        n_y, n_x = u.shape
        inv_dx2 = 1.0 / (dx * dx)

        lbc = bc.get_left_bc_1(time, n_y)
        rbc = bc.get_right_bc_1(time, n_y)

        for j in range(1, n_y - 1):
            for i in range(0, n_x - 1):
                inv_c = 1.0 / c_smoothed(iter_u[j, i], delta)

                # Коэффициент при T_(i+1,j)^(n+1/2)
                a_x[i] = (
                    -dt
                    * k_smoothed(0.5 * (iter_u[j, i + 1] + iter_u[j, i]), delta)
                    * inv_c
                    * inv_dx2
                )

                # Коэффициент при T_(i,j)^(n+1/2)
                b_x[i] = (
                    1.0
                    + dt
                    * (
                        k_smoothed(0.5 * (iter_u[j, i + 1] + iter_u[j, i]), delta)
                        + k_smoothed(0.5 * (iter_u[j, i] + iter_u[j, i - 1]), delta)
                    )
                    * inv_c
                    * inv_dx2
                )

                # Коэффициент при T_(i-1,j)^(n+1/2)
                c_x[i] = (
                    -dt
                    * k_smoothed(0.5 * (iter_u[j, i] + iter_u[j, i - 1]), delta)
                    * inv_c
                    * inv_dx2
                )

            result[j, :] = solve_tridiagonal(
                a=a_x,
                b=b_x,
                c=c_x,
                f=u[j, :],
                left_type=left_cond_type,
                left_value=lbc[0],
                right_type=right_cond_type,
                right_value=rbc[0],
                h=dx,
            )

        result[0, :] = u[0, :]
        result[n_y - 1, :] = u[n_y - 1, :]

        return result

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_y(
        u: NDArray[np.float64],
        iter_u: NDArray[np.float64],
        result: NDArray[np.float64],
        a_y: NDArray[np.float64],
        b_y: NDArray[np.float64],
        c_y: NDArray[np.float64],
        dy: float,
        dt: float,
        top_cond_type: int,
        bottom_cond_type: int,
        delta: float,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        n_y, n_x = u.shape
        inv_dy2 = 1.0 / (dy * dy)

        bbc = bc.get_bottom_bc_1(time, n_x)
        tbc = bc.get_top_bc_1(time, n_x)
        phi = bc.get_top_bc_2(time)
        psi, ksi = bc.get_top_bc_3(time)

        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                inv_c = 1.0 / c_smoothed(iter_u[j, i], delta)

                # Коэффициент при T_(i,j-1)^n
                a_y[j] = (
                    -dt
                    * k_smoothed(0.5 * (iter_u[j + 1, i] + iter_u[j, i]), delta)
                    * inv_c
                    * inv_dy2
                )

                # Коэффициент при T_(i,j)^n
                b_y[j] = (
                    1.0
                    + dt
                    * (
                        k_smoothed(0.5 * (iter_u[j + 1, i] + iter_u[j, i]), delta)
                        + k_smoothed(0.5 * (iter_u[j, i] + iter_u[j - 1, i]), delta)
                    )
                    * inv_c
                    * inv_dy2
                )

                # Коэффициент при T_(i,j+1)^n
                c_y[j] = (
                    -dt
                    * k_smoothed(0.5 * (iter_u[j, i] + iter_u[j - 1, i]), delta)
                    * inv_c
                    * inv_dy2
                )

            result[:, i] = solve_tridiagonal(
                a=a_y,
                b=b_y,
                c=c_y,
                f=u[:, i],
                left_type=bottom_cond_type,
                left_value=bbc[0],
                right_type=top_cond_type,
                right_value=tbc[0],
                right_phi=phi,
                right_psi=psi,
                right_ksi=ksi,
                h=dy,
            )

        result[:, 0] = u[:, 0]
        result[:, n_x - 1] = u[:, n_x - 1]

        return result

    def solve(
        self, u: NDArray[np.float64], time: float = 0.0, iters: int = 1
    ) -> NDArray[np.float64]:

        self._iter_u = np.copy(u)

        # Run the x-direction sweep iterations
        for i in range(iters):
            delta = cfg.delta if self.fixed_delta else get_max_delta(self._iter_u)
            self._temp_u = self._compute_sweep_x(
                u=u,
                iter_u=self._iter_u,
                result=self._temp_u,
                a_x=self._a_x,
                b_x=self._b_x,
                c_x=self._c_x,
                dx=self.geometry.dx,
                dt=self.geometry.dt,
                right_cond_type=self.right_cond_type,
                left_cond_type=self.left_cond_type,
                delta=delta,
                time=time,
            )
            self._iter_u = np.copy(self._temp_u)

        # Run the y-direction sweep iterations
        for i in range(iters):
            delta = cfg.delta if self.fixed_delta else get_max_delta(self._temp_u)
            self._new_u = self._compute_sweep_y(
                u=self._temp_u,
                iter_u=self._iter_u,
                result=self._new_u,
                a_y=self._a_y,
                b_y=self._b_y,
                c_y=self._c_y,
                dy=self.geometry.dy,
                dt=self.geometry.dt,
                top_cond_type=self.top_cond_type,
                bottom_cond_type=self.bottom_cond_type,
                delta=delta,
                time=time,
            )
            self._iter_u = np.copy(self._new_u)

        return self._new_u


class AltDirSolver(HeatTransferSolver):
    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(
        u: NDArray[np.float64],
        iter_u: NDArray[np.float64],
        temp_u: NDArray[np.float64],
        alpha: NDArray[np.float64],
        beta: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        right_cond_type: int,
        left_cond_type: int,
        delta: float,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        n_y, n_x = u.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)

        lbc = bc.get_left_bc_1(time, n_y)
        rbc = bc.get_right_bc_1(time, n_y)

        for j in range(1, n_y - 1):
            if left_cond_type == cfg.DIRICHLET:
                alpha[0] = 0.0
                beta[0] = lbc[j]
            else:
                alpha[0] = 1.0
                beta[0] = 0.0
            for i in range(1, n_x - 1):
                inv_c = 1.0 / c_smoothed(u[j, i], delta)
                a_i = (
                    -dt
                    * 0.5
                    * k_smoothed(0.5 * (u[j, i + 1] + u[j, i]), delta)
                    * inv_c
                    * inv_dx2
                )
                b_i = (
                    1.0
                    + dt
                    * (
                        k_smoothed(0.5 * (u[j, i + 1] + u[j, i]), delta)
                        + k_smoothed(0.5 * (u[j, i] + u[j, i - 1]), delta)
                    )
                    * inv_c
                    * inv_dx2
                    * 0.5
                )
                c_i = (
                    -dt
                    * 0.5
                    * k_smoothed(0.5 * (u[j, i] + u[j, i - 1]), delta)
                    * inv_c
                    * inv_dx2
                )

                rhs_i = u[j, i] + dt * 0.5 * inv_c * inv_dy2 * (
                    k_smoothed(0.5 * (u[j + 1, i] + u[j, i]), delta)
                    * (u[j + 1, i] - u[j, i])
                    - k_smoothed(0.5 * (u[j, i] + u[j - 1, i]), delta)
                    * (u[j, i] - u[j - 1, i])
                )

                alpha[i] = -a_i / (b_i + c_i * alpha[i - 1])
                beta[i] = (rhs_i - c_i * beta[i - 1]) / (b_i + c_i * alpha[i - 1])

            temp_u[j, n_x - 1] = (
                rbc[j]
                if right_cond_type == 1
                else beta[n_x - 2] / (1.0 - alpha[n_x - 2])  # Neumann
            )

            for i in range(n_x - 2, -1, -1):
                temp_u[j, i] = alpha[i] * temp_u[j, i + 1] + beta[i]

        temp_u[0, :] = u[0, :]
        temp_u[n_y - 1, :] = u[n_y - 1, :]

        return temp_u

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_y(
        temp_u: NDArray[np.float64],
        iter_u: NDArray[np.float64],
        new_u: NDArray[np.float64],
        alpha: NDArray[np.float64],
        beta: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        top_cond_type: int,
        bottom_cond_type: int,
        delta: float,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        n_y, n_x = temp_u.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)

        bbc = bc.get_bottom_bc_1(time, n_x)
        tbc = bc.get_top_bc_1(time, n_x)
        phi = bc.get_top_bc_2(time)
        psi, ksi = bc.get_top_bc_3(time)

        for i in range(1, n_x - 1):
            if bottom_cond_type == cfg.DIRICHLET:
                alpha[0] = 0.0
                beta[0] = bbc[i]
            else:  # Neumann
                alpha[0] = 1.0
                beta[0] = 0.0
            for j in range(1, n_y - 1):
                inv_c = 1.0 / c_smoothed(temp_u[j, i], delta)
                a_j = (
                    -dt
                    * 0.5
                    * k_smoothed(0.5 * (temp_u[j + 1, i] + temp_u[j, i]), delta)
                    * inv_c
                    * inv_dy2
                )
                b_j = (
                    1.0
                    + dt
                    * (
                        k_smoothed(0.5 * (temp_u[j + 1, i] + temp_u[j, i]), delta)
                        + k_smoothed(0.5 * (temp_u[j, i] + temp_u[j - 1, i]), delta)
                    )
                    * inv_c
                    * inv_dy2
                    * 0.5
                )
                c_j = (
                    -dt
                    * 0.5
                    * k_smoothed(0.5 * (temp_u[j, i] + temp_u[j - 1, i]), delta)
                    * inv_c
                    * inv_dy2
                )

                rhs_j = temp_u[j, i] + dt * 0.5 * inv_c * inv_dx2 * (
                    k_smoothed(0.5 * (temp_u[j, i + 1] + temp_u[j, i]), delta)
                    * (temp_u[j, i + 1] - temp_u[j, i])
                    - k_smoothed(0.5 * (temp_u[j, i] + temp_u[j, i - 1]), delta)
                    * (temp_u[j, i] - temp_u[j, i - 1])
                )

                alpha[j] = -a_j / (b_j + c_j * alpha[j - 1])
                beta[j] = (rhs_j - c_j * beta[j - 1]) / (b_j + c_j * alpha[j - 1])

            if top_cond_type == cfg.DIRICHLET:
                new_u[n_y - 1, i] = tbc[i]
            elif top_cond_type == cfg.NEUMANN:
                new_u[n_y - 1, i] = (dy * phi + beta[n_y - 2]) / (1.0 - alpha[n_y - 2])
            else:  # ROBIN
                new_u[n_y - 1, i] = (dy * psi + beta[n_y - 2]) / (
                    1 - alpha[n_y - 2] - dy * ksi
                )

            # Вычисление температуры на новом временном слое
            for j in range(n_y - 2, -1, -1):
                new_u[j, i] = alpha[j] * new_u[j + 1, i] + beta[j]

        new_u[:, 0] = temp_u[:, 0]
        new_u[:, n_x - 1] = temp_u[:, n_x - 1]

        return new_u

    def solve(
        self, u: NDArray[np.float64], time: float = 0.0, iters: int = 1
    ) -> NDArray[np.float64]:

        self._iter_u = np.copy(u)

        # Run the x-direction sweep iterations
        for i in range(iters):
            delta = cfg.delta if self.fixed_delta else get_max_delta(self._iter_u)
            self._temp_u = self._compute_sweep_x(
                u=u,
                iter_u=self._iter_u,
                temp_u=self._temp_u,
                alpha=self._alpha,
                beta=self._beta,
                dx=self.geometry.dx,
                dy=self.geometry.dy,
                dt=self.geometry.dt,
                right_cond_type=self.right_cond_type,
                left_cond_type=self.left_cond_type,
                delta=delta,
                time=time,
            )
            self._iter_u = np.copy(self._temp_u)

        # Run the y-direction sweep iterations
        for i in range(iters):
            delta = cfg.delta if self.fixed_delta else get_max_delta(self._temp_u)
            self._new_u = self._compute_sweep_y(
                temp_u=self._temp_u,
                iter_u=self._iter_u,
                new_u=self._new_u,
                alpha=self._alpha,
                beta=self._beta,
                dx=self.geometry.dx,
                dy=self.geometry.dy,
                dt=self.geometry.dt,
                top_cond_type=self.top_cond_type,
                bottom_cond_type=self.bottom_cond_type,
                delta=delta,
                time=time,
            )
            self._iter_u = np.copy(self._new_u)

        return self._new_u
