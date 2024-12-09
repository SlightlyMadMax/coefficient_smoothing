import numba
import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryConditionType
import src.parameters as cfg
from src.temperature.coefficient_smoothing.coefficients import c_smoothed, k_smoothed
from src.temperature.coefficient_smoothing.delta import get_max_delta
from src.temperature.solvers.base import HeatTransferSolver
from src.utils import solve_tridiagonal


class LocOneDimSolver(HeatTransferSolver):
    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(
        u: NDArray[np.float64],
        iter_u: NDArray[np.float64],
        sf: NDArray[np.float64],
        result: NDArray[np.float64],
        a_x: NDArray[np.float64],
        b_x: NDArray[np.float64],
        c_x: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        delta: float,
        rbc_type: int,
        lbc_type: int,
        right_value: NDArray[np.float64] = None,
        left_value: NDArray[np.float64] = None,
        right_flux: NDArray[np.float64] = None,
        left_flux: NDArray[np.float64] = None,
        right_psi: NDArray[np.float64] = None,
        left_psi: NDArray[np.float64] = None,
        right_phi: NDArray[np.float64] = None,
        left_phi: NDArray[np.float64] = None,
    ) -> NDArray[np.float64]:
        n_y, n_x = u.shape
        inv_dx = 1.0 / dx
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy = 1.0 / dy

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c = 1.0 / c_smoothed(iter_u[j, i], delta)

                # Coefficient at T_{i + 1, j}^{n + 1/2}
                a_x[i] = (
                    dt
                    * inv_dx
                    * (
                        # 0.125
                        # * inv_dy
                        # * (
                        #     sf[j + 1, i]
                        #     - sf[j - 1, i]
                        #     + sf[j + 1, i + 1]
                        #     - sf[j - 1, i + 1]
                        # )
                        -k_smoothed(0.5 * (iter_u[j, i + 1] + iter_u[j, i]), delta)
                        * inv_c
                        * inv_dx
                    )
                )

                # Coefficient at T_{i, j}^{n + 1/2}
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

                # Coefficient at T_{i - 1, j}^{n + 1/2}
                c_x[i] = (
                    -dt
                    * inv_dx
                    * (
                        # 0.125
                        # * inv_dy
                        # * (
                        #     sf[j + 1, i]
                        #     - sf[j - 1, i]
                        #     + sf[j + 1, i - 1]
                        #     - sf[j - 1, i - 1]
                        # ) +
                        k_smoothed(0.5 * (iter_u[j, i] + iter_u[j, i - 1]), delta)
                        * inv_c
                        * inv_dx
                    )
                )

            result[j, :] = solve_tridiagonal(
                a=a_x,
                b=b_x,
                c=c_x,
                f=u[j, :],
                left_type=lbc_type,
                left_value=left_value[j] if left_value is not None else 0.0,
                left_flux=left_flux[j] if left_flux is not None else 0.0,
                left_psi=left_psi[j] if left_psi is not None else 0.0,
                left_phi=left_phi[j] if left_phi is not None else 0.0,
                right_type=rbc_type,
                right_value=right_value[j] if right_value is not None else 0.0,
                right_flux=right_flux[j] if right_flux is not None else 0.0,
                right_psi=right_psi[j] if right_psi is not None else 0.0,
                right_phi=right_phi[j] if right_phi is not None else 0.0,
                h=dx,
            )

        return result

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_y(
        u: NDArray[np.float64],
        iter_u: NDArray[np.float64],
        sf: NDArray[np.float64],
        result: NDArray[np.float64],
        a_y: NDArray[np.float64],
        b_y: NDArray[np.float64],
        c_y: NDArray[np.float64],
        dx: float,
        dy: float,
        dt: float,
        delta: float,
        tbc_type: int,
        bbc_type: int,
        top_value: NDArray[np.float64] = None,
        bottom_value: NDArray[np.float64] = None,
        top_flux: NDArray[np.float64] = None,
        bottom_flux: NDArray[np.float64] = None,
        top_psi: NDArray[np.float64] = None,
        bottom_psi: NDArray[np.float64] = None,
        top_phi: NDArray[np.float64] = None,
        bottom_phi: NDArray[np.float64] = None,
    ) -> NDArray[np.float64]:
        n_y, n_x = u.shape
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_dy2 = 1.0 / (dy * dy)

        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                inv_c = 1.0 / c_smoothed(iter_u[j, i], delta)

                # Coefficient at T_{i, j + 1}^{n + 1}
                a_y[j] = (
                    dt
                    * inv_dy
                    * (
                        # 0.125
                        # * inv_dx
                        # * (
                        #     sf[j, i - 1]
                        #     - sf[j, i + 1]
                        #     + sf[j + 1, i - 1]
                        #     - sf[j + 1, i + 1]
                        # )
                        -k_smoothed(0.5 * (iter_u[j + 1, i] + iter_u[j, i]), delta)
                        * inv_c
                        * inv_dy
                    )
                )

                # Coefficient at T_{i, j}^{n + 1}
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

                # Coefficient at T_{i, j - 1}^{n + 1}
                c_y[j] = (
                    -dt
                    * inv_dy
                    * (
                        # 0.125
                        # * inv_dx
                        # * (
                        #     sf[j, i - 1]
                        #     - sf[j, i + 1]
                        #     + sf[j - 1, i - 1]
                        #     - sf[j - 1, i + 1]
                        # ) +
                        k_smoothed(0.5 * (iter_u[j, i] + iter_u[j - 1, i]), delta)
                        * inv_c
                        * inv_dy
                    )
                )

            result[:, i] = solve_tridiagonal(
                a=a_y,
                b=b_y,
                c=c_y,
                f=u[:, i],
                left_type=bbc_type,
                left_value=bottom_value[i] if bottom_value is not None else 0.0,
                left_flux=bottom_flux[i] if bottom_flux is not None else 0.0,
                left_psi=bottom_psi[i] if bottom_psi is not None else 0.0,
                left_phi=bottom_phi[i] if bottom_phi is not None else 0.0,
                right_type=tbc_type,
                right_value=top_value[i] if top_value is not None else 0.0,
                right_flux=top_flux[i] if top_flux is not None else 0.0,
                right_psi=top_psi[i] if top_psi is not None else 0.0,
                right_phi=top_phi[i] if top_phi is not None else 0.0,
                h=dy,
            )

        return result

    def solve(
        self,
        u: NDArray[np.float64],
        sf: NDArray[np.float64],
        time: float = 0.0,
        iters: int = 1,
    ) -> NDArray[np.float64]:

        self._iter_u = np.copy(u)

        # Run the x-direction sweep iterations
        for i in range(iters):
            delta = cfg.delta if self.fixed_delta else get_max_delta(self._iter_u)
            self._temp_u = self._compute_sweep_x(
                u=u,
                iter_u=self._iter_u,
                sf=sf,
                result=u,
                a_x=self._a_x,
                b_x=self._b_x,
                c_x=self._c_x,
                dx=self.geometry.dx,
                dy=self.geometry.dy,
                dt=self.geometry.dt,
                delta=delta,
                rbc_type=self.right_bc.boundary_type.value,
                lbc_type=self.left_bc.boundary_type.value,
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
                right_flux=(
                    self.right_bc.get_flux(t=time)
                    if self.right_bc.boundary_type == BoundaryConditionType.NEUMANN
                    else None
                ),
                left_flux=(
                    self.left_bc.get_flux(t=time)
                    if self.left_bc.boundary_type == BoundaryConditionType.NEUMANN
                    else None
                ),
                right_psi=(
                    self.right_bc.get_psi(t=time)
                    if self.right_bc.boundary_type == BoundaryConditionType.ROBIN
                    else None
                ),
                left_psi=(
                    self.left_bc.get_psi(t=time)
                    if self.left_bc.boundary_type == BoundaryConditionType.ROBIN
                    else None
                ),
                right_phi=(
                    self.right_bc.get_phi(t=time)
                    if self.right_bc.boundary_type == BoundaryConditionType.ROBIN
                    else None
                ),
                left_phi=(
                    self.left_bc.get_phi(t=time)
                    if self.left_bc.boundary_type == BoundaryConditionType.ROBIN
                    else None
                ),
            )
            self._iter_u = np.copy(self._temp_u)

        # Run the y-direction sweep iterations
        for i in range(iters):
            delta = cfg.delta if self.fixed_delta else get_max_delta(self._temp_u)
            self._new_u = self._compute_sweep_y(
                u=self._temp_u,
                iter_u=self._iter_u,
                sf=sf,
                result=self._temp_u,
                a_y=self._a_y,
                b_y=self._b_y,
                c_y=self._c_y,
                dx=self.geometry.dx,
                dy=self.geometry.dy,
                dt=self.geometry.dt,
                delta=delta,
                tbc_type=self.top_bc.boundary_type.value,
                bbc_type=self.bottom_bc.boundary_type.value,
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
                top_flux=(
                    self.top_bc.get_flux(t=time)
                    if self.top_bc.boundary_type == BoundaryConditionType.NEUMANN
                    else None
                ),
                bottom_flux=(
                    self.bottom_bc.get_flux(t=time)
                    if self.bottom_bc.boundary_type == BoundaryConditionType.NEUMANN
                    else None
                ),
                top_psi=(
                    self.top_bc.get_psi(t=time)
                    if self.top_bc.boundary_type == BoundaryConditionType.ROBIN
                    else None
                ),
                bottom_psi=(
                    self.bottom_bc.get_psi(t=time)
                    if self.bottom_bc.boundary_type == BoundaryConditionType.ROBIN
                    else None
                ),
                top_phi=(
                    self.top_bc.get_phi(t=time)
                    if self.top_bc.boundary_type == BoundaryConditionType.ROBIN
                    else None
                ),
                bottom_phi=(
                    self.bottom_bc.get_phi(t=time)
                    if self.bottom_bc.boundary_type == BoundaryConditionType.ROBIN
                    else None
                ),
            )
            self._iter_u = np.copy(self._new_u)

        return self._new_u
