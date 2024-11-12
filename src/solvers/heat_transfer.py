import numba
import numpy as np
from enum import Enum
from numpy.typing import NDArray

from src.geometry import DomainGeometry
import src.parameters as cfg
from src.temperature.coefficient_smoothing.coefficients import c_smoothed, k_smoothed
from src.temperature.coefficient_smoothing.delta import get_max_delta
from src.temperature import boundary_conditions as bc
from src.utils import solve_tridiagonal


class HeatTransferSolver:
    class SolvingMethod(Enum):
        LOC_ONE_DIM = "Locally One-Dimensional Linearized Difference Scheme"
        ALT_DIR = "Classic Linearized Alternating Direction Scheme"

    def __init__(
        self,
        geometry: DomainGeometry,
        top_cond_type: int,
        right_cond_type: int,
        bottom_cond_type: int,
        left_cond_type: int,
        fixed_delta: bool = False,
    ):
        self.dx: float = geometry.dx
        self.dy: float = geometry.dy
        self.dt: float = geometry.dt
        self.n_y: int = geometry.n_y
        self.n_x: int = geometry.n_x
        self.fixed_delta = fixed_delta
        self.top_cond_type = top_cond_type
        self.right_cond_type = right_cond_type
        self.bottom_cond_type = bottom_cond_type
        self.left_cond_type = left_cond_type
        self._temp_u: NDArray[np.float64] = np.empty((self.n_y, self.n_x))
        self._new_u: NDArray[np.float64] = np.empty((self.n_y, self.n_x))

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_x(
        u: NDArray[np.float64],
        temp_u: NDArray[np.float64],
        dx: float,
        dt: float,
        right_cond_type: int,
        left_cond_type: int,
        delta: float,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        n_y, n_x = u.shape
        inv_dx2 = 1.0 / (dx * dx)

        a = np.zeros(n_x - 2)
        b = np.zeros(n_x - 2)
        c = np.zeros(n_x - 2)

        lbc = bc.get_left_bc_1(time, n_y)
        rbc = bc.get_right_bc_1(time, n_y)

        for j in range(1, n_y - 1):
            for i in range(1, n_x - 1):
                inv_c = 1.0 / c_smoothed(u[j, i], delta)

                a[i - 1] = (
                    -dt
                    * k_smoothed(0.5 * (u[j, i + 1] + u[j, i]), delta)
                    * inv_c
                    * inv_dx2
                )
                b[i - 1] = (
                    1.0
                    + dt
                    * (
                        k_smoothed(0.5 * (u[j, i + 1] + u[j, i]), delta)
                        + k_smoothed(0.5 * (u[j, i] + u[j, i - 1]), delta)
                    )
                    * inv_c
                    * inv_dx2
                )
                c[i - 1] = (
                    -dt
                    * k_smoothed(0.5 * (u[j, i] + u[j, i - 1]), delta)
                    * inv_c
                    * inv_dx2
                )

            temp_u[j, :] = solve_tridiagonal(
                a=a,
                b=b,
                c=c,
                f=u[j, :],
                left_type=left_cond_type,
                left_value=lbc[j],
                right_type=right_cond_type,
                right_value=rbc[j],
            )

        temp_u[0, :] = u[0, :]
        temp_u[n_y - 1, :] = u[n_y - 1, :]

        return temp_u

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_sweep_y(
        temp_u: NDArray[np.float64],
        new_u: NDArray[np.float64],
        dy: float,
        dt: float,
        top_cond_type: int,
        bottom_cond_type: int,
        delta: float,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        n_y, n_x = temp_u.shape
        inv_dy2 = 1.0 / (dy * dy)

        bbc = bc.get_bottom_bc_1(time, n_x)
        tbc = bc.get_top_bc_1(time, n_x)
        psi, ksi = bc.get_top_bc_3(time)

        a = np.zeros(n_y - 2)
        b = np.zeros(n_y - 2)
        c = np.zeros(n_y - 2)

        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                inv_c = 1.0 / c_smoothed(temp_u[j, i], delta)

                # Коэффициент при T_(i,j-1)^n
                a[j - 1] = (
                    -dt
                    * k_smoothed(0.5 * (temp_u[j + 1, i] + temp_u[j, i]), delta)
                    * inv_c
                    * inv_dy2
                )
                # Коэффициент при T_(i,j)^n
                b[j - 1] = (
                    1.0
                    + dt
                    * (
                        k_smoothed(0.5 * (temp_u[j + 1, i] + temp_u[j, i]), delta)
                        + k_smoothed(0.5 * (temp_u[j, i] + temp_u[j - 1, i]), delta)
                    )
                    * inv_c
                    * inv_dy2
                )
                # Коэффициент при T_(i,j+1)^n
                c[j - 1] = (
                    -dt
                    * k_smoothed(0.5 * (temp_u[j, i] + temp_u[j - 1, i]), delta)
                    * inv_c
                    * inv_dy2
                )

            new_u[:, i] = solve_tridiagonal(
                a,
                b,
                c,
                temp_u[:, i],
                left_type=bottom_cond_type,
                left_value=bbc[i],
                right_type=top_cond_type,
                right_value=tbc[i],
                right_psi=psi,
                right_ksi=ksi,
            )

        new_u[:, 0] = temp_u[:, 0]
        new_u[:, n_x - 1] = temp_u[:, n_x - 1]

        return new_u

    def solve(self, u: NDArray[np.float64], time: float = 0.0) -> NDArray[np.float64]:
        delta = cfg.delta if self.fixed_delta else get_max_delta(u)

        # Run the x-direction sweep
        self._temp_u = self._compute_sweep_x(
            u=u,
            temp_u=self._temp_u,
            dx=self.dx,
            dt=self.dt,
            right_cond_type=self.right_cond_type,
            left_cond_type=self.left_cond_type,
            delta=delta,
            time=time,
        )

        delta = cfg.delta if self.fixed_delta else get_max_delta(self._temp_u)

        # Run the y-direction sweep
        self._new_u = self._compute_sweep_y(
            temp_u=self._temp_u,
            new_u=self._new_u,
            dy=self.dy,
            dt=self.dt,
            top_cond_type=self.top_cond_type,
            bottom_cond_type=self.bottom_cond_type,
            delta=delta,
            time=time,
        )

        return self._new_u
