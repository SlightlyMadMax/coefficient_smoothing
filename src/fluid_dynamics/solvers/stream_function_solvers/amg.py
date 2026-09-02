import numpy as np
import pyamg
from pyamg import MultilevelSolver
from scipy.sparse import csr_array

from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.core.solvers.base_solver import BaseSolver
from src.fluid_dynamics.solvers.stream_function_solvers.registry import (
    register_sf_solver,
    StreamFunctionSolverName,
)
from src.parameters.config import ExperimentConfig


@register_sf_solver(StreamFunctionSolverName.AMG)
class AlgebraicMultigridSolver(BaseSolver):
    """
    A solver for elliptic equations of the form Δu - c(x,y)u = -f(x,y) using Algebraic Multigrid.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        bcs: BoundaryConditions,
        max_iters: int = 10000,
        stopping_criteria: float = 1e-6,
        rebuild_every: int = 1,
    ):
        """
        Initialize the ConjugateGradientSolver with domain geometry and boundary conditions.

        :param cfg: The configuration of the experiment (domain geometry, material properties, etc.).
        :param bcs: An object containing boundary conditions.
        :param max_iters: Maximum number of iterations for convergence. Default is 10000.
        :param stopping_criteria: Convergence criteria for the solver. Default is 1e-6.
        :param rebuild_every: Rebuild the AMG hierarchy every N calls; in between, only
            the fine-level operator is refreshed (see `_get_hierarchy`). 1 rebuilds on
            every call, reproducing the original behaviour.
        """
        super().__init__(cfg=cfg, bcs=bcs)
        self.geometry: DomainGeometry = cfg.geometry
        self.max_iters = max_iters
        self.stopping_criteria = stopping_criteria
        self.rebuild_every = max(1, int(rebuild_every))

        # Cached AMG hierarchy, reused between rebuilds
        self._ml: MultilevelSolver | None = None
        self._calls_since_rebuild: int = 0

        # Pre-allocate some arrays that will be used in the calculations
        self._result: np.ndarray = np.empty((self.geometry.n_y, self.geometry.n_x))

    def _get_hierarchy(self, A: csr_array) -> MultilevelSolver:
        """
        Return an AMG hierarchy for `A`, rebuilding it only every `rebuild_every` calls.

        Between rebuilds the cached hierarchy is kept but its fine-level operator is
        replaced by the current `A`. The V-cycle then forms fine-level residuals with
        the true matrix, so its fixed point is the exact solution of `A x = b`; only the
        coarse-grid operators lag behind, which affects the convergence rate and not the
        answer. The stream-function matrix changes slowly — it varies only through the
        penalty term as the phase interface advances — so the rate penalty is negligible
        while the saving is large: building the hierarchy costs roughly twice as much as
        the solve itself.
        """
        stale = (
            self._ml is None
            or self._calls_since_rebuild >= self.rebuild_every
            or self._ml.levels[0].A.shape != A.shape
        )
        if stale:
            self._ml = pyamg.ruge_stuben_solver(A, strength="symmetric")
            self._calls_since_rebuild = 0
        else:
            self._ml.levels[0].A = A
        self._calls_since_rebuild += 1
        return self._ml

    def solve(
        self,
        A: csr_array,
        b_flat: np.ndarray,
        initial_guess: np.ndarray,
        time: float,
    ) -> np.ndarray:
        n_y, n_x = self.geometry.n_y, self.geometry.n_x
        inner_slice = (slice(1, -1), slice(1, -1))
        inner_n_y, inner_n_x = n_y - 2, n_x - 2

        self._result[:, 0] = self.bcs.left.get_value(t=time)
        self._result[:, -1] = self.bcs.right.get_value(t=time)
        self._result[0, :] = self.bcs.top.get_value(t=time)
        self._result[-1, :] = self.bcs.bottom.get_value(t=time)

        # Initial guess interior flattened
        x0 = initial_guess[inner_slice].ravel()

        ml: MultilevelSolver = self._get_hierarchy(A)
        solution_inner_flat = ml.solve(
            b_flat, x0=x0, tol=self.stopping_criteria, maxiter=self.max_iters
        )  # type: ignore

        self._result[inner_slice] = solution_inner_flat.reshape((inner_n_y, inner_n_x))  # type: ignore

        return self._result
