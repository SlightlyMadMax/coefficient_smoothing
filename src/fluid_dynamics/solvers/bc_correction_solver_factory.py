import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from scipy import sparse


from src.convective_operators import (
    ConvectiveTermForm,
    StreamFunctionBasedConvectiveOperator,
)
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.solvers.stream_function_solvers import (
    StreamFunctionSolverRegistry,
    StreamFunctionSolverName,
)
from src.fluid_dynamics.solvers.vorticity_solvers import (
    VorticitySolverRegistry,
    VorticitySolverName,
)
from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import PenaltyTermForm
from src.fluid_dynamics.utils import calculate_vorticity_from_sf
from src.parameters.config import ExperimentConfig


class BCCorrectionNVSolver:
    def __init__(
        self,
        cfg: ExperimentConfig,
        sf_bcs: BoundaryConditions,
        sf_max_iters: int = 10000,
        sf_tolerance: float = 1e-6,
        convective_term_form: ConvectiveTermForm = ConvectiveTermForm.DIVERGENT_CENTRAL,
        penalty_term_form: PenaltyTermForm = PenaltyTermForm.LINEAR,
        vorticity_solver_name: VorticitySolverName = VorticitySolverName.PEACEMAN_RACHFORD,
        stream_function_solver_name: StreamFunctionSolverName = StreamFunctionSolverName.AMG,
        vorticity_bc_order: int = 1,
        sf_solver_kwargs: dict | None = None,
        penalty_time_scheme: str = "cn",
        penalty_ramp: float = 0.0,
        penalty_ramp_mode: str = "linear",
    ):
        self.cfg = cfg
        self.vorticity_bc_order = vorticity_bc_order
        if penalty_time_scheme not in ("cn", "implicit", "dr"):
            raise ValueError(
                f"penalty_time_scheme must be 'cn', 'dr' or 'implicit', got "
                f"{penalty_time_scheme!r}"
            )
        self.penalty_time_scheme = penalty_time_scheme
        _, _, tau = cfg.scaled_grid_steps
        self._sigma = 1.0 if penalty_time_scheme == "dr" else 0.5
        self._sigma_tau = self._sigma * tau
        self.convective_operator = StreamFunctionBasedConvectiveOperator(
            cfg=cfg, form=convective_term_form
        )
        n_y, n_x = cfg.geometry.n_y, cfg.geometry.n_x

        vorticity_solver_class = VorticitySolverRegistry.get_solver_class(
            solver_name=vorticity_solver_name
        )
        stream_function_solver_class = StreamFunctionSolverRegistry.get_solver_class(
            solver_name=stream_function_solver_name
        )

        self.vorticity_solver = vorticity_solver_class(
            cfg=cfg,
            convective_operator=self.convective_operator,
            bc_order=vorticity_bc_order,
            penalty_term_form=penalty_term_form,
        )
        self.stream_function_solver = stream_function_solver_class(
            cfg=cfg,
            bcs=sf_bcs,
            max_iters=sf_max_iters,
            stopping_criteria=sf_tolerance,
            **(sf_solver_kwargs or {}),
        )

        self.vorticity_solver.penalty_in_predictor = penalty_time_scheme != "implicit"
        self.vorticity_solver.penalty_ramp = penalty_ramp
        self.vorticity_solver.penalty_ramp_mode = penalty_ramp_mode

        self._vorticity: NDArray[np.float64] = np.empty((n_y, n_x))
        self._stream_function: NDArray[np.float64] = np.empty((n_y, n_x))
        self.rho = self.calculate_rho_first_order()

    def calculate_rho_first_order(self):
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, _ = self.cfg.scaled_grid_steps

        inv_dx4 = dx**-4
        inv_dy4 = dy**-4

        rho = np.zeros((n_y, n_x))

        # edges (excluding corners)
        rho[2 : n_y - 2, 1] = 2.0 * inv_dx4
        rho[2 : n_y - 2, n_x - 2] = 2.0 * inv_dx4
        rho[1, 2 : n_x - 2] = 2.0 * inv_dy4
        rho[n_y - 2, 2 : n_x - 2] = 2.0 * inv_dy4

        # corners
        val = 2.0 * (inv_dx4 + inv_dy4)
        rho[1, 1] = rho[1, n_x - 2] = rho[n_y - 2, 1] = rho[n_y - 2, n_x - 2] = val

        return rho

    def apply_rho_to_psi_second_order(self, psi: np.ndarray) -> np.ndarray:
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, _ = self.cfg.scaled_grid_steps

        res = np.zeros_like(psi)

        j_slice = slice(2, n_y - 2)
        i_left = 1
        res[j_slice, i_left] = 4.0 * psi[j_slice, i_left] / dx**4 - psi[
            j_slice, i_left + 1
        ] / (2.0 * dx**4)

        # right side: i = n_x - 2, neighbor is n_x - 3
        i_right = n_x - 2
        res[j_slice, i_right] = 4.0 * psi[j_slice, i_right] / dx**4 - psi[
            j_slice, i_right - 1
        ] / (2.0 * dx**4)

        # top side: j = 1, i = 2 .. n_x-3
        i_slice = slice(2, n_x - 2)
        j_top = 1
        res[j_top, i_slice] = 4.0 * psi[j_top, i_slice] / dy**4 - psi[
            j_top + 1, i_slice
        ] / (2.0 * dy**4)

        # bottom side: j = n_y - 2, neighbor is n_y - 3
        j_bot = n_y - 2
        res[j_bot, i_slice] = 4.0 * psi[j_bot, i_slice] / dy**4 - psi[
            j_bot - 1, i_slice
        ] / (2.0 * dy**4)

        # corners: combine x- and y- contributions
        # top-left (j=1, i=1)
        res[j_top, i_left] = (
            4.0 * psi[j_top, i_left] / dx**4
            - psi[j_top, i_left + 1] / (2.0 * dx**4)
            + 4.0 * psi[j_top, i_left] / dy**4
            - psi[j_top + 1, i_left] / (2.0 * dy**4)
        )

        # top-right (j=1, i=n_x-2)
        res[j_top, i_right] = (
            4.0 * psi[j_top, i_right] / dx**4
            - psi[j_top, i_right - 1] / (2.0 * dx**4)
            + 4.0 * psi[j_top, i_right] / dy**4
            - psi[j_top + 1, i_right] / (2.0 * dy**4)
        )

        # bottom-left (j=n_y-2, i=1)
        res[j_bot, i_left] = (
            4.0 * psi[j_bot, i_left] / dx**4
            - psi[j_bot, i_left + 1] / (2.0 * dx**4)
            + 4.0 * psi[j_bot, i_left] / dy**4
            - psi[j_bot - 1, i_left] / (2.0 * dy**4)
        )

        # bottom-right (j=n_y-2, i=n_x-2)
        res[j_bot, i_right] = (
            4.0 * psi[j_bot, i_right] / dx**4
            - psi[j_bot, i_right - 1] / (2.0 * dx**4)
            + 4.0 * psi[j_bot, i_right] / dy**4
            - psi[j_bot - 1, i_right] / (2.0 * dy**4)
        )

        return res

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        delta: float,
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._vorticity[:, :] = w

        self._solve_vorticity(
            old_vorticity=self._vorticity,
            stream_function=sf,
            temperature=u,
            delta=delta,
            time=time,
        )
        self._solve_stream_function(
            sf_old=sf,
            vorticity=self._vorticity,
            time=time,
        )

        calculate_vorticity_from_sf(
            sf=self._stream_function,
            result=self._vorticity,
            cfg=self.cfg,
            bc_order=self.vorticity_bc_order,
        )
        return self._stream_function, self._vorticity

    def _solve_vorticity(
        self,
        old_vorticity: np.ndarray,
        stream_function: np.ndarray,
        temperature: np.ndarray,
        delta: float,
        time: float,
    ) -> None:
        self._vorticity[:, :] = self.vorticity_solver.solve(
            w=old_vorticity,
            sf=stream_function,
            u=temperature,
            delta=delta,
            time=time,
        )

    def _solve_stream_function(
        self,
        sf_old: np.ndarray,
        vorticity: np.ndarray,
        time: float,
    ) -> None:
        b = self._construct_rhs(
            vorticity=vorticity,
            sf_old=sf_old,
            px_half=self.vorticity_solver.px_pred,
            py_half=self.vorticity_solver.py_pred,
        )
        A = self._construct_matrix(
            px_half=self.vorticity_solver.px_half,
            py_half=self.vorticity_solver.py_half,
        )
        self._stream_function[:, :] = self.stream_function_solver.solve(
            initial_guess=sf_old,
            A=A,
            b_flat=b,
            time=time,
        )

    def _construct_rhs(
        self,
        vorticity: np.ndarray,
        sf_old: np.ndarray,
        px_half: np.ndarray,
        py_half: np.ndarray,
    ) -> np.ndarray:
        dx, dy, tau = self.cfg.scaled_grid_steps
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_re = 1.0 / self.cfg.reynolds_number

        # interior (i = 1..n_x-2, j = 1..n_y-2)
        psi = sf_old[1:-1, 1:-1]  # shape (n_y-2, n_x-2)
        w = vorticity[1:-1, 1:-1]
        if self.vorticity_bc_order == 1:
            r = self.rho[1:-1, 1:-1] * psi
        else:  # second order bc
            r = self.apply_rho_to_psi_second_order(sf_old)[1:-1, 1:-1]

        # X-direction: px_half has shape (n_y, n_x-1)
        # we need px_half[j, i] and px_half[j, i-1] for i=1..n_x-2, j=1..n_y-2
        px_i = px_half[1:-1, 1:]  # selects columns 1..(n_x-2) -> shape (n_y-2, n_x-2)
        px_im1 = px_half[
            1:-1, :-1
        ]  # selects columns 0..(n_x-3) -> shape (n_y-2, n_x-2)

        sf_x_fwd = sf_old[1:-1, 2:]  # sf[j, i+1]
        sf_x = psi  # sf[j, i]
        sf_x_bak = sf_old[1:-1, 0:-2]  # sf[j, i-1]

        term_x = px_i * (sf_x_fwd - sf_x) - px_im1 * (sf_x - sf_x_bak)

        # Y-direction: py_half has shape (n_y-1, n_x)
        # we need py_half[j, i] and py_half[j-1, i] for j=1..n_y-2, i=1..n_x-2
        py_j = py_half[1:, 1:-1]  # rows 1..(n_y-2), cols 1..(n_x-2) -> (n_y-2, n_x-2)
        py_jm1 = py_half[:-1, 1:-1]  # rows 0..(n_y-3), cols 1..(n_x-2)

        sf_y_fwd = sf_old[2:, 1:-1]  # sf[j+1, i]
        sf_y = psi  # sf[j, i]
        sf_y_bak = sf_old[0:-2, 1:-1]  # sf[j-1, i]

        term_y = py_j * (sf_y_fwd - sf_y) - py_jm1 * (sf_y - sf_y_bak)

        c_inner = -inv_dx2 * term_x - inv_dy2 * term_y

        b_int = -w - self._sigma_tau * (c_inner + inv_re * r)

        return b_int.ravel()

    def _init_matrix_structure(self) -> None:
        """
        Precompute everything about the stream-function matrix that never changes.

        Only the penalty term varies from one step to the next, and it enters through
        four neighbour coefficients. The sparsity pattern, the Laplacian part, the
        vorticity boundary-condition correction and the second-order boundary terms are
        all fixed, so they are assembled once here. `_construct_matrix` then only has to
        refresh the varying part and scatter it into the CSR value array, which avoids
        rebuilding the sparse structure on every time step.
        """
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        inv_re = 1.0 / self.cfg.reynolds_number

        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        if inner_n_x < 2 or inner_n_y < 2:
            raise ValueError(
                f"Grid {n_x}x{n_y} leaves fewer than 2 interior nodes per direction"
            )
        size = inner_n_x * inner_n_y

        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        tau_half = 0.5 * tau

        self._m_inner = (inner_n_y, inner_n_x)
        self._m_size = size
        self._m_tau_half = tau_half
        self._m_penalty_tau = (
            tau if self.penalty_time_scheme == "implicit" else self._sigma_tau
        )
        self._m_inv_dx2 = inv_dx2
        self._m_inv_dy2 = inv_dy2

        # Constants are kept separate rather than folded together so that the per-step
        # arithmetic can be performed in exactly the same order as a from-scratch
        # assembly. Folding them would reassociate the sums and shift the result by one
        # ULP, which is harmless but makes bit-for-bit comparison with earlier runs
        # impossible; the folding saved nothing measurable anyway.
        self._m_lam = -2.0 * inv_dx2 - 2.0 * inv_dy2
        self._m_rho_flat = (self._sigma_tau * inv_re * self.rho[1:-1, 1:-1]).ravel()

        # The last entry of every row block of the +/-1 diagonals must stay zero:
        # column inner_n_x-1 of row r is not a neighbour of column 0 of row r+1.
        side_mask = np.ones(size - 1, dtype=bool)
        side_mask[inner_n_x - 1 :: inner_n_x] = False
        self._m_side_mask = side_mask

        self._m_bc = None
        if self.vorticity_bc_order != 1:
            rows = np.arange(inner_n_y)
            cols = np.arange(inner_n_x)
            self._m_bc = {
                "left": rows * inner_n_x,
                "right": rows * inner_n_x + (inner_n_x - 1),
                "top": cols,
                "bottom": (inner_n_y - 1) * inner_n_x + cols,
                "diag_x": self._sigma_tau * inv_re * (2.0 / (dx**4)),
                "off_x": self._sigma_tau * inv_re * (1.0 / (2.0 * dx**4)),
                "diag_y": self._sigma_tau * inv_re * (2.0 / (dy**4)),
                "off_y": self._sigma_tau * inv_re * (1.0 / (2.0 * dy**4)),
            }

        # --- sparsity pattern and the diagonal -> CSR value permutation ----------
        lengths = [size, size - 1, size - 1, size - inner_n_x, size - inner_n_x]
        offsets = [0, -1, 1, -inner_n_x, inner_n_x]
        total = int(sum(lengths))

        # Tag every slot with its index in the concatenated-diagonal ordering, then read
        # the tags back in CSR order to obtain the gather permutation. Tags start at 1 so
        # that a real slot is never mistaken for a structural zero. The gaps between row
        # blocks of the +/-1 diagonals are tagged 0 on purpose: they carry no coupling
        # and are dropped here, which keeps the pattern identical to a from-scratch
        # assembly instead of storing explicit zeros.
        tags, start = [], 0
        for k, length in enumerate(lengths):
            tag = np.arange(start, start + length, dtype=np.float64) + 1.0
            if offsets[k] in (-1, 1):
                tag[~side_mask] = 0.0
            tags.append(tag)
            start += length
        tag_m = sparse.diags(tags, offsets, shape=(size, size), format="csr")

        n_gaps = int((~side_mask).sum())
        expected = total - 2 * n_gaps
        if tag_m.nnz != expected:
            raise RuntimeError(
                f"Sparsity pattern mismatch: {tag_m.nnz} stored vs {expected} expected"
            )

        self._m_order = tag_m.data.astype(np.int64) - 1
        self._m_slices = []
        start = 0
        for length in lengths:
            self._m_slices.append(slice(start, start + length))
            start += length
        self._m_concat = np.empty(total)

        tag_m.data[:] = 0.0
        self._m_csr = tag_m

    def _construct_matrix(self, px_half: np.ndarray, py_half: np.ndarray):
        """
        Refresh the stream-function matrix in place.

        The returned CSR object is reused between calls, so callers must not assume it
        stays constant after the next call. This is deliberate: the AMG solver aliases
        it as its fine-level operator, which is exactly the operator that has to follow
        the current penalty field.
        """
        if getattr(self, "_m_csr", None) is None:
            self._init_matrix_structure()

        inner_n_y, inner_n_x = self._m_inner
        tau_half = self._m_penalty_tau
        inv_dx2, inv_dy2 = self._m_inv_dx2, self._m_inv_dy2

        a_e = px_half[1:-1, 1:] * inv_dx2
        a_w = px_half[1:-1, :-1] * inv_dx2
        a_n = py_half[1:, 1:-1] * inv_dy2
        a_s = py_half[:-1, 1:-1] * inv_dy2

        # Same operation order as the reference assembly, so the values agree bit for
        # bit: Laplacian, then the penalty sum, then the vorticity-BC correction, then
        # the second-order boundary terms in the order left, right, top, bottom.
        main = np.full(self._m_size, self._m_lam)
        main -= (tau_half * (a_e + a_w + a_n + a_s)).ravel()
        main -= self._m_rho_flat

        side = np.zeros(self._m_size - 1)
        side[self._m_side_mask] = (inv_dx2 + tau_half * a_e[:, :-1]).ravel()
        ud = (inv_dy2 + tau_half * a_n[:-1, :]).ravel()

        bc = self._m_bc
        if bc is None:
            lower_side = upper_side = side
            lower_ud = upper_ud = ud
        else:
            main[bc["left"]] -= bc["diag_x"]
            main[bc["right"]] -= bc["diag_x"]
            main[bc["top"]] -= bc["diag_y"]
            main[bc["bottom"]] -= bc["diag_y"]

            lower_side, upper_side = side.copy(), side.copy()
            lower_ud, upper_ud = ud.copy(), ud.copy()
            upper_side[bc["left"]] += bc["off_x"]
            lower_side[bc["right"] - 1] += bc["off_x"]
            upper_ud[bc["top"]] += bc["off_y"]
            lower_ud[bc["bottom"] - inner_n_x] += bc["off_y"]

        concat, sl = self._m_concat, self._m_slices
        concat[sl[0]] = main
        concat[sl[1]] = lower_side
        concat[sl[2]] = upper_side
        concat[sl[3]] = lower_ud
        concat[sl[4]] = upper_ud

        np.take(concat, self._m_order, out=self._m_csr.data)
        return self._m_csr

    def _construct_matrix_reference(self, px_half: np.ndarray, py_half: np.ndarray):
        """Original from-scratch assembly, kept as the reference for verification."""
        geometry: DomainGeometry = self.cfg.geometry
        n_y, n_x = geometry.n_y, geometry.n_x
        dx, dy, tau = self.cfg.scaled_grid_steps
        inv_re = 1.0 / self.cfg.reynolds_number

        inner_n_y, inner_n_x = n_y - 2, n_x - 2
        size = inner_n_x * inner_n_y

        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        tau_half = 0.5 * tau

        rho_inner = self.rho[1:-1, 1:-1]
        rho_term_flat = (tau_half * inv_re * rho_inner).ravel()

        p_e = px_half[1:-1, 1:]
        p_w = px_half[1:-1, :-1]
        p_n = py_half[1:, 1:-1]
        p_s = py_half[:-1, 1:-1]

        a_e = p_e * inv_dx2
        a_w = p_w * inv_dx2
        a_n = p_n * inv_dy2
        a_s = p_s * inv_dy2

        sum_neighbors = a_e + a_w + a_n + a_s

        lam_main = -2.0 * inv_dx2 - 2.0 * inv_dy2
        main_diag = np.full(size, lam_main)

        main_diag -= (tau_half * sum_neighbors).ravel()

        main_diag -= rho_term_flat

        base_side = np.zeros(size - 1)
        base_updown = np.zeros(size - inner_n_x)

        base = 0
        for r in range(inner_n_y):
            if inner_n_x > 1:
                idx = base + np.arange(inner_n_x - 1)
                base_side[idx] = inv_dx2 + tau_half * a_e[r, :-1]
            base += inner_n_x

        base = 0
        for r in range(inner_n_y - 1):
            idx = base + np.arange(inner_n_x)
            base_updown[idx] = inv_dy2 + tau_half * a_n[r, :]
            base += inner_n_x

        if self.vorticity_bc_order == 1:
            diagonals = [main_diag, base_side, base_side, base_updown, base_updown]
        else:  # second order bc
            lower_side = base_side.copy()  # offset -1
            upper_side = base_side.copy()  # offset +1
            lower_ud = base_updown.copy()  # offset -inner_n_x
            upper_ud = base_updown.copy()  # offset +inner_n_x

            delta_diag_x = tau_half * inv_re * (2.0 / (dx**4))
            delta_off_x = tau_half * inv_re * (1.0 / (2.0 * dx**4))

            delta_diag_y = tau_half * inv_re * (2.0 / (dy**4))
            delta_off_y = tau_half * inv_re * (1.0 / (2.0 * dy**4))

            def idx_from_rc(r, c):
                return r * inner_n_x + c

            # Left boundary (inner c == 0): change diag and add coupling to east (c==1)
            for r in range(0, inner_n_y):
                center = idx_from_rc(r, 0)
                main_diag[center] -= delta_diag_x
                upper_side[center] += delta_off_x

            # Right boundary (inner c == inner_n_x-1): coupling to west
            for r in range(0, inner_n_y):
                center = idx_from_rc(r, inner_n_x - 1)
                main_diag[center] -= delta_diag_x
                # coupling center -> west is stored in lower_side at index (center-1)
                if center - 1 >= 0:
                    lower_side[center - 1] += delta_off_x

            # Top boundary (inner r == 0): change diag and add coupling to south (r==1)
            for c in range(0, inner_n_x):
                center = idx_from_rc(0, c)
                main_diag[center] -= delta_diag_y
                # coupling center -> south is an *upper* updown entry at position `center`
                upper_ud[center] += delta_off_y

            # Bottom boundary (inner r == inner_n_y-1): coupling to north
            for c in range(0, inner_n_x):
                center = idx_from_rc(inner_n_y - 1, c)
                main_diag[center] -= delta_diag_y
                # coupling center -> north is stored in lower_ud at index (center - inner_n_x)
                pos = center - inner_n_x
                if pos >= 0:
                    lower_ud[pos] += delta_off_y

            diagonals = [main_diag, lower_side, upper_side, lower_ud, upper_ud]

        offsets = [0, -1, 1, -inner_n_x, inner_n_x]

        m = sparse.diags(diagonals, offsets, shape=(size, size), format="csr")
        return m
