import typing
import numpy as np
from enum import Enum
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.fluid_dynamics.parameters import FluidParameters
from src.geometry import DomainGeometry
from src.fluid_dynamics.schemes.douglas_rachford import DRNavierStokesScheme
from src.fluid_dynamics.schemes.explicit_central import ExpCentralNavierStokesScheme
from src.fluid_dynamics.schemes.explicit_upwind import ExpUpwindNavierStokesScheme
from src.fluid_dynamics.schemes.peaceman_rachford import PRNavierStokesScheme


class NavierStokesSchemes(Enum):
    EXPLICIT_CENTRAL = 1, "Explicit central differences"
    EXPLICIT_UPWIND = 2, "Explicit upwind"
    DOUGLAS_RACHFORD = 3, "Douglas-Rachford"
    PEACEMAN_RACHFORD = 4, "Peaceman-Rachford"


class NavierStokesSolver:
    def __init__(
        self,
        scheme: NavierStokesSchemes,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
        sf_max_iters: int = 50,
        sf_stopping_criteria: float = 1e-6,
        implicit_sf_max_iters: int = 5,
        implicit_sf_stopping_criteria: float = 1e-6,
    ):
        self.scheme = scheme
        SchemeClass: typing.Type = self.get_scheme_class()
        self.solver = SchemeClass(
            geometry=geometry,
            parameters=parameters,
            top_bc=top_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            left_bc=left_bc,
            sf_max_iters=sf_max_iters,
            sf_stopping_criteria=sf_stopping_criteria,
            implicit_sf_max_iters=implicit_sf_max_iters,
            implicit_sf_stopping_criteria=implicit_sf_stopping_criteria,
        )

    def get_scheme_class(self) -> typing.Type:
        if self.scheme == NavierStokesSchemes.EXPLICIT_CENTRAL:
            return ExpCentralNavierStokesScheme
        elif self.scheme == NavierStokesSchemes.EXPLICIT_UPWIND:
            return ExpUpwindNavierStokesScheme
        elif self.scheme == NavierStokesSchemes.DOUGLAS_RACHFORD:
            return DRNavierStokesScheme
        elif self.scheme == NavierStokesSchemes.PEACEMAN_RACHFORD:
            return PRNavierStokesScheme
        else:
            raise NotImplemented()

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ):
        return self.solver.solve(w, sf, u, time)
