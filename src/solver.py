from abc import ABC

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
