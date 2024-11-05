from abc import ABC, abstractmethod

from numpy import ndarray


class ISolver(ABC):
    @abstractmethod
    def solve(
        self,
        u: ndarray,
        time: float = 0.0,
    ) -> ndarray: ...
