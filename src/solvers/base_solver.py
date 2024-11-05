from abc import ABC, abstractmethod

from numpy.typing import NDArray


class ISolver(ABC):
    @abstractmethod
    def solve(
        self,
        u: NDArray,
        time: float = 0.0,
    ) -> NDArray: ...
