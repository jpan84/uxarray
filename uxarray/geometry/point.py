from __future__ import annotations

import numpy as np

from numpy.typing import NDArray

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import Optional


class Point:
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        normalized: bool = False,
        parent_data: NDArray[np.float64] | None = None,
        idx: Optional[int] = None,
    ):
        if parent_data is not None:
            # External view mode
            if idx is None:
                raise ValueError("Must provide idx when parent_data is provided")
            self._parent_data = parent_data
            self._idx = idx
            self._owns_data = False
        else:
            # Create our own data array to view into
            self._parent_data = np.array([[x, y, z]], dtype=np.float64)  # Shape (1,3)
            self._idx = 0
            self._owns_data = True

        self._normalized = normalized

    @property
    def x(self) -> float:
        return self._parent_data[self._idx, 0]

    @property
    def y(self) -> float:
        return self._parent_data[self._idx, 1]

    @property
    def z(self) -> float:
        return self._parent_data[self._idx, 2]

    @property
    def normalized(self) -> bool:
        return self._normalized

    @property
    def data(self) -> NDArray[np.float64]:
        return self._parent_data[self._idx, :]

    def normalize(self) -> None:
        pass


class PointArray:
    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        normalized: bool = False,
    ):
        self.data = np.stack([x, y, z], axis=1).astype(np.float64)
        self._normalized = normalized

    def __getitem__(self, idx: int) -> Point:
        return Point(normalized=self._normalized, parent_data=self.data, idx=idx)

    @property
    def x(self) -> NDArray[np.float64]:
        return self.data[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        return self.data[:, 1]

    @property
    def z(self) -> NDArray[np.float64]:
        return self.data[:, 2]

    def normalize(self) -> None:
        pass
