from __future__ import annotations

import numpy as np

from numpy.typing import NDArray
from uxarray import Grid

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gca import GreatCircleArc


class Point:
    def __init__(
        self,
        x: np.float64 | float,
        y: np.float64 | float,
        z: np.float64 | float,
        normalized: bool = False,
    ):
        self.data = np.array([x, y, z], dtype=np.float64)
        self._normalized = normalized

    def __eq__(self, other):
        pass

    def __ne__(self, other):
        pass

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    # ==================================================================================================================

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def z(self):
        return self.data[2]

    # ==================================================================================================================

    def normalize(self):
        pass

    def within(self, gca: GreatCircleArc):
        from .point_within_gca import point_within_gca

        return point_within_gca(self, gca)


class PointArray:
    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        normalized: bool = False,
        grid_mapping: str = None,
    ):
        # TODO: Add checks for the dimension size

        self.data = np.stack([x, y, z], axis=1, dtype=np.float64)
        self._normalized = normalized
        self._grid_mapping = grid_mapping

    def __getitem__(self, index):
        x, y, z = self.data[index]
        return Point(x, y, z, normalized=self._normalized)

    @classmethod
    def from_grid(cls, grid: Grid, element: str = "corner nodes"):
        if element == "corner nodes":
            return cls(grid.node_x.values, grid.node_y.values, grid.node_z.values)
        elif element == "face centers":
            return cls(grid.face_x.values, grid.face_y.values, grid.face_z.values)
        elif element == "edge centers":
            return cls(grid.edge_x.values, grid.edge_y.values, grid.edge_z.values)
        else:
            raise ValueError("TODO")

    @property
    def normalized(self):
        return self._normalized

    @property
    def grid_mapping(self):
        return self._grid_mapping

    @property
    def x(self):
        return self.data[:, 0]

    @property
    def y(self):
        return self.data[:, 1]

    @property
    def z(self):
        return self.data[:, 2]

    def normalize(self):
        pass
