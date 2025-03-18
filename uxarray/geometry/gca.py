from __future__ import annotations

import numpy as np

from numpy.typing import NDArray

from .point import Point, PointArray


class GreatCircleArc:
    def __init__(
        self, start_point: Point, end_point: Point, constant_latitude: bool = False
    ):
        self._start_point = start_point
        self._end_point = end_point
        self._constant_latitude = constant_latitude

    # ==================================================================================================================

    @property
    def constant_latitude(self) -> bool:
        return self._constant_latitude

    @property
    def start_point(self) -> Point:
        return self._start_point

    @property
    def end_point(self) -> Point:
        return self._end_point

    @property
    def bounds(self):
        return -1

    @property
    def length(self):
        return -1

    def contains(self, point: Point) -> bool:
        from .point_within_gca import point_within_gca

        return point_within_gca(point, self)

    # ==================================================================================================================

    def intersects(self, gca: GreatCircleArc) -> bool:
        pass


class GreatCircleArcArray:
    """ """

    def __init__(
        self,
        start_points: PointArray,
        end_points: PointArray,
        constant_latitude: NDArray[np.bool] | bool = False,
    ):
        self._start_points = start_points
        self._end_points = end_points
        self._constant_latitude = constant_latitude

    def __getitem__(self, index):
        start_point = self.start_points[index]
        end_point = self.end_points[index]
        return GreatCircleArc(start_point, end_point, self._constant_latitude)

    # ==================================================================================================================

    @property
    def start_points(self):
        return self._start_points

    @property
    def end_points(self):
        return self._end_points

    # ==================================================================================================================

    @classmethod
    def from_edge_node_connectivity(
        cls, points: PointArray, edge_node_connectivity: NDArray[np.int64]
    ):
        arc_xyz = points.data[edge_node_connectivity].astype(np.float64)

        return cls(
            PointArray(arc_xyz[:, 0, 0], arc_xyz[:, 0, 1], arc_xyz[:, 0, 2]),
            PointArray(arc_xyz[:, 1, 0], arc_xyz[:, 1, 1], arc_xyz[:, 1, 2]),
        )

    # def intersects(self, gca: GreatCircleArc) -> NDArray[np.bool]:
    #     pass
