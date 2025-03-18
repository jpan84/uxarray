from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Union

from .point import Point, PointArray


class Arc:
    def __init__(
        self,
        start_point: Point,
        end_point: Point,
        constant_latitude: bool = False,
    ):
        self._start_point = start_point
        self._end_point = end_point
        self._constant_latitude = constant_latitude

    @property
    def constant_latitude(self) -> bool:
        """Indicates whether the arc is along a line of constant latitude.
        If False, the arc is treated as a great circle arc."""
        return self._constant_latitude

    @property
    def start_point(self) -> Point:
        return self._start_point

    @property
    def end_point(self) -> Point:
        return self._end_point

    @property
    def bounds(self):
        return -1  # Placeholder implementation

    @property
    def length(self):
        return -1  # Placeholder implementation

    def reverse(self) -> Arc:
        """Return a new Arc with start/end swapped."""
        return Arc(self.end_point, self.start_point, self.constant_latitude)

    def contains(self, point: Point) -> bool:
        # Example logic: for a great circle arc
        if not self.constant_latitude:
            from .point_within_gca import point_within_gca

            return point_within_gca(point, self)
        else:
            # TODO:
            return False

    def intersects(self, arc: Arc) -> bool:
        # Example logic
        if arc.constant_latitude:
            # TODO
            return True
        else:
            return True

    def intersection(self, arc: Arc):
        # Return the intersection point(s) or None
        pass


class ArcArray:
    """Stores many arcs defined by arrays of start and end points."""

    def __init__(
        self,
        start_points: PointArray,
        end_points: PointArray,
        constant_latitude: Union[bool, NDArray[np.bool_]] = False,
    ):
        """
        Parameters
        ----------
        start_points : PointArray
            The Nx3 array of starting points for each arc.
        end_points : PointArray
            The Nx3 array of ending points for each arc.
        constant_latitude : bool or NDArray of bool, optional
            If a single bool, applies to all arcs. If an array, each arc can be flagged independently.
        """
        self._start_points = start_points
        self._end_points = end_points
        self._constant_latitude = constant_latitude

    @classmethod
    def from_edge_node_connectivity(
        cls, points: PointArray, edge_node_connectivity: NDArray[np.int64]
    ) -> ArcArray:
        """Create an ArcArray from a PointArray and edge-node connectivity array.

        Parameters
        ----------
        points : PointArray
            The points referenced by the connectivity array.
        edge_node_connectivity : NDArray[np.int64]
            Array of shape (N,2) where each row contains indices into the points array
            that define the start and end points of an arc.

        Returns
        -------
        ArcArray
            An array of arcs defined by the connectivity.
        """
        arc_xyz = points.data[edge_node_connectivity]  # shape = (N,2,3)

        # Make two separate PointArrays for start/end:
        start_points = PointArray(arc_xyz[:, 0, 0], arc_xyz[:, 0, 1], arc_xyz[:, 0, 2])
        end_points = PointArray(arc_xyz[:, 1, 0], arc_xyz[:, 1, 1], arc_xyz[:, 1, 2])

        return cls(start_points, end_points)

    def __getitem__(self, index: int) -> Arc:
        """Returns an Arc view into this ArcArray at the specified index."""
        # Create Points that view into our PointArrays
        start_point = self._start_points[index]
        end_point = self._end_points[index]

        # Determine constant_latitude for this arc
        if isinstance(self._constant_latitude, bool):
            const_lat = self._constant_latitude
        else:
            const_lat = self._constant_latitude[index]

        # Create a new Arc with these view Points
        return Arc(start_point, end_point, const_lat)

    @property
    def start_points(self) -> PointArray:
        """Get the array of all start points."""
        return self._start_points

    @property
    def end_points(self) -> PointArray:
        """Get the array of all end points."""
        return self._end_points

    @property
    def constant_latitude(self) -> Union[bool, NDArray[np.bool_]]:
        """Get the constant_latitude flag(s) for all arcs."""
        return self._constant_latitude

    def contains(self, point: Point) -> NDArray[np.bool_]:
        """Check if each arc contains the given point.

        Returns a boolean array where each element indicates if the
        corresponding arc contains the point.
        """
        # Placeholder implementation
        return np.zeros(len(self._start_points.data), dtype=bool)

    def intersects(self, arc: Arc) -> NDArray[np.bool_]:
        """Check if each arc intersects with the given arc.

        Returns a boolean array where each element indicates if the
        corresponding arc intersects with the given arc.
        """
        # Placeholder implementation
        return np.zeros(len(self._start_points.data), dtype=bool)
