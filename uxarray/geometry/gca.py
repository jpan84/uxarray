# import numpy as np
#
# from .point import Point, PointArray
#
#
# class GreatCircleArc:
#     def __init__(self, a: Point, b: Point, constant_latitude: bool = False):
#         self._a = a
#         self._b = b
#         self._constant_latitude = constant_latitude
#
#     @property
#     def bounds(self):
#         return -1
#
#
# class GreatCircleArcArray:
#     """ """
#
#     def __init__(
#         self,
#         start_points: PointArray,
#         end_points: PointArray,
#         constant_latitude: np.ndarray | bool = False,
#     ):
#         self._start_points = start_points
#         self._end_points = end_points
#         self._constant_latitude = constant_latitude
#
#     @classmethod
#     def from_edge_node_connectivity(
#         cls, points: PointArray, edge_node_connectivity: np.ndarray[np.int64, 2]
#     ):
#         # TODO: Align arrays in memory
#         arc_x = points._data[0][edge_node_connectivity].astype(np.float64)
#         arc_y = points._data[1][edge_node_connectivity].astype(np.float64)
#         arc_z = points._data[2][edge_node_connectivity].astype(np.float64)
#
#         aligned_data = np.column_stack((arc_x, arc_y, arc_z))
#
#         print(arc_x.shape)
#         return cls(
#             PointArray(arc_x[:, 0], arc_y[:, 0], arc_z[:, 0]),
#             PointArray(arc_x[:, 0], arc_y[:, 0], arc_z[:, 0]),
#         )
#
#     def intersects(self):
#         pass
