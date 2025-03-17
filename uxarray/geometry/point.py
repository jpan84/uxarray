import numpy as np


class Point:
    def __init__(self, x: np.float64, y: np.float64, z: np.float64):
        self._data = np.array([x, y, z], dtype=np.float64)

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

    def normalize(self):
        pass


class PointArray:
    pass
