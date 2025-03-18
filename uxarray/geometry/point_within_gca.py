import numpy as np

from numba import njit

from uxarray.constants import MACHINE_EPSILON

from .point import Point, PointArray

from .gca import GreatCircleArc, GreatCircleArcArray


def point_within_gca(
    point: Point | PointArray, gca: GreatCircleArc | GreatCircleArcArray
):
    if isinstance(point, Point) and isinstance(gca, GreatCircleArc):
        return _point_within_gca(point.data, gca.start_point.data, gca.end_point.data)

    elif isinstance(point, Point) and isinstance(gca, GreatCircleArcArray):
        raise NotImplementedError

    elif isinstance(point, PointArray) and isinstance(gca, GreatCircleArc):
        raise NotImplementedError

    elif isinstance(point, PointArray) and isinstance(gca, GreatCircleArcArray):
        raise NotImplementedError

    else:
        raise TypeError("TODO")


@njit(cache=True)
def _point_within_gca(pt_xyz, gca_a_xyz, gca_b_xyz):
    """
    Check if a point lies on a given Great Circle Arc (GCA) interval, considering the smaller arc of the circle.
    Handles the anti-meridian case as well.

    Parameters
    ----------
    pt_xyz : numpy.ndarray
        Cartesian coordinates of the point.
    gca_a_xyz : numpy.ndarray
        Cartesian coordinates of the first endpoint of the Great Circle Arc.
    gca_b_xyz : numpy.ndarray
        Cartesian coordinates of the second endpoint of the Great Circle Arc.

    Returns
    -------
    bool
        True if the point lies within the specified GCA interval, False otherwise.

    Raises
    ------
    ValueError
        If the input GCA spans exactly 180 degrees (π radians), as this GCA can have multiple planes.
        In such cases, consider breaking the GCA into two separate arcs.

    Notes
    -----
    - The function ensures that the point lies on the same plane as the GCA before performing interval checks.
    - It assumes the input represents the smaller arc of the Great Circle.
    - The `_angle_of_2_vectors` and `_xyz_to_lonlat_rad_scalar` functions are used for calculations.
    """
    # 1. Check if the input GCA spans exactly 180 degrees
    normal = np.cross(gca_a_xyz, gca_b_xyz)

    # Calculate the angle using arctangent of cross and dot products
    angle_u_v_rad = np.arctan2(np.linalg.norm(normal), np.dot(gca_a_xyz, gca_b_xyz))

    if np.allclose(angle_u_v_rad, np.pi, rtol=0.0, atol=MACHINE_EPSILON):
        raise ValueError(
            "The input Great Circle Arc spans exactly 180 degrees, which can correspond to multiple planes. "
            "Consider breaking the Great Circle Arc into two smaller arcs."
        )

    # 2. Verify if the point lies on the plane of the GCA
    if not np.allclose(
        np.dot(normal, pt_xyz), 0, rtol=MACHINE_EPSILON, atol=MACHINE_EPSILON
    ):
        return False

    # 3. Check if the point lies within the Great Circle Arc interval
    pt_a = gca_a_xyz - pt_xyz
    pt_b = gca_b_xyz - pt_xyz

    # Use the dot product to determine the sign of the angle between pt_a and pt_b
    cos_theta = np.dot(pt_a, pt_b)

    # Return True if the point lies within the interval (smaller arc)
    if cos_theta < 0:
        return True
    elif np.isclose(cos_theta, 0.0, atol=MACHINE_EPSILON):
        # set error tolerance to 0.0
        return True
    else:
        return False
