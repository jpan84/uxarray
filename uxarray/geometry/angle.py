from numba import njit
import numpy as np


@njit(cache=True)
def _angle_of_2_vectors(u: np.ndarray, v: np.ndarray):
    """
    Calculate the angle between two 3D vectors `u` and `v` on the unit sphere in radians.

    This function computes the angle between two vectors originating from the center of a unit sphere.
    The result is returned in the range [0, 2π]. It can be used to calculate the span of a great circle arc (GCA).

    Parameters
    ----------
    u : numpy.ndarray
        The first 3D vector (float), originating from the center of the unit sphere.
    v : numpy.ndarray
        The second 3D vector (float), originating from the center of the unit sphere.

    Returns
    -------
    float
        The angle between `u` and `v` in radians, in the range [0, 2π].

    Notes
    -----
    - The direction of the angle (clockwise or counter-clockwise) is determined using the cross product of `u` and `v`.
    - Special cases such as vectors aligned along the same longitude are handled explicitly.
    """
    # Compute the cross product to determine the direction of the normal
    normal = np.cross(u, v)

    # Calculate the angle using arctangent of cross and dot products
    angle_u_v_rad = np.arctan2(np.linalg.norm(normal), np.dot(u, v))

    # Determine the direction of the angle
    normal_z = np.dot(normal, np.array([0.0, 0.0, 1.0]))
    if normal_z > 0:
        # Counterclockwise direction
        return angle_u_v_rad
    elif normal_z == 0:
        # Handle collinear vectors (same longitude)
        if u[2] > v[2]:
            return angle_u_v_rad
        elif u[2] < v[2]:
            return 2 * np.pi - angle_u_v_rad
        else:
            return 0.0  # u == v
    else:
        # Clockwise direction
        return 2 * np.pi - angle_u_v_rad
