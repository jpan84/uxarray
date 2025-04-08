import numpy as np

import cartopy.crs as ccrs

from uxarray.plot.polygons import _get_polygons

from matplotlib.collections import LineCollection


# ======================================================================================================================
# Matplotlib LineCollection
# ======================================================================================================================
def _grid_to_matplotlib_linecollection(
    grid, periodic_elements, projection=None, **kwargs
):
    """Constructs and returns a ``matplotlib.collections.LineCollection``"""

    if periodic_elements == "split" and projection is not None:
        apply_projection = False
    else:
        apply_projection = True

    # do not explicitly project when splitting elements
    polygons, central_longitude, _, _ = _get_polygons(
        grid, periodic_elements, projection, apply_projection
    )

    # Convert polygons to line segments for the LineCollection
    lines = []
    for pol in polygons:
        boundary = pol.boundary
        if boundary.geom_type == "MultiLineString":
            for line in list(boundary.geoms):
                lines.append(np.array(line.coords))
        else:
            lines.append(np.array(boundary.coords))

    if "transform" not in kwargs:
        # Set default transform if one is not provided not provided
        if projection is None or not apply_projection:
            kwargs["transform"] = ccrs.PlateCarree(central_longitude=central_longitude)
        else:
            kwargs["transform"] = projection

    return LineCollection(lines, **kwargs)
