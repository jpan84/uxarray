import holoviews as hv
import matplotlib as mpl

import cartopy.crs as ccrs


def _correct_central_longitude(node_lon, node_lat, projection):
    """Shifts the central longitude of an unstructured grid, which moves the
    antimeridian when visualizing, which is used when projections have a
    central longitude other than 0.0."""
    if projection:
        central_longitude = projection.proj4_params["lon_0"]
        if central_longitude != 0.0:
            _source_projection = ccrs.PlateCarree(central_longitude=0.0)
            _destination_projection = ccrs.PlateCarree(
                central_longitude=projection.proj4_params["lon_0"]
            )

            lonlat_proj = _destination_projection.transform_points(
                _source_projection, node_lon, node_lat
            )

            node_lon = lonlat_proj[:, 0]
    else:
        central_longitude = 0.0

    return node_lon, node_lat, central_longitude


class HoloviewsBackend:
    """Utility class to compare and set a HoloViews plotting backend for
    visualization."""

    def __init__(self):
        self.matplotlib_backend = mpl.get_backend()

    def assign(self, backend: str):
        """Assigns a backend for use with HoloViews visualization.

        Parameters
        ----------
        backend : str
            Plotting backend to use, one of 'matplotlib', 'bokeh'
        """

        if backend not in ["bokeh", "matplotlib", None]:
            raise ValueError(
                f"Unsupported backend. Expected one of ['bokeh', 'matplotlib'], but received {backend}"
            )
        if backend is not None and backend != hv.Store.current_backend:
            # only call hv.extension if it needs to be changed
            hv.extension(backend)

    def reset_mpl_backend(self):
        """Resets the default backend for the ``matplotlib`` module."""
        mpl.use(self.matplotlib_backend)


# global reference to holoviews backend utility class
backend = HoloviewsBackend()
