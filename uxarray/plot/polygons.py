import numpy as np
import geopandas
import shapely
import spatialpandas
import antimeridian

import cartopy.crs as ccrs

from uxarray.plot.utils import _correct_central_longitude
from uxarray.plot.antimeridian import _build_antimeridian_face_indices
from uxarray.grid.connectivity import _pad_closed_face_nodes

from shapely import Polygon
from shapely import polygons as Polygons
from spatialpandas.geometry import MultiPolygonArray, PolygonArray

from matplotlib.collections import PolyCollection

# ======================================================================================================================
# GeoDataFrame
# ======================================================================================================================


def _grid_to_polygon_geodataframe(grid, periodic_elements, projection, project, engine):
    """Converts the faces of a ``Grid`` into a ``spatialpandas.GeoDataFrame``
    or ``geopandas.GeoDataFrame`` with a geometry column of polygons."""

    node_lon, node_lat, central_longitude = _correct_central_longitude(
        grid.node_lon.values, grid.node_lat.values, projection
    )
    polygon_shells = _build_polygon_shells(
        node_lon,
        node_lat,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
        projection=None,
        central_longitude=central_longitude,
    )

    if projection is not None and project:
        projected_polygon_shells = _build_polygon_shells(
            node_lon,
            node_lat,
            grid.face_node_connectivity.values,
            grid.n_face,
            grid.n_max_face_nodes,
            grid.n_nodes_per_face.values,
            projection=projection,
            central_longitude=central_longitude,
        )
    else:
        projected_polygon_shells = None

    antimeridian_face_indices = _build_antimeridian_face_indices(
        polygon_shells[:, :, 0]
    )

    non_nan_polygon_indices = None
    if projection is not None and project:
        shells_d = np.delete(
            projected_polygon_shells, antimeridian_face_indices, axis=0
        )

        # Check for NaN in each sub-array and invert the condition
        does_not_contain_nan = ~np.isnan(shells_d).any(axis=1)

        # Get the indices where NaN is NOT present
        non_nan_polygon_indices = np.where(does_not_contain_nan)[0]

    grid._gdf_cached_parameters["antimeridian_face_indices"] = antimeridian_face_indices

    if periodic_elements == "split":
        gdf = _build_geodataframe_with_antimeridian(
            polygon_shells,
            projected_polygon_shells,
            antimeridian_face_indices,
            engine=engine,
        )
    elif periodic_elements == "ignore":
        if engine == "geopandas":
            # create a geopandas.GeoDataFrame
            if projected_polygon_shells is not None:
                geometry = projected_polygon_shells
            else:
                geometry = polygon_shells

            gdf = geopandas.GeoDataFrame({"geometry": shapely.polygons(geometry)})
        else:
            # create a spatialpandas.GeoDataFrame
            if projected_polygon_shells is not None:
                geometry = PolygonArray.from_exterior_coords(projected_polygon_shells)
            else:
                geometry = PolygonArray.from_exterior_coords(polygon_shells)
            gdf = spatialpandas.GeoDataFrame({"geometry": geometry})

    else:
        gdf = _build_geodataframe_without_antimeridian(
            polygon_shells,
            projected_polygon_shells,
            antimeridian_face_indices,
            engine=engine,
        )
    return gdf, non_nan_polygon_indices


def _build_geodataframe_without_antimeridian(
    polygon_shells, projected_polygon_shells, antimeridian_face_indices, engine
):
    """Builds a ``spatialpandas.GeoDataFrame`` or
    ``geopandas.GeoDataFrame``excluding any faces that cross the
    antimeridian."""
    if projected_polygon_shells is not None:
        # use projected shells if a projection is applied
        shells_without_antimeridian = np.delete(
            projected_polygon_shells, antimeridian_face_indices, axis=0
        )
    else:
        shells_without_antimeridian = np.delete(
            polygon_shells, antimeridian_face_indices, axis=0
        )

    if engine == "geopandas":
        # create a geopandas.GeoDataFrame
        gdf = geopandas.GeoDataFrame(
            {"geometry": shapely.polygons(shells_without_antimeridian)}
        )
    else:
        # create a spatialpandas.GeoDataFrame
        geometry = PolygonArray.from_exterior_coords(shells_without_antimeridian)
        gdf = spatialpandas.GeoDataFrame({"geometry": geometry})

    return gdf


def _build_geodataframe_with_antimeridian(
    polygon_shells,
    projected_polygon_shells,
    antimeridian_face_indices,
    engine,
):
    """Builds a ``spatialpandas.GeoDataFrame`` or ``geopandas.GeoDataFrame``
    including any faces that cross the antimeridian."""
    polygons = _build_corrected_shapely_polygons(
        polygon_shells, projected_polygon_shells, antimeridian_face_indices
    )
    if engine == "geopandas":
        # Create a geopandas.GeoDataFrame
        gdf = geopandas.GeoDataFrame({"geometry": polygons})
    else:
        # Create a spatialpandas.GeoDataFrame
        geometry = MultiPolygonArray(polygons)
        gdf = spatialpandas.GeoDataFrame({"geometry": geometry})

    return gdf


# ======================================================================================================================
# Matplotlib PolyCollection
# ======================================================================================================================
def _grid_to_matplotlib_polycollection(
    grid, periodic_elements, projection=None, **kwargs
):
    """Constructs and returns a ``matplotlib.collections.PolyCollection``"""

    # Handle unsupported configuration: splitting periodic elements with projection
    if periodic_elements == "split" and projection is not None:
        raise ValueError(
            "Explicitly projecting lines is not supported. Please pass in your projection"
            "using the 'transform' parameter"
        )

    # Correct the central longitude and build polygon shells
    node_lon, node_lat, central_longitude = _correct_central_longitude(
        grid.node_lon.values, grid.node_lat.values, projection
    )

    if "transform" not in kwargs:
        if projection is None:
            kwargs["transform"] = ccrs.PlateCarree(central_longitude=central_longitude)
        else:
            kwargs["transform"] = projection

    polygon_shells = _build_polygon_shells(
        node_lon,
        node_lat,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
        projection=None,
        central_longitude=central_longitude,
    )

    # Projected polygon shells if a projection is specified
    if projection is not None:
        projected_polygon_shells = _build_polygon_shells(
            node_lon,
            node_lat,
            grid.face_node_connectivity.values,
            grid.n_face,
            grid.n_max_face_nodes,
            grid.n_nodes_per_face.values,
            projection=projection,
            central_longitude=central_longitude,
        )
    else:
        projected_polygon_shells = None

    # Determine indices of polygons crossing the antimeridian
    antimeridian_face_indices = _build_antimeridian_face_indices(
        polygon_shells[:, :, 0]
    )

    # Filter out NaN-containing polygons if projection is applied
    non_nan_polygon_indices = None
    if projected_polygon_shells is not None:
        # Delete polygons at the antimeridian
        shells_d = np.delete(
            projected_polygon_shells, antimeridian_face_indices, axis=0
        )

        # Get the indices of polygons that do not contain NaNs
        does_not_contain_nan = ~np.isnan(shells_d).any(axis=(1, 2))
        non_nan_polygon_indices = np.where(does_not_contain_nan)[0]

    grid._poly_collection_cached_parameters["non_nan_polygon_indices"] = (
        non_nan_polygon_indices
    )
    grid._poly_collection_cached_parameters["antimeridian_face_indices"] = (
        antimeridian_face_indices
    )

    # Select which shells to use: projected or original
    if projected_polygon_shells is not None:
        shells_to_use = projected_polygon_shells
    else:
        shells_to_use = polygon_shells

    # Handle periodic elements: exclude or split antimeridian polygons
    if periodic_elements == "exclude":
        # Remove antimeridian polygons and keep only non-NaN polygons if available
        shells_without_antimeridian = np.delete(
            shells_to_use, antimeridian_face_indices, axis=0
        )

        # Filter the shells using non-NaN indices
        if non_nan_polygon_indices is not None:
            shells_to_use = shells_without_antimeridian[non_nan_polygon_indices]
        else:
            shells_to_use = shells_without_antimeridian

        # Get the corrected indices of original faces
        corrected_to_original_faces = np.delete(
            np.arange(grid.n_face), antimeridian_face_indices, axis=0
        )

        # Create the PolyCollection using the cleaned shells
        return PolyCollection(shells_to_use, **kwargs), corrected_to_original_faces

    elif periodic_elements == "split":
        # Split polygons at the antimeridian
        (
            corrected_polygon_shells,
            corrected_to_original_faces,
        ) = _build_corrected_polygon_shells(polygon_shells)

        # Create PolyCollection using the corrected shells
        return PolyCollection(
            corrected_polygon_shells, **kwargs
        ), corrected_to_original_faces

    else:
        # Default: use original polygon shells
        return PolyCollection(polygon_shells, **kwargs), []


# ======================================================================================================================
# Shapely
# ======================================================================================================================
def _build_corrected_shapely_polygons(
    polygon_shells,
    projected_polygon_shells,
    antimeridian_face_indices,
):
    if projected_polygon_shells is not None:
        # use projected shells if a projection is applied
        shells = projected_polygon_shells
    else:
        shells = polygon_shells

    # list of shapely Polygons representing each face in our grid
    polygons = Polygons(shells)

    # construct antimeridian polygons
    antimeridian_polygons = Polygons(polygon_shells[antimeridian_face_indices])

    # correct each antimeridian polygon
    corrected_polygons = [antimeridian.fix_polygon(P) for P in antimeridian_polygons]

    # insert correct polygon back into original array
    for i in reversed(antimeridian_face_indices):
        polygons[i] = corrected_polygons.pop()

    return polygons


# ======================================================================================================================
# Polygon Coordinates (Consider moving to geometry module after refactor?)
# ======================================================================================================================


def _get_polygons(grid, periodic_elements, projection=None, apply_projection=True):
    # Correct the central longitude if projection is provided
    node_lon, node_lat, central_longitude = _correct_central_longitude(
        grid.node_lon.values, grid.node_lat.values, projection
    )

    # Build polygon shells without projection
    polygon_shells = _build_polygon_shells(
        node_lon,
        node_lat,
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_nodes,
        grid.n_nodes_per_face.values,
        projection=None,
        central_longitude=central_longitude,
    )

    # If projection is provided, create the projected polygon shells
    if projection and apply_projection:
        projected_polygon_shells = _build_polygon_shells(
            node_lon,
            node_lat,
            grid.face_node_connectivity.values,
            grid.n_face,
            grid.n_max_face_nodes,
            grid.n_nodes_per_face.values,
            projection=projection,
            central_longitude=central_longitude,
        )
    else:
        projected_polygon_shells = None

    # Determine indices of polygons crossing the antimeridian
    antimeridian_face_indices = _build_antimeridian_face_indices(
        polygon_shells[:, :, 0]
    )

    # Filter out NaN-containing polygons if projection is applied
    non_nan_polygon_indices = None
    if projected_polygon_shells is not None:
        # Delete polygons at the antimeridian
        shells_d = np.delete(
            projected_polygon_shells, antimeridian_face_indices, axis=0
        )

        # Get the indices of polygons that do not contain NaNs
        does_not_contain_nan = ~np.isnan(shells_d).any(axis=(1, 2))
        non_nan_polygon_indices = np.where(does_not_contain_nan)[0]

    # Determine which shells to use
    if projected_polygon_shells is not None:
        shells_to_use = projected_polygon_shells
    else:
        shells_to_use = polygon_shells

    # Exclude or handle periodic elements based on the input parameter
    if periodic_elements == "exclude":
        # Remove antimeridian polygons and keep only non-NaN polygons if available
        shells_without_antimeridian = np.delete(
            shells_to_use, antimeridian_face_indices, axis=0
        )

        # Filter the shells using non-NaN indices
        if non_nan_polygon_indices is not None:
            shells_to_use = shells_without_antimeridian[non_nan_polygon_indices]
        else:
            shells_to_use = shells_without_antimeridian

        polygons = _convert_shells_to_polygons(shells_to_use)
    elif periodic_elements == "split":
        # Correct for antimeridian crossings and split polygons as necessary
        polygons = _build_corrected_shapely_polygons(
            polygon_shells, projected_polygon_shells, antimeridian_face_indices
        )
    else:
        # Default: use original polygon shells
        polygons = _convert_shells_to_polygons(polygon_shells)

    return (
        polygons,
        central_longitude,
        antimeridian_face_indices,
        non_nan_polygon_indices,
    )


def _build_polygon_shells(
    node_lon,
    node_lat,
    face_node_connectivity,
    n_face,
    n_max_face_nodes,
    n_nodes_per_face,
    projection=None,
    central_longitude=0.0,
):
    """Builds an array of polygon shells, which can be used with Shapely to
    construct polygons."""

    closed_face_nodes = _pad_closed_face_nodes(
        face_node_connectivity, n_face, n_max_face_nodes, n_nodes_per_face
    )

    if projection:
        lonlat_proj = projection.transform_points(
            ccrs.PlateCarree(central_longitude=central_longitude), node_lon, node_lat
        )

        node_lon = lonlat_proj[:, 0]
        node_lat = lonlat_proj[:, 1]

    polygon_shells = (
        np.array(
            [node_lon[closed_face_nodes], node_lat[closed_face_nodes]], dtype=np.float32
        )
        .swapaxes(0, 1)
        .swapaxes(1, 2)
    )

    return polygon_shells


def _build_corrected_polygon_shells(polygon_shells):
    """Constructs ``corrected_polygon_shells`` and
    ``Grid.original_to_corrected), representing the polygon shells, with
    antimeridian polygons split.

     Parameters
    ----------
    grid : uxarray.Grid
        Grid Object

    Returns
    -------
    corrected_polygon_shells : np.ndarray
        Array containing polygon shells, with antimeridian polygons split
    _corrected_shells_to_original_faces : np.ndarray
        Original indices used to map the corrected polygon shells to their entries in face nodes
    """

    # import optional dependencies
    import antimeridian

    # list of shapely Polygons representing each Face in our grid
    polygons = [Polygon(shell) for shell in polygon_shells]

    # List of Polygons (non-split) and MultiPolygons (split across antimeridian)
    corrected_polygons = [
        antimeridian.fix_polygon(P, fix_winding=False) for P in polygons
    ]

    _corrected_shells_to_original_faces = []
    corrected_polygon_shells = []

    for i, polygon in enumerate(corrected_polygons):
        # Convert MultiPolygons into individual Polygon Vertices
        if polygon.geom_type == "MultiPolygon":
            for individual_polygon in polygon.geoms:
                corrected_polygon_shells.append(
                    np.array(
                        [
                            individual_polygon.exterior.coords.xy[0],
                            individual_polygon.exterior.coords.xy[1],
                        ]
                    ).T
                )
                _corrected_shells_to_original_faces.append(i)

        # Convert Shapely Polygon into Polygon Vertices
        else:
            corrected_polygon_shells.append(
                np.array(
                    [polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1]]
                ).T
            )
            _corrected_shells_to_original_faces.append(i)

    return corrected_polygon_shells, _corrected_shells_to_original_faces


def _convert_shells_to_polygons(shells):
    """Convert polygon shells to shapely Polygon or MultiPolygon objects."""
    polygons = []
    for shell in shells:
        # Remove NaN values from each polygon shell
        cleaned_shell = shell[~np.isnan(shell[:, 0])]
        if len(cleaned_shell) > 2:  # A valid polygon needs at least 3 points
            polygons.append(Polygon(cleaned_shell))

    return polygons
