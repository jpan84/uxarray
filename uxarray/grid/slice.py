from __future__ import annotations

from typing import TYPE_CHECKING, List, Set, Union

import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

if TYPE_CHECKING:
    pass


def _slice_node_indices(
    grid,
    indices,
    inclusive=True,
):
    """Slices (indexes) an unstructured grid given a list/array of node
    indices, returning a new Grid composed of elements that contain the nodes
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: array-like
        A list or 1-D array of node indices
    inclusive: bool
        Whether to perform inclusive (i.e. elements must contain at least one desired feature from a slice) as opposed
        to exclusive (i.e elements be made up all desired features from a slice)
    """

    if inclusive is False:
        raise ValueError("Exclusive slicing is not yet supported.")

    # faces that saddle nodes given in 'indices'
    face_indices = np.unique(grid.node_face_connectivity.values[indices].ravel())
    face_indices = face_indices[face_indices != INT_FILL_VALUE]

    return _slice_face_indices(grid, face_indices)


def _slice_edge_indices(
    grid,
    indices,
    inclusive=True,
):
    """Slices (indexes) an unstructured grid given a list/array of edge
    indices, returning a new Grid composed of elements that contain the edges
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: array-like
        A list or 1-D array of edge indices
    inclusive: bool
        Whether to perform inclusive (i.e. elements must contain at least one desired feature from a slice) as opposed
        to exclusive (i.e elements be made up all desired features from a slice)
    """

    if inclusive is False:
        raise ValueError("Exclusive slicing is not yet supported.")

    # faces that saddle nodes given in 'indices'
    face_indices = np.unique(grid.edge_face_connectivity.values[indices].ravel())
    face_indices = face_indices[face_indices != INT_FILL_VALUE]

    return _slice_face_indices(grid, face_indices)


def _slice_face_indices(
    grid,
    indices,
    inclusive=True,
    inverse_indices: Union[List[str], Set[str], bool] = False,
):
    """Slices (indexes) an unstructured grid given a list/array of face
    indices, returning a new Grid composed of elements that contain the faces
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: array-like
        A list or 1-D array of face indices
    inclusive: bool
        Whether to perform inclusive (i.e. elements must contain at least one desired feature from a slice) as opposed
        to exclusive (i.e elements be made up all desired features from a slice)
    inverse_indices : Union[List[str], Set[str], bool], optional
        Indicates whether to store the original grids indices. Passing `True` stores the original face centers,
        other reverse indices can be stored by passing any or all of the following: (["face", "edge", "node"], True)
    """
    if inclusive is False:
        raise ValueError("Exclusive slicing is not yet supported.")

    from uxarray.grid import Grid

    ds = grid._ds

    indices = np.asarray(indices, dtype=INT_DTYPE)

    if indices.ndim == 0:
        indices = np.expand_dims(indices, axis=0)

    face_indices = indices

    # nodes of each face (inclusive)
    node_indices = np.unique(grid.face_node_connectivity.values[face_indices].ravel())
    node_indices = node_indices[node_indices != INT_FILL_VALUE]

    # index original dataset to obtain a 'subgrid'
    ds = ds.isel(n_node=node_indices)
    ds = ds.isel(n_face=face_indices)

    # Only slice edge dimension if we already have the connectivity
    if "face_edge_connectivity" in grid._ds:
        edge_indices = np.unique(
            grid.face_edge_connectivity.values[face_indices].ravel()
        )
        edge_indices = edge_indices[edge_indices != INT_FILL_VALUE]
        ds = ds.isel(n_edge=edge_indices)
        ds["subgrid_edge_indices"] = xr.DataArray(edge_indices, dims=["n_edge"])
    else:
        edge_indices = None

    ds["subgrid_node_indices"] = xr.DataArray(node_indices, dims=["n_node"])
    ds["subgrid_face_indices"] = xr.DataArray(face_indices, dims=["n_face"])

    # mapping to update existing connectivity
    node_indices_dict = {
        key: val for key, val in zip(node_indices, np.arange(0, len(node_indices)))
    }
    node_indices_dict[INT_FILL_VALUE] = INT_FILL_VALUE

    for conn_name in grid._ds.data_vars:
        # update or drop connectivity variables to correctly point to the new index of each element

        if "_node_connectivity" in conn_name:
            # update connectivity vars that index into nodes
            ds[conn_name] = xr.DataArray(
                np.vectorize(node_indices_dict.__getitem__, otypes=[INT_DTYPE])(
                    ds[conn_name].values
                ),
                dims=ds[conn_name].dims,
                attrs=ds[conn_name].attrs,
            )

        elif "_connectivity" in conn_name:
            # drop any conn that would require re-computation
            ds = ds.drop_vars(conn_name)

    if inverse_indices:
        inverse_indices_ds = xr.Dataset()

        index_types = {
            "face": face_indices,
            "node": node_indices,
        }

        if edge_indices is not None:
            index_types["edge"] = edge_indices

        if isinstance(inverse_indices, bool):
            inverse_indices_ds["face"] = face_indices
        else:
            for index_type in inverse_indices[0]:
                if index_type in index_types:
                    inverse_indices_ds[index_type] = index_types[index_type]
                else:
                    raise ValueError(
                        "Incorrect type of index for `inverse_indices`. Try passing one of the following "
                        "instead: 'face', 'edge', 'node'"
                    )

        return Grid.from_dataset(
            ds,
            source_grid_spec=grid.source_grid_spec,
            is_subset=True,
            inverse_indices=inverse_indices_ds,
        )

    return Grid.from_dataset(ds, source_grid_spec=grid.source_grid_spec, is_subset=True)
