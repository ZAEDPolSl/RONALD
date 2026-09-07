import numpy as np
from scipy.spatial import KDTree

from ronald.modelling.visvalingam import (
    visvalingam_whyatt_3d,
    visvalingam_whyatt_3d_many,
)

ADAPTIVE_AREAS = (20.0, 15.0, 10.0, 5.0, 2.0, 1.0)
DEFAULT_AREA = 10.0
FOREGROUND_CHUNK_DEPTH = 16


class EdgeSpatialIndex:
    """Reusable global spatial index over every skeleton edge point."""

    def __init__(self, edge_list):
        self.edge_count = len(edge_list)
        self.edge_points = (
            np.concatenate(edge_list, axis=0)
            if edge_list
            else np.empty((0, 3), dtype=float)
        )
        self.edge_ids = (
            np.concatenate(
                [
                    np.full(len(edge), edge_idx, dtype=np.int32)
                    for edge_idx, edge in enumerate(edge_list)
                ]
            )
            if edge_list
            else np.empty(0, dtype=np.int32)
        )
        self.tree = KDTree(self.edge_points) if len(self.edge_points) else None

        # The legacy exact-match pass overwrites earlier matches, so the largest
        # edge index owns coordinates shared by multiple skeleton edges.
        self.exact_edge = {}
        for edge_idx, edge in enumerate(edge_list):
            for coordinate in edge:
                self.exact_edge[tuple(coordinate)] = edge_idx

    def query(self, points, tol=1e-6):
        min_distances = np.full(points.shape[0], np.inf)
        min_edge_indices = np.full(points.shape[0], -1, dtype=int)
        if len(points) == 0 or self.tree is None:
            return min_edge_indices, min_distances

        exact = np.fromiter(
            (self.exact_edge.get(tuple(point), -1) for point in points),
            dtype=np.int32,
            count=len(points),
        )
        exact_mask = exact >= 0
        min_edge_indices[exact_mask] = exact[exact_mask]
        min_distances[exact_mask] = 0.0

        query_indices = np.flatnonzero(~exact_mask)
        if len(query_indices) == 0:
            return min_edge_indices, min_distances

        query_points = points[query_indices]

        # Preserve the legacy tolerance behavior: once the first edge within
        # ``tol`` is encountered, later edges are no longer considered.
        nearby = self.tree.query_ball_point(query_points, tol)
        tolerance_locked = np.zeros(len(query_points), dtype=bool)
        for local_idx, candidates in enumerate(nearby):
            if not candidates:
                continue
            candidates = np.asarray(candidates)
            candidate_edge_ids = self.edge_ids[candidates]
            selected_edge = candidate_edge_ids.min()
            selected = candidates[candidate_edge_ids == selected_edge]
            selected_points = self.edge_points[selected]
            selected_distance = np.linalg.norm(
                selected_points - query_points[local_idx], axis=1
            ).min()
            min_edge_indices[query_indices[local_idx]] = selected_edge
            min_distances[query_indices[local_idx]] = selected_distance
            tolerance_locked[local_idx] = True

        query_indices = query_indices[~tolerance_locked]
        query_points = query_points[~tolerance_locked]
        if len(query_indices) == 0:
            return min_edge_indices, min_distances

        if len(self.edge_points) == 1:
            distances, point_indices = self.tree.query(query_points, k=1)
            min_edge_indices[query_indices] = self.edge_ids[point_indices]
            min_distances[query_indices] = distances
            return min_edge_indices, min_distances

        distances, point_indices = self.tree.query(query_points, k=2, workers=-1)
        selected_edges = self.edge_ids[point_indices[:, 0]].copy()
        selected_distances = distances[:, 0].copy()

        # A global KD-tree may return any point when distances tie. The legacy
        # edge-by-edge loop deterministically keeps the smallest edge index.
        ambiguous = np.flatnonzero(distances[:, 0] == distances[:, 1])
        for local_idx in ambiguous:
            radius = np.nextafter(selected_distances[local_idx], np.inf)
            candidates = self.tree.query_ball_point(query_points[local_idx], radius)
            candidate_points = self.edge_points[candidates]
            squared_distances = np.sum(
                (candidate_points - query_points[local_idx]) ** 2, axis=1
            )
            nearest = squared_distances == squared_distances.min()
            selected_edges[local_idx] = self.edge_ids[
                np.asarray(candidates)[nearest]
            ].min()

        min_edge_indices[query_indices] = selected_edges
        min_distances[query_indices] = selected_distances
        return min_edge_indices, min_distances


def closest_edge_indices(branch, edge_list, tol=1e-6, spatial_index=None):
    if spatial_index is None:
        spatial_index = EdgeSpatialIndex(edge_list)
    return spatial_index.query(branch, tol=tol)


def indices_of_kept_points(branch, points):
    branch_view = branch.view([("", branch.dtype)] * branch.shape[1]).reshape(-1)
    points_view = points.view([("", points.dtype)] * points.shape[1]).reshape(-1)
    mask = np.isin(branch_view, points_view)
    return np.where(mask)[0]


def _preserve_duplicate_coordinate_indices(branch, indices, has_duplicates):
    """Match the legacy coordinate-membership behavior for repeated points."""
    if not has_duplicates:
        return indices
    return indices_of_kept_points(branch, branch[indices])


def segment_branch(branch: np.ndarray, adaptive=False) -> list:
    """Segment a branch by using the Visvalingam-Whyatt algorithm.

    Parameters
    ----------
    branch : np.ndarray
        A 2D array of shape (n_points, 3) representing the branch.
    adaptive : bool, optional
        Whether to use adaptive segmentation (default is False).

    Returns
    -------
    list
        A list of 1D numpy arrays of indices of the points that were kept.
    """
    segmentation_options = []
    has_duplicates = len({tuple(point) for point in branch}) != len(branch)

    def add_unique(indices):
        if not any(
            np.array_equal(indices, existing) for existing in segmentation_options
        ):
            segmentation_options.append(indices)

    if adaptive:
        add_unique(np.array([0, len(branch) - 1]))

        simplified_options = visvalingam_whyatt_3d_many(
            branch, ADAPTIVE_AREAS, return_indices=True
        )
        for indices in simplified_options:
            indices = _preserve_duplicate_coordinate_indices(
                branch, indices, has_duplicates
            )
            add_unique(indices)
    else:
        indices = visvalingam_whyatt_3d(branch, DEFAULT_AREA, return_indices=True)
        indices = _preserve_duplicate_coordinate_indices(
            branch, indices, has_duplicates
        )
        add_unique(indices)
    return segmentation_options


def assign_edge_number(graph):
    edge_list = []
    for u, v, data in graph.edges(data=True):
        edge_list.append(np.asarray(data["pts"]))
        data["mask"] = len(edge_list)
    return edge_list, graph


def _foreground_coordinates(image, chunk_depth=FOREGROUND_CHUNK_DEPTH):
    """Return coordinates equal to one without a full-volume temporary."""
    chunks = []
    for start in range(0, image.shape[0], chunk_depth):
        stop = min(start + chunk_depth, image.shape[0])
        image_chunk = image[start:stop]
        foreground = image_chunk if image_chunk.dtype == np.bool_ else image_chunk == 1
        coordinates = np.argwhere(foreground)
        if len(coordinates):
            coordinates[:, 0] += start
            chunks.append(coordinates)

    if not chunks:
        return np.empty((0, image.ndim), dtype=np.intp)
    return np.concatenate(chunks)


def assign_branch(image, graph, label_dtype=int):
    edge_list, graph = assign_edge_number(graph)
    spatial_index = graph.graph.get("_ronald_edge_spatial_index")
    if spatial_index is None or spatial_index.edge_count != len(edge_list):
        spatial_index = EdgeSpatialIndex(edge_list)
        graph.graph["_ronald_edge_spatial_index"] = spatial_index

    checkpoints = _foreground_coordinates(image)
    edge_indices, _ = closest_edge_indices(
        checkpoints, edge_list, spatial_index=spatial_index
    )

    branch_labels = np.zeros_like(image, dtype=label_dtype)
    branch_labels[tuple(checkpoints.T)] = edge_indices + 1
    return branch_labels, graph
