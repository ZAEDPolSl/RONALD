import numpy as np
from scipy.spatial import KDTree


def densify_point_cloud(points: np.ndarray, factor=2) -> np.ndarray:
    """
    Densify a point cloud by interpolating between close pairs of points.
    """
    if len(points) < 2 or factor < 2:
        return points.copy()

    tree = KDTree(points)
    k = min(10, len(points))
    distances, neighbors = tree.query(points, k=k)

    new_points = []
    for i, point in enumerate(points):
        for neighbor_idx in neighbors[i][1:]:  # Skip self
            if neighbor_idx <= i or neighbor_idx >= len(points):
                continue  # Avoid duplicates and invalid indices
            neighbor = points[neighbor_idx]
            for j in range(1, factor):
                frac = j / factor
                intermediate_point = point * (1 - frac) + neighbor * frac
                new_points.append(intermediate_point)

    if new_points:
        new_points = np.vstack((points, np.array(new_points)))
    else:
        new_points = points.copy()
    return new_points
