import numpy as np
from scipy.spatial import KDTree

def densify_point_cloud(points: np.ndarray, factor=2) -> np.ndarray:
    """
    Densify a point cloud by interpolating between close pairs of points.
    """
    # Build a KDTree for efficient neighbor searching
    tree = KDTree(points)

    # Collect new points by interpolating between close pairs
    new_points = []
    for point in points:
        # Find neighbors within a small distance (proximity threshold)
        distances, neighbors = tree.query(point, k=min(10, len(points)))  # Adjust k for the number of points

        for neighbor_idx in neighbors[1:]:  # Skip the point itself (first neighbor)
            if neighbor_idx >= len(points):
                continue  # Skip invalid indices
            neighbor = points[neighbor_idx]
            # Generate intermediate points between the current point and its neighbor
            for i in range(1, factor):
                frac = i / factor
                intermediate_point = point * (1 - frac) + neighbor * frac
                new_points.append(intermediate_point)

    # Combine the original points with the new, densified points
    new_points = np.vstack((points, np.array(new_points)))
    return new_points
