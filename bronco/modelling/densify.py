import numpy as np
from scipy.spatial import cKDTree


def densify_point_cloud(points, factor=2):
    # Build a KDTree for efficient neighbor searching
    tree = cKDTree(points)

    # Collect new points by interpolating between close pairs
    new_points = []
    for point in points:
        # Find neighbors within a small distance (proximity threshold)
        distances, neighbors = tree.query(point, k=10)  # Adjust k for more neighbors

        for neighbor_idx in neighbors[1:]:  # Skip the point itself (first neighbor)
            neighbor = points[neighbor_idx]
            # Generate intermediate points between the current point and its neighbor
            for i in range(1, factor):
                frac = i / factor
                intermediate_point = point * (1 - frac) + neighbor * frac
                new_points.append(intermediate_point)

    # Combine the original points with the new, densified points
    new_points = np.vstack((points, np.array(new_points)))
    return new_points
