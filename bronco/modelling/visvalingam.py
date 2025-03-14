import numpy as np


def calculate_triangle_areas(points):
    """
    Vectorized calculation of areas for multiple 3D triangles.

    Args:
        points (numpy.ndarray): Array of shape (N, 3, 3), where each row contains
                                the coordinates of the three vertices of a triangle.

    Returns:
        numpy.ndarray: Array of shape (N,) containing the areas of the triangles.
    """
    AB = points[:, 1] - points[:, 0]
    AC = points[:, 2] - points[:, 0]
    cross_product = np.cross(AB, AC)
    magnitudes = np.linalg.norm(cross_product, axis=1)
    areas = 0.5 * magnitudes
    return areas


def prepare_triangles(points):
    """
    Prepare triangles from consecutive points for area calculation in a vectorized way.

    Args:
        points (numpy.ndarray): Array of 3D points with shape (N, 3).

    Returns:
        numpy.ndarray: Array of shape (N-2, 3, 3) containing triangles.
    """
    # Use slicing to get consecutive triplets of points
    triangles = np.stack([points[:-2], points[1:-1], points[2:]], axis=1)
    return triangles


def visvalingam_whyatt_3d(points, epsilon=0.51):
    """
    Simplify a polyline in 3D using the Visvalingam-Whyatt algorithm with NumPy arrays.

    Args:
        points (numpy.ndarray): Ordered array of 3D points.
        epsilon (float): Minimum effective area threshold.

    Returns:
        numpy.ndarray: Simplified array of 3D points.
    """
    if len(points) <= 2:
        return points

    while True:
        triangles = prepare_triangles(points)
        areas = calculate_triangle_areas(triangles)

        # Initialize point_areas with infinity
        point_areas = np.full(len(points), float("inf"))
        point_areas[1:-1] = areas

        # Find index of point with smallest area
        min_area_index = np.argmin(point_areas)

        # Stop if all remaining areas exceed epsilon
        if point_areas[min_area_index] >= epsilon:
            break

        # Remove point with smallest area
        points = np.delete(points, min_area_index, axis=0)

        # If only two points remain, stop
        if len(points) <= 2:
            break
    print(areas)
    return points
