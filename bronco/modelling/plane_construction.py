import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


def points_in_plane(points, plane_normal, plane_point):
    """
    Check if points are in the plane defined by a normal vector and a point on the plane.

    Parameters:
        points: numpy array of shape (N, 3) - Array of 3D points.
        plane_normal: numpy array of shape (3,) - Normal vector of the plane.
        plane_point: numpy array of shape (3,) - Point on the plane.

    Returns:
        numpy array of bools - True if point is in the plane; False otherwise.
    """
    # Compute vectors from plane point to each point
    vectors = points - plane_point

    # Project onto plane normal
    projection_lengths = np.dot(vectors, plane_normal)

    return np.isclose(projection_lengths, 0)


def fit_plane(points):
    """
    Fit a plane to a set of 3D points using SVD.

    Parameters:
        points: numpy array of shape (N, 3) - Array of 3D points.

    Returns:
        plane_point: numpy array (3,) - A point on the plane (centroid).
        plane_normal: numpy array (3,) - Normal vector of the plane.
    """
    # Compute centroid
    plane_point = np.mean(points, axis=0)

    # Perform SVD to find the normal vector
    _, _, vh = np.linalg.svd(points - plane_point)
    plane_normal = vh[2]  # Normal is the last row of V^T

    return plane_point, plane_normal
