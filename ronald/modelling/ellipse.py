"""Membership of planar ellipse sections in the local PCA frame."""

import numpy as np


def check_ellipse(points, center, major_axis, minor_axis, eps=1e-10):
    """
    Check if points are inside an ellipse defined by center and semi-major/minor axes.

    Args:
        points: Array of points to check
        center: Center of the ellipse
        major_axis: Semi-major axis vector
        minor_axis: Semi-minor axis vector
        eps: Small epsilon for numerical stability

    Returns:
        Boolean array indicating which points are inside the ellipse
    """
    points = np.asarray(points)
    center = np.asarray(center)
    major_axis = np.asarray(major_axis)
    minor_axis = np.asarray(minor_axis)

    # Vector from center to points
    local_vectors = points - center

    # Get axis lengths
    major_len = np.linalg.norm(major_axis)
    minor_len = np.linalg.norm(minor_axis)

    if major_len < eps or minor_len < eps:
        return np.zeros(points.shape[0], dtype=bool)

    # Normalize axes
    major_unit = major_axis / (major_len + eps)
    minor_unit = minor_axis / (minor_len + eps)

    # Project points onto the axes
    major_proj = np.dot(local_vectors, major_unit) / (major_len + eps)
    minor_proj = np.dot(local_vectors, minor_unit) / (minor_len + eps)

    # Check if points are inside the ellipse
    return (major_proj**2 + minor_proj**2) <= 1
