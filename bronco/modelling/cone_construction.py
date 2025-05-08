import numpy as np


def is_point_in_cylinder(
    points,
    center1,
    center2,
    semi_major_vector1,
    semi_minor_vector1,
    semi_major_vector2,
    semi_minor_vector2,
    eps=1e-10,
):
    """
    Check if points are inside a cylinder with oval bases defined by semi-major and semi-minor axis vectors.

    Parameters:
        points: numpy array of shape (N, 3) - Array of points to check.
        center1: tuple (x1, y1, z1) - Center of one base.
        center2: tuple (x2, y2, z2) - Center of the other base.
        semi_major_vector1: tuple (dx_major1, dy_major1, dz_major1) - Semi-major axis vector of the first base.
        semi_minor_vector1: tuple (dx_minor1, dy_minor1, dz_minor1) - Semi-minor axis vector of the first base.
        semi_major_vector2: tuple (dx_major2, dy_major2, dz_major2) - Semi-major axis vector of the second base.
        semi_minor_vector2: tuple (dx_minor2, dy_minor2, dz_minor2) - Semi-minor axis vector of the second base.
        eps: float - Small value to avoid division by zero.

    Returns:
        numpy array of bools - True if point is inside or on the cylinder; False otherwise.
    """
    # Convert inputs to numpy arrays
    points_array = np.array(points)
    c1 = np.array(center1)
    c2 = np.array(center2)
    major1 = np.array(semi_major_vector1)
    minor1 = np.array(semi_minor_vector1)
    major2 = np.array(semi_major_vector2)
    minor2 = np.array(semi_minor_vector2)

    # Compute cylinder axis vector and normalize
    axis_vector = c2 - c1
    height = np.linalg.norm(axis_vector)
    axis_unit_vector = axis_vector / (height + eps)

    # Project points onto cylinder's axis
    vector_to_points = points_array - c1
    projection_lengths = np.dot(vector_to_points, axis_unit_vector)

    # Check if projections are within height bounds
    height_bounds_check = (0 <= projection_lengths) & (projection_lengths <= height)

    # Find closest points on axis
    closest_points_on_axis = c1 + projection_lengths[:, None] * axis_unit_vector

    # Compute local coordinates relative to cylinder axis
    local_vectors = points_array - closest_points_on_axis

    # Interpolation factor (clamped between 0 and 1)
    t = np.clip(projection_lengths / (height + eps), 0, 1)

    # Interpolate and renormalize semi-major and semi-minor vectors
    major_interp = (1 - t[:, None]) * major1 + t[:, None] * major2
    minor_interp = (1 - t[:, None]) * minor1 + t[:, None] * minor2

    # Renormalize interpolated vectors to preserve ellipse shape
    major_unit = major_interp / (
        np.linalg.norm(major_interp, axis=1, keepdims=True) + eps
    )
    minor_unit = minor_interp / (
        np.linalg.norm(minor_interp, axis=1, keepdims=True) + eps
    )

    # Compute the correct semi-major and semi-minor axes lengths at each interpolation step
    semi_major_length = (1 - t) * np.linalg.norm(major1) + t * np.linalg.norm(major2)
    semi_minor_length = (1 - t) * np.linalg.norm(minor1) + t * np.linalg.norm(minor2)

    # Project local vectors onto the unit major and minor axes
    major_projections = (
        np.einsum("ij,ij->i", local_vectors, major_unit) / (semi_major_length + eps)
    )
    minor_projections = (
        np.einsum("ij,ij->i", local_vectors, minor_unit) / (semi_minor_length + eps)
    )

    # Ellipse equation check
    ellipse_checks = (major_projections**2 + minor_projections**2) <= 1

    # Final result: inside the cylinder if within height bounds and inside the ellipse
    inside_cylinder = height_bounds_check & ellipse_checks

    return inside_cylinder


if __name__ == "__main__":
    # Example usage
    points_to_check = np.array(
        [[3.5, 3.5, 4], [3.6, 3.6, 4.1], [3.7, 3.7, 4.2]]
    )  # Array of points to check
    center_base_1 = (0, 0, 0)  # Center of one base
    center_base_2 = (3, 3, 5)  # Center of the other base
    semi_major_vector1 = (6, 0, 0)  # Semi-major axis vector of the first base
    semi_minor_vector1 = (0, 2, 0)  # Semi-minor axis vector of the first base
    semi_major_vector2 = (8, 0, 0)  # Semi-major axis vector of the second base
    semi_minor_vector2 = (0, 1, 0)  # Semi-minor axis vector of the second base

    # Check if points are inside cylinder
    inside_cylinder = is_point_in_cylinder(
        points_to_check,
        center_base_1,
        center_base_2,
        semi_major_vector1,
        semi_minor_vector1,
        semi_major_vector2,
        semi_minor_vector2,
    )

    print("Points are inside cylinder:", inside_cylinder)
