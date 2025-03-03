import numpy as np


def is_point_in_cylinder(
    points,
    center1,
    center2,
    semi_major_vector1,
    semi_minor_vector1,
    semi_major_vector2,
    semi_minor_vector2,
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

    Returns:
        numpy array of bools - True if point is inside or on the cylinder; False otherwise.
    """
    # Convert inputs to numpy arrays
    points_array = np.array(points)
    c1 = np.array(center1)
    c2 = np.array(center2)
    semi_major_vector1 = np.array(semi_major_vector1)
    semi_minor_vector1 = np.array(semi_minor_vector1)
    semi_major_vector2 = np.array(semi_major_vector2)
    semi_minor_vector2 = np.array(semi_minor_vector2)

    # Compute cylinder axis vector and its unit vector
    axis_vector = c2 - c1
    height = np.linalg.norm(axis_vector)
    axis_unit_vector = axis_vector / height

    # Project points onto cylinder's axis
    vector_to_points = points_array - c1
    projection_lengths = np.dot(vector_to_points, axis_unit_vector)

    # Check if projections are within height bounds
    height_bounds_check = (0 <= projection_lengths) & (projection_lengths <= height)

    # Find closest points on cylinder's axis
    closest_points_on_axis = c1 + projection_lengths[:, None] * axis_unit_vector

    # Compute local coordinates relative to cylinder's axis
    local_vectors = points_array - closest_points_on_axis

    # Determine which base is closer for each point
    base_choice = np.where(projection_lengths < height / 2, 1, 2)

    # Initialize arrays to store results
    major_projections = np.zeros(len(points))
    minor_projections = np.zeros(len(points))

    # Normalize axis vectors
    semi_major_unit_vector1 = semi_major_vector1 / np.linalg.norm(semi_major_vector1)
    semi_minor_unit_vector1 = semi_minor_vector1 / np.linalg.norm(semi_minor_vector1)
    semi_major_unit_vector2 = semi_major_vector2 / np.linalg.norm(semi_major_vector2)
    semi_minor_unit_vector2 = semi_minor_vector2 / np.linalg.norm(semi_minor_vector2)

    # Project local vectors onto semi-major and semi-minor axis vectors
    for i, choice in enumerate(base_choice):
        if choice == 1:
            major_projection = np.dot(local_vectors[i, :], semi_major_unit_vector1)
            minor_projection = np.dot(local_vectors[i, :], semi_minor_unit_vector1)
        else:
            major_projection = np.dot(local_vectors[i, :], semi_major_unit_vector2)
            minor_projection = np.dot(local_vectors[i, :], semi_minor_unit_vector2)

        major_projections[i] = major_projection
        minor_projections[i] = minor_projection

    # Calculate semi-major and semi-minor axes lengths
    semi_major_axis1 = np.linalg.norm(semi_major_vector1)
    semi_minor_axis1 = np.linalg.norm(semi_minor_vector1)
    semi_major_axis2 = np.linalg.norm(semi_major_vector2)
    semi_minor_axis2 = np.linalg.norm(semi_minor_vector2)

    # Check against ellipse equation
    ellipse_checks = np.zeros(len(points))
    for i, choice in enumerate(base_choice):
        if choice == 1:
            ellipse_checks[i] = (major_projections[i] ** 2 / semi_major_axis1**2) + (
                minor_projections[i] ** 2 / semi_minor_axis1**2
            )
        else:
            ellipse_checks[i] = (major_projections[i] ** 2 / semi_major_axis2**2) + (
                minor_projections[i] ** 2 / semi_minor_axis2**2
            )
    ellipse_checks = ellipse_checks <= 1
    inside_cylinder = height_bounds_check & ellipse_checks
    return inside_cylinder.astype(int)


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
