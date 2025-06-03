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
    """
    points = np.asarray(points)
    c1 = np.asarray(center1)
    c2 = np.asarray(center2)
    major1 = np.asarray(semi_major_vector1)
    minor1 = np.asarray(semi_minor_vector1)
    major2 = np.asarray(semi_major_vector2)
    minor2 = np.asarray(semi_minor_vector2)

    axis_vector = c2 - c1
    height = np.linalg.norm(axis_vector)
    if height < eps:
        # Degenerate case: treat as a single ellipse at c1
        local_vectors = points - c1
        major_len = np.linalg.norm(major1)
        minor_len = np.linalg.norm(minor1)
        if major_len < eps or minor_len < eps:
            return np.zeros(points.shape[0], dtype=bool)
        major_unit = major1 / (major_len + eps)
        minor_unit = minor1 / (minor_len + eps)
        major_proj = np.dot(local_vectors, major_unit) / (major_len + eps)
        minor_proj = np.dot(local_vectors, minor_unit) / (minor_len + eps)
        return (major_proj**2 + minor_proj**2) <= 1

    axis_unit = axis_vector / (height + eps)
    vec_to_points = points - c1
    proj_lengths = np.dot(vec_to_points, axis_unit)
    height_mask = (proj_lengths >= 0) & (proj_lengths <= height)
    closest_on_axis = c1 + np.outer(proj_lengths, axis_unit)
    local_vectors = points - closest_on_axis

    t = np.clip(proj_lengths / (height + eps), 0, 1)
    # Interpolate axes
    major_interp = (1 - t)[:, None] * major1 + t[:, None] * major2
    minor_interp = (1 - t)[:, None] * minor1 + t[:, None] * minor2

    major_len = np.linalg.norm(major_interp, axis=1)
    minor_len = np.linalg.norm(minor_interp, axis=1)
    # Avoid division by zero
    major_unit = major_interp / (major_len[:, None] + eps)
    minor_unit = minor_interp / (minor_len[:, None] + eps)

    major_proj = np.einsum("ij,ij->i", local_vectors, major_unit) / (major_len + eps)
    minor_proj = np.einsum("ij,ij->i", local_vectors, minor_unit) / (minor_len + eps)

    ellipse_mask = (major_proj**2 + minor_proj**2) <= 1
    return height_mask & ellipse_mask


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
