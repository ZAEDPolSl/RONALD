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
    """Membership in a loft between two parallel elliptical bases.

    Sections are parallel to the fitted base planes, even when their centers
    move sideways. Their centers and *shape matrices* are interpolated linearly.
    A shape matrix is ``major @ major.T + minor @ minor.T``: reversing either
    axis or exchanging major/minor therefore cannot pinch or twist the loft.
    This interpolates squared radii, rather than signed axis vectors; for
    coaxial circles the intermediate radius is the RMS of the endpoint radii.

    Nonfinite, collapsed, or nonparallel bases produce an empty result. Bases
    in the same plane produce only the union of their bounded endpoint disks,
    never an extrusion normal to that plane.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    result = np.zeros(len(points), dtype=bool)
    if not len(points):
        return result
    c1, c2, major1, minor1, major2, minor2 = (
        np.asarray(value, dtype=float)
        for value in (
            center1,
            center2,
            semi_major_vector1,
            semi_minor_vector1,
            semi_major_vector2,
            semi_minor_vector2,
        )
    )
    vectors = (c1, c2, major1, minor1, major2, minor2)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
        return result
    axes = (major1, minor1, major2, minor2)
    lengths = [np.linalg.norm(v) for v in axes]
    if any(not np.isfinite(length) or length <= eps for length in lengths):
        return result

    normal = np.cross(major1 / lengths[0], minor1 / lengths[1])
    normal_length = np.linalg.norm(normal)
    normal2 = np.cross(major2 / lengths[2], minor2 / lengths[3])
    normal2_length = np.linalg.norm(normal2)
    if normal_length <= eps or normal2_length <= eps:
        return result
    normal /= normal_length
    normal2 /= normal2_length
    if np.linalg.norm(np.cross(normal, normal2)) > max(1e-8, 100 * eps):
        return result

    # An orthonormal transverse basis, independent of center displacement.
    u = major1 / lengths[0]
    v = np.cross(normal, u)
    frame = np.column_stack((u, v))
    a1, b1, a2, b2 = (axis @ frame for axis in axes)
    shape1 = np.outer(a1, a1) + np.outer(b1, b1)
    shape2 = np.outer(a2, a2) + np.outer(b2, b2)
    if not np.isfinite(shape1).all() or not np.isfinite(shape2).all():
        return result

    displacement = c2 - c1
    height = displacement @ normal
    finite = np.isfinite(points).all(axis=1)
    safe_points = np.where(finite[:, None], points, c1)
    relative = safe_points - c1
    plane_distance = relative @ normal

    def inside_section(local, xx, xy, yy):
        # Explicit 2x2 inverse avoids a matrix solve for every candidate voxel.
        determinant = xx * yy - xy * xy
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            quadratic = (
                yy * local[:, 0] ** 2
                - 2 * xy * local[:, 0] * local[:, 1]
                + xx * local[:, 1] ** 2
            )
            return (
                (determinant > 0)
                & np.isfinite(determinant)
                & (quadratic <= determinant * (1 + eps))
            )

    if abs(height) <= eps:
        in_plane = finite & (np.abs(plane_distance) <= eps)
        local1 = relative @ frame
        local2 = (safe_points - c2) @ frame
        return in_plane & (
            inside_section(local1, shape1[0, 0], shape1[0, 1], shape1[1, 1])
            | inside_section(local2, shape2[0, 0], shape2[0, 1], shape2[1, 1])
        )

    t = plane_distance / height
    candidates = finite & (t >= -eps) & (t <= 1 + eps)
    if not candidates.any():
        return result
    section_t = np.clip(t[candidates], 0, 1)
    local = (relative[candidates] - section_t[:, None] * displacement) @ frame
    xx = shape1[0, 0] + section_t * (shape2[0, 0] - shape1[0, 0])
    xy = shape1[0, 1] + section_t * (shape2[0, 1] - shape1[0, 1])
    yy = shape1[1, 1] + section_t * (shape2[1, 1] - shape1[1, 1])
    result[candidates] = inside_section(local, xx, xy, yy)
    return result


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
