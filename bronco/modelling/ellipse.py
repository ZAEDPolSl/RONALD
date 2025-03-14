import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel, CircleModel


def compute_ellipse_axes_3d(xc, yc, zc, a, b, theta, pca):
    """
    Compute the semi-major and semi-minor axis vectors in 3D after applying PCA.inverse_transform.

    Parameters:
    - xc, yc, zc: Center coordinates in 3D space.
    - a: Semi-major axis length.
    - b: Semi-minor axis length.
    - theta: Rotation angle in radians (ellipse orientation in 2D).
    - pca: Fitted PCA object for inverse transformation.

    Returns:
    - Semi-major axis vector in 3D.
    - Semi-minor axis vector in 3D.
    """
    # Define points on the ellipse in 2D (parameter t = 0, π for major; π/2, 3π/2 for minor)
    points_2d = np.array(
        [
            [xc + a * np.cos(theta), yc + a * np.sin(theta)],  # Major axis point 1
            [xc - a * np.cos(theta), yc - a * np.sin(theta)],  # Major axis point 2
            [xc - b * np.sin(theta), yc + b * np.cos(theta)],  # Minor axis point 1
            [xc + b * np.sin(theta), yc - b * np.cos(theta)],  # Minor axis point 2
        ]
    )

    # Add z-coordinate (assume all points lie on the same z-plane initially)
    points_3d = np.column_stack([points_2d, np.full(points_2d.shape[0], zc)])

    # Apply inverse transform to map back to original space
    transformed_points = pca.inverse_transform(points_3d)

    # Compute vectors for major and minor axes
    major_axis_vector = transformed_points[1] - transformed_points[0]
    minor_axis_vector = transformed_points[3] - transformed_points[2]

    return major_axis_vector / 2, minor_axis_vector / 2


def get_axes_3d(xc, yc, zc, a, b, theta):
    center_3d = np.array([zc, xc, yc])
    major_axis_3d = np.array(
        [
            0.0,
            a * np.cos(theta),
            a * np.sin(theta),
        ]
    )
    minor_axis_3d = np.array(
        [
            0.0,
            -b * np.sin(theta),
            b * np.cos(theta),
        ]
    )

    return center_3d, major_axis_3d, minor_axis_3d


def fit_ellipse_3d(points, center_point, SVD):
    """
    Fit an ellipse to a set of 3D points.
    :param points: Nx3 array of 3D points
    :param center_point: The center point to exclude from the fit
    :param SVD: The fitted SVD transformer (from scikit-learn)
    :return: center (3,), major_axis (3,), minor_axis (3,)
    """
    # Get the components from the fitted SVD
    height_axis, x_axis, y_axis = SVD.components_

    points_2d = points[:, 1:]  # Taking second and third coordinates as 2D projection
    hull_obj = ConvexHull(points_2d)
    hull = points_2d[hull_obj.vertices]

    # Handle case with too few points
    circle_check = points.shape[0] < 5

    # Fit ellipse in 2D using skimage
    if not circle_check:
        ellipse = EllipseModel()
        if ellipse.estimate(hull):
            xc, yc, a, b, theta = ellipse.params
        else:
            circle_check = True

    if circle_check:
        theta = 0
        center = np.mean(hull, axis=0)
        if points.shape[0] >= 3:
            circle = CircleModel()
            circle.estimate(hull)
            xc, yc, r = circle.params
            a = b = r
        elif points.shape[0] == 2:
            xc, yc = center
            r = np.linalg.norm(hull[0] - hull[1]) / 2
            a = b = r
        else:
            xc, yc = center
            a = b = 1e-10
    # Calculate the 3D center using the height component from the original center point
    # center_2d = np.array([xc, yc])
    # center_3d = SVD.inverse_transform(np.array([center_point[0], *center_2d]).reshape(1, -1)).flatten()

    # axes_3d = compute_ellipse_axes_3d(xc, yc, center_point[0], a, b, theta, SVD)
    # major_axis_3d, minor_axis_3d = axes_3d[0], axes_3d[1]
    center_3d, major_axis_3d, minor_axis_3d = get_axes_3d(
        xc, yc, center_point[0], a, b, theta
    )

    return center_3d, major_axis_3d, minor_axis_3d


def find_ellipse(points, center_point, SVD):
    # plane_normal = plane_normal.astype(float)
    # plane_normal /= np.linalg.norm(plane_normal)

    # Check which points are in the plane
    # in_plane = points_in_plane(points, plane_normal, point)
    params = fit_ellipse_3d(points, center_point, SVD)
    # params = fit_ellipse_3d(points[in_plane])
    return params
