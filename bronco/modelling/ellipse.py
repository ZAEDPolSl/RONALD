import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import CircleModel, EllipseModel
from sklearn.decomposition import PCA


def are_points_collinear(points):
    # Ensure there are at least 3 points to check
    if len(points) < 3:
        return True  # Two points or fewer are always collinear
    
    # Take the first two points as reference
    p1 = points[0]
    p2 = points[1]
    
    # Check cross product for all subsequent points
    for i in range(2, len(points)):
        p3 = points[i]
        # Compute cross product of vectors (p2 - p1) and (p3 - p1)
        cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        if cross_product != 0:
            return False  # Points are not collinear
    return True


def get_axes_3d(
    xc: float, yc: float, zc: float, a: float, b: float, theta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the center, major axis, and minor axis vectors in 3D space.
    """
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


def fit_ellipse_3d(
    points: np.ndarray, center_point: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """
    Fit an ellipse in 3D space to a set of points.
    """
    points_2d = points[:, 1:]  # Taking second and third coordinates as 2D projection
    points_2d = np.unique(points_2d, axis=0)  # Remove duplicates

    if points_2d.shape[0] >= 3:
        if are_points_collinear(points_2d):
            hull = points_2d
        else:
            hull_obj = ConvexHull(points_2d)
            hull = points_2d[hull_obj.vertices]
    else:
        hull = points_2d
    # Handle case with too few points
    circle_check = hull.shape[0] < 5

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
        if hull.shape[0] >= 3:
            circle = CircleModel()
            circle.estimate(hull)
            xc, yc, r = circle.params
            a = b = r
        elif hull.shape[0] == 2:
            xc, yc = center
            r = np.linalg.norm(hull[0] - hull[1]) / 2
            a = b = r
        else:
            xc, yc = center
            a = b = 1e-10

    return xc, yc, center_point[0], a, b, theta


def find_ellipse(
    points: np.ndarray, center_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the center, major axis, and minor axis of an ellipse in 3D space.
    """
    xc, yc, zc, a, b, theta = fit_ellipse_3d(points, center_point)
    params = get_axes_3d(xc, yc, zc, a, b, theta)
    return params
