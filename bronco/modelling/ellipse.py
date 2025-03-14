import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import CircleModel, EllipseModel
from sklearn.decomposition import PCA


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
