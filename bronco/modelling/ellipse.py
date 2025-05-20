import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import CircleModel, EllipseModel
from sklearn.decomposition import PCA


def are_points_symmetric(points, tol=1e-10):
    """
    Check if a set of collinear points are symmetric about their midpoint.
    Assumes points are collinear.
    """
    # Normalize line direction
    line_vec = points[1] - points[0]
    line_vec /= np.linalg.norm(line_vec)
    
    # Project points onto the line
    projections = np.dot(points - points[0], line_vec)
    
    # Sort points by projection
    sorted_indices = np.argsort(projections)
    sorted_points = points[sorted_indices]
    sorted_proj = projections[sorted_indices]
    
    # Midpoint projection and center point
    mid_proj = (sorted_proj[0] + sorted_proj[-1]) / 2
    center_point = (sorted_points[0] + sorted_points[-1]) / 2
    
    n = len(sorted_points)
    half = n // 2
    
    # Pairwise midpoints of projections
    pair_mid_proj = (sorted_proj[:half] + sorted_proj[-1:-half-1:-1]) / 2
    # Pairwise midpoints of points in 2D
    pair_mid_points = (sorted_points[:half] + sorted_points[-1:-half-1:-1]) / 2
    
    # Check if all midpoints align with center within tolerance
    proj_check = np.all(np.abs(pair_mid_proj - mid_proj) <= tol)
    point_check = np.all(np.linalg.norm(pair_mid_points - center_point, axis=1) <= tol)
    
    return proj_check and point_check


def are_points_collinear(points, tol=1e-8):
    if len(points) < 3:
        return True  # Two or fewer points are always collinear
    
    p1 = points[0]
    p2 = points[1]
    
    for i in range(2, len(points)):
        p3 = points[i]
        cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        if abs(cross_product) > tol:
            return False  # Points are not collinear within tolerance
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
            if are_points_symmetric(points_2d):
                # Points are collinear and symmetric: treat as circle case
                circle_check = True
                hull = points_2d
            else:
                # Collinear but not symmetric: just use hull as is (line segment)
                hull = points_2d
                circle_check = False
        else:
            hull_obj = ConvexHull(points_2d, qhull_options='QJ')
            hull = points_2d[hull_obj.vertices]
            circle_check = False
    else:
        hull = points_2d
        circle_check = hull.shape[0] < 5

    # Fit ellipse in 2D using skimage
    if not circle_check:
        ellipse = EllipseModel()
        hull_centered = hull - hull.mean(axis=0)
        try:
            success = ellipse.estimate(hull)
            if success and ellipse.params is not None:
                xc, yc, a, b, theta = ellipse.params
                xc += hull.mean(axis=0)[0]
                yc += hull.mean(axis=0)[1]
            else:
                circle_check = True
        except TypeError:
            circle_check = True

    if circle_check:
        theta = 0
        center = np.mean(hull, axis=0)
        xc, yc = center
        a = b = 1e-10
    
        if hull.shape[0] >= 3:
            try:
                circle = CircleModel()
                success = circle.estimate(hull)
                if success and circle.params is not None:
                    xc, yc, r = circle.params
                    a = b = r
                else:
                    r = np.linalg.norm(hull[0] - hull[1]) / 2
                    a = b = r
            except TypeError:
                r = np.linalg.norm(hull[0] - hull[1]) / 2
                a = b = r
        elif hull.shape[0] == 2:
            r = np.linalg.norm(hull[0] - hull[1]) / 2
            a = b = r

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
