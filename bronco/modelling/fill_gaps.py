import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon


def fill_gaps(upper_ellipse, lower_ellipse, smooth_tree_mask):
    """
    Fill the gaps between the upper and lower ellipses in the smooth tree mask
    using the convex hull of the ellipse points.
    """
    # Stack points and compute convex hull
    all_points = np.vstack((upper_ellipse, lower_ellipse))
    hull = ConvexHull(all_points)
    hull_points = all_points[hull.vertices]

    # Convert to row, col format for polygon
    rr, cc = polygon(hull_points[:, 1], hull_points[:, 0], smooth_tree_mask.shape)

    # Create mask for the hull and fill
    hull_mask = np.zeros_like(smooth_tree_mask, dtype=bool)
    hull_mask[rr, cc] = 1

    # Combine with the original mask
    filled_mask = np.logical_or(smooth_tree_mask, hull_mask)
    return filled_mask.astype(int)
