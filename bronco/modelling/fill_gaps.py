import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon, line


def fill_gaps(upper_ellipse, lower_ellipse, smooth_tree_mask):
    """
    Fill the gaps between the upper and lower ellipses in the smooth tree mask
    using the convex hull of the ellipse points.
    """
    # Stack points and compute convex hull
    all_points = np.unique(np.concatenate([upper_ellipse, lower_ellipse], axis=0), axis=0)
    if len(all_points) > 2:
        hull = ConvexHull(all_points)
        hull_points = all_points[hull.vertices]
        rr, cc = polygon(hull_points[:, 1], hull_points[:, 0], smooth_tree_mask.shape)
        mask = np.zeros_like(smooth_tree_mask, dtype=bool)
        mask[rr, cc] = 1
    else:
        r0, c0 = all_points[0][1], all_points[0][0]
        r1, c1 = all_points[1][1], all_points[1][0]
        rr, cc = line(r0, c0, r1, c1)
        mask[rr, cc] = 1

    # Combine with the original mask
    filled_mask = np.logical_or(smooth_tree_mask, mask)
    return filled_mask.astype(int)
