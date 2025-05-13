import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from skimage.draw import line_nd


def fill_gaps(upper_ellipse, lower_ellipse, smooth_tree_mask):
    all_points = np.unique(
        np.concatenate([upper_ellipse, lower_ellipse], axis=0), axis=0
    )
    n_points = all_points.shape[0]
    mask = np.zeros_like(smooth_tree_mask, dtype=bool)

    if n_points == 0:
        return smooth_tree_mask.astype(int)

    elif n_points == 1:
        mask[tuple(all_points[0])] = True

    else:
        centered = all_points - all_points.mean(axis=0)
        rank = np.linalg.matrix_rank(centered, tol=1e-5)

        if rank == 1:
            # Points lie approximately on a line
            p_min = all_points.min(axis=0)
            p_max = all_points.max(axis=0)
            rr, cc, zz = line_nd(p_min, p_max)
            valid = (
                (rr >= 0)
                & (rr < mask.shape[0])
                & (cc >= 0)
                & (cc < mask.shape[1])
                & (zz >= 0)
                & (zz < mask.shape[2])
            )
            mask[rr[valid], cc[valid], zz[valid]] = True

        else:
            hull = ConvexHull(all_points, qhull_options="QJ")

            min_bb = np.maximum(np.floor(all_points.min(axis=0)) - 1, 0).astype(int)
            max_bb = np.minimum(
                np.ceil(all_points.max(axis=0)) + 1, np.array(mask.shape) - 1
            ).astype(int)

            x, y, z = np.mgrid[
                min_bb[0] : max_bb[0] + 1,
                min_bb[1] : max_bb[1] + 1,
                min_bb[2] : max_bb[2] + 1,
            ]
            grid_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

            delaunay = Delaunay(all_points[hull.vertices])
            mask_flat = delaunay.find_simplex(grid_points) >= 0

            mask_sub = mask[
                min_bb[0] : max_bb[0] + 1,
                min_bb[1] : max_bb[1] + 1,
                min_bb[2] : max_bb[2] + 1,
            ]
            mask_sub[:] = mask_flat.reshape(x.shape)

    filled_mask = np.logical_or(smooth_tree_mask, mask)
    return filled_mask.astype(int)
