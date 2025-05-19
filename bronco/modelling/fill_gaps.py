import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from skimage.draw import line_nd


def fill_gaps(upper_ellipse, lower_ellipse, smooth_tree_mask):
    upper_unique = np.unique(upper_ellipse, axis=0)
    lower_unique = np.unique(lower_ellipse, axis=0)
    same_values = np.array_equal(upper_unique, lower_unique)
    if same_values:
        return smooth_tree_mask

    all_points = np.unique(np.concatenate([upper_unique, lower_unique], axis=0), axis=0)
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

        elif rank == 2:
            # Points lie approximately on a plane
            # Project points to 2D by dropping the axis with smallest variance
            variances = all_points.var(axis=0)
            drop_axis = np.argmin(variances)
            axes_2d = [i for i in range(3) if i != drop_axis]

            # Compute convex hull in 2D
            points_2d = all_points[:, axes_2d]
            hull_2d = ConvexHull(points_2d, qhull_options="QJ")

            # Bounding box in 2D
            min_bb = np.maximum(np.floor(points_2d.min(axis=0)) - 1, 0).astype(int)
            max_bb = np.minimum(np.ceil(points_2d.max(axis=0)) + 1, np.array(mask.shape)[axes_2d] - 1).astype(int)

            # Create grid in 2D bounding box
            x, y = np.mgrid[min_bb[0] : max_bb[0] + 1, min_bb[1] : max_bb[1] + 1]
            grid_points = np.vstack((x.ravel(), y.ravel())).T

            # Delaunay triangulation in 2D on hull vertices
            delaunay = Delaunay(points_2d[hull_2d.vertices])
            mask_flat = delaunay.find_simplex(grid_points) >= 0

            # Fill mask slice-wise along dropped axis
            for slice_idx in range(mask.shape[drop_axis]):
                # Build slice mask
                slice_mask = np.zeros(mask.shape, dtype=bool)
                # Indices for this slice in 3D
                if drop_axis == 0:
                    slice_mask[slice_idx, x, y] = mask_flat.reshape(x.shape)
                elif drop_axis == 1:
                    slice_mask[x, slice_idx, y] = mask_flat.reshape(x.shape)
                else:  # drop_axis == 2
                    slice_mask[x, y, slice_idx] = mask_flat.reshape(x.shape)

                # Check if any original points lie in this slice (within tolerance)
                points_in_slice = np.isclose(all_points[:, drop_axis], slice_idx, atol=0.5)
                if points_in_slice.any():
                    mask |= slice_mask

        else:
            # Full 3D case: use all points for Delaunay to avoid flat simplex error
            hull = ConvexHull(all_points, qhull_options="QJ")

            min_bb = np.maximum(np.floor(all_points.min(axis=0)) - 1, 0).astype(int)
            max_bb = np.minimum(np.ceil(all_points.max(axis=0)) + 1, np.array(mask.shape) - 1).astype(int)

            x, y, z = np.mgrid[
                min_bb[0] : max_bb[0] + 1,
                min_bb[1] : max_bb[1] + 1,
                min_bb[2] : max_bb[2] + 1,
            ]
            grid_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

            # Use all points (not just hull vertices) for Delaunay
            delaunay = Delaunay(all_points)
            mask_flat = delaunay.find_simplex(grid_points) >= 0

            mask_sub = mask[
                min_bb[0] : max_bb[0] + 1,
                min_bb[1] : max_bb[1] + 1,
                min_bb[2] : max_bb[2] + 1,
            ]
            mask_sub[:] = mask_flat.reshape(x.shape)

    filled_mask = np.logical_or(smooth_tree_mask, mask)
    return filled_mask.astype(int)
