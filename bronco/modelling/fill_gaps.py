import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from skimage.draw import line_nd, polygon


def fill_gaps(upper_ellipse, lower_ellipse, smooth_tree_mask):
    upper_unique = np.unique(upper_ellipse, axis=0)
    lower_unique = np.unique(lower_ellipse, axis=0)

    if np.array_equal(upper_unique, lower_unique):
        return smooth_tree_mask

    all_points = np.unique(np.concatenate([upper_unique, lower_unique], axis=0), axis=0)
    n_points = all_points.shape[0]

    if n_points == 0:
        return smooth_tree_mask.astype(int)
    elif n_points == 1:
        mask = np.zeros_like(smooth_tree_mask, dtype=bool)
        mask[tuple(all_points[0])] = True
        np.logical_or(smooth_tree_mask, mask, out=smooth_tree_mask)
        return smooth_tree_mask.astype(int)

    centered = all_points - all_points.mean(axis=0)
    rank = np.linalg.matrix_rank(centered, tol=1e-5)

    mask = np.zeros_like(smooth_tree_mask, dtype=bool)
    if rank == 1:
        mask = fill_line_3d(all_points, mask)
    elif rank == 2:
        mask = fill_plane_3d(all_points, mask, all_points)
    else:
        mask = fill_volume_3d(all_points, mask)

    np.logical_or(smooth_tree_mask, mask, out=smooth_tree_mask)
    return smooth_tree_mask.astype(int)


def fill_line_3d(points, mask):
    """Fill mask along a 3D line defined by points."""
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
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
    return mask


def fill_plane_3d(points, mask, all_points):
    """Fill mask for points approximately on a plane in 3D."""
    variances = points.var(axis=0)
    drop_axis = np.argmin(variances)
    axes_2d = [i for i in range(3) if i != drop_axis]
    points_2d = points[:, axes_2d]

    rank_2d = np.linalg.matrix_rank(points_2d - points_2d.mean(axis=0), tol=1e-5)

    if rank_2d == 1:
        mask = fill_line_2d(points_2d, mask, drop_axis, axes_2d)
    else:
        mask = fill_polygon_2d(points_2d, mask, drop_axis, axes_2d, all_points)
    return mask


def fill_line_2d(points_2d, mask, drop_axis, axes_2d):
    """Fill mask along a line in 2D projection, repeated along dropped axis (vectorized)."""
    p_min = points_2d.min(axis=0)
    p_max = points_2d.max(axis=0)
    rr, cc = line_nd(p_min, p_max)
    valid = (
        (rr >= 0)
        & (rr < mask.shape[axes_2d[0]])
        & (cc >= 0)
        & (cc < mask.shape[axes_2d[1]])
    )
    rr, cc = rr[valid], cc[valid]

    slices = np.arange(mask.shape[drop_axis])

    if drop_axis == 0:
        mask[np.ix_(slices, rr, cc)] = True
    elif drop_axis == 1:
        mask[np.ix_(rr, slices, cc)] = True
    else:
        mask[np.ix_(rr, cc, slices)] = True

    return mask


def fill_polygon_2d(points_2d, mask, drop_axis, axes_2d, all_points):
    try:
        hull_2d = ConvexHull(points_2d, qhull_options="QJ")
        delaunay = Delaunay(points_2d[hull_2d.vertices], qhull_options="QJ")
    except Exception:
        return rasterize_polygon(points_2d, mask, drop_axis, axes_2d)

    min_bb = np.maximum(np.floor(points_2d.min(axis=0)) - 1, 0).astype(int)
    max_bb = np.minimum(
        np.ceil(points_2d.max(axis=0)) + 1, np.array(mask.shape)[axes_2d] - 1
    ).astype(int)

    x, y = np.mgrid[min_bb[0] : max_bb[0] + 1, min_bb[1] : max_bb[1] + 1]
    grid_points = np.vstack((x.ravel(), y.ravel())).T

    mask_flat = delaunay.find_simplex(grid_points) >= 0
    mask_flat = mask_flat.reshape(x.shape)  # Shape (N, M)

    # Get valid slices using original 3D points
    slices_with_points = np.unique(np.round(all_points[:, drop_axis]).astype(int))
    valid_slices = slices_with_points[
        (slices_with_points >= 0) & (slices_with_points < mask.shape[drop_axis])
    ]

    # Loop through slices and assign 2D mask
    for slice_idx in valid_slices:
        if drop_axis == 0:
            mask[
                slice_idx, min_bb[0] : max_bb[0] + 1, min_bb[1] : max_bb[1] + 1
            ] |= mask_flat
        elif drop_axis == 1:
            mask[
                min_bb[0] : max_bb[0] + 1, slice_idx, min_bb[1] : max_bb[1] + 1
            ] |= mask_flat
        else:
            mask[
                min_bb[0] : max_bb[0] + 1, min_bb[1] : max_bb[1] + 1, slice_idx
            ] |= mask_flat

    return mask


def rasterize_polygon(points_2d, mask, drop_axis, axes_2d):
    hull_2d = ConvexHull(points_2d, qhull_options="QJ")
    hull_points = points_2d[hull_2d.vertices]

    min_bb = np.maximum(np.floor(hull_points.min(axis=0)) - 1, 0).astype(int)
    max_bb = np.minimum(
        np.ceil(hull_points.max(axis=0)) + 1, np.array(mask.shape)[axes_2d] - 1
    ).astype(int)

    hull_pts_shifted = hull_points - min_bb

    rr_poly, cc_poly = polygon(
        hull_pts_shifted[:, 0],
        hull_pts_shifted[:, 1],
        shape=(max_bb[0] - min_bb[0] + 1, max_bb[1] - min_bb[1] + 1),
    )
    mask_flat = np.zeros(
        (max_bb[0] - min_bb[0] + 1, max_bb[1] - min_bb[1] + 1), dtype=bool
    )
    mask_flat[rr_poly, cc_poly] = True

    slices = np.arange(mask.shape[drop_axis])
    dim0_indices = np.arange(min_bb[0], max_bb[0] + 1)
    dim1_indices = np.arange(min_bb[1], max_bb[1] + 1)

    if drop_axis == 0:
        mask[np.ix_(slices, dim0_indices, dim1_indices)] |= mask_flat
    elif drop_axis == 1:
        mask[np.ix_(dim0_indices, slices, dim1_indices)] |= mask_flat
    else:
        mask[np.ix_(dim0_indices, dim1_indices, slices)] |= mask_flat

    return mask


def fill_volume_3d(points, mask):
    """Fill mask for full 3D point cloud using Delaunay triangulation with robust fallback."""
    try:
        delaunay = Delaunay(points, qhull_options="QJ0.5")
    except Exception:
        hull = ConvexHull(points, qhull_options="QJ0.5")
        delaunay = Delaunay(points[hull.vertices], qhull_options="QJ0.5")

    # Bounding box calculation
    min_bb = np.maximum(np.floor(points.min(axis=0)) - 1, 0).astype(int)
    max_bb = np.minimum(
        np.ceil(points.max(axis=0)) + 1, np.array(mask.shape) - 1
    ).astype(int)

    # Grid generation
    x, y, z = np.mgrid[
        min_bb[0] : max_bb[0] + 1,
        min_bb[1] : max_bb[1] + 1,
        min_bb[2] : max_bb[2] + 1,
    ]
    grid_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    # Mask filling
    mask_flat = delaunay.find_simplex(grid_points) >= 0
    mask[
        min_bb[0] : max_bb[0] + 1,
        min_bb[1] : max_bb[1] + 1,
        min_bb[2] : max_bb[2] + 1,
    ] |= mask_flat.reshape(x.shape)

    return mask
