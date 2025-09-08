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
    if rank == 1 or n_points <= 2:
        mask = fill_line_3d(all_points, mask)
    elif rank == 2 or n_points == 3:
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
    # Qhull (ConvexHull/Delaunay) needs at least 3 points for 2D hull, 4 for 2D Delaunay
    if points_2d.shape[0] < 4:
        # Not enough points for Delaunay triangulation, fallback to rasterization
        return rasterize_polygon(points_2d, mask, drop_axis, axes_2d)
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
    # ConvexHull needs at least 3 points in 2D
    if points_2d.shape[0] < 3:
        # Not enough points to form a polygon, return mask unchanged
        return mask
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
        mask[np.ix_(slices, dim0_indices, dim1_indices)] |= mask_flat[None, :, :]
    elif drop_axis == 1:
        mask[np.ix_(dim0_indices, slices, dim1_indices)] |= mask_flat[:, None, :]
    else:
        mask[np.ix_(dim0_indices, dim1_indices, slices)] |= mask_flat[:, :, None]

    return mask


def fill_volume_3d(points, mask):
    """Fill mask for full 3D point cloud using Delaunay triangulation with robust fallback."""
    n_points = points.shape[0]

    # Skip Delaunay if we don't have enough points
    if n_points < 5:
        # Fallback: just mark the points
        for p in points:
            idx = tuple(np.round(p).astype(int))
            if all(0 <= idx[d] < mask.shape[d] for d in range(3)):
                mask[idx] = True
        return mask

    # Calculate the volume of the bounding box
    min_bb = np.maximum(np.floor(points.min(axis=0)) - 1, 0).astype(int)
    max_bb = np.minimum(
        np.ceil(points.max(axis=0)) + 1, np.array(mask.shape) - 1
    ).astype(int)

    # Calculate the volume of the bounding box
    bbox_volume = np.prod([max_bb[i] - min_bb[i] + 1 for i in range(3)])

    # If the volume is too large, use a simpler approach
    if bbox_volume > 20_000_000:  # 20 million voxels threshold
        # Fallback: just mark the convex hull edges
        try:
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                for i in range(3):
                    for j in range(i + 1, 4):
                        p1 = points[simplex[i]]
                        p2 = points[simplex[j]]
                        rr, cc, zz = line_nd(p1, p2)
                        valid = (
                            (rr >= 0)
                            & (rr < mask.shape[0])
                            & (cc >= 0)
                            & (cc < mask.shape[1])
                            & (zz >= 0)
                            & (zz < mask.shape[2])
                        )
                        mask[rr[valid], cc[valid], zz[valid]] = True
        except Exception:
            # If even that fails, just mark the points
            for p in points:
                idx = tuple(np.round(p).astype(int))
                if all(0 <= idx[d] < mask.shape[d] for d in range(3)):
                    mask[idx] = True
        return mask

    try:
        # Use Delaunay triangulation with QJ option for robustness
        delaunay = Delaunay(points, qhull_options="QJ")

        # Process the grid in chunks to save memory
        chunk_size = 100  # Process this many slices at a time
        for z_start in range(min_bb[0], max_bb[0] + 1, chunk_size):
            z_end = min(z_start + chunk_size, max_bb[0] + 1)

            # Generate grid for this chunk
            z, y, x = np.mgrid[
                z_start:z_end,
                min_bb[1] : max_bb[1] + 1,
                min_bb[2] : max_bb[2] + 1,
            ]
            grid_points = np.vstack((z.ravel(), y.ravel(), x.ravel())).T

            # Find points inside the triangulation
            mask_flat = delaunay.find_simplex(grid_points) >= 0

            # Update the mask for this chunk
            mask[
                z_start:z_end,
                min_bb[1] : max_bb[1] + 1,
                min_bb[2] : max_bb[2] + 1,
            ] |= mask_flat.reshape(z.shape)

            # Free memory
            z, y, x, grid_points, mask_flat = None, None, None, None, None

    except Exception as e:
        # Fallback: just mark the points
        for p in points:
            idx = tuple(np.round(p).astype(int))
            if all(0 <= idx[d] < mask.shape[d] for d in range(3)):
                mask[idx] = True
    return mask
