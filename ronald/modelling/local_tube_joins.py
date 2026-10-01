"""Short extensions of adjacent fitted tubes, followed by local seam closing.

Only graph-adjacent ellipse pairs belong here. This is deliberately not a
connector for detached components. Coordinates and radii are in array voxels.
"""

import numpy as np
from scipy import ndimage


def _ellipse(outline):
    points = np.asarray(outline, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 5:
        return None
    if not np.isfinite(points).all():
        return None
    # BranchAnalyser.ellipse_boundary supplies uniform angular samples without
    # a repeated closing point; covariance therefore recovers both semiaxes.
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    center = points.mean(axis=0)
    _, singular, frame = np.linalg.svd(points - center, full_matrices=False)
    radii = singular[:2] * np.sqrt(2.0 / len(points))
    if radii[1] < 0.05 or singular[2] > max(1e-5, singular[0] * 1e-4):
        return None
    return center, frame, radii


def fill_local_tube_join(
    upper_ellipse,
    lower_ellipse,
    smooth_tree_mask,
    walls_mask,
    *,
    junction=None,
    closing_radius=1,
    cast_to_int=True,
    diagnostics=None,
):
    """Add a bounded join in-place, preserving all preexisting foreground.

    Each ellipse is extended along its normal toward the shared graph junction.
    Without a junction, the closest points of the two normal lines define the
    target. Extensions longer than twice the respective minor radius (minimum
    two voxels), or targets too far sideways, are rejected. A small closing is
    applied to these two generated extensions only, never to nearby unrelated
    foreground. Newly added voxels must be inside ``walls_mask``.

    Returns the modified mask, optionally cast to an integer array. A
    supplied diagnostic dict is populated with acceptance and voxel counts.
    Invalid outlines are skipped rather than converted to lines or large hulls.
    """
    if smooth_tree_mask.ndim != 3 or walls_mask.shape != smooth_tree_mask.shape:
        raise ValueError("join masks must have the same 3D shape")
    if closing_radius not in (0, 1, 2):
        raise ValueError("closing_radius must be 0, 1, or 2 voxels")
    info = {} if diagnostics is None else diagnostics

    def finish(reason, **extra):
        info.update(reason=reason, added_voxels=0, **extra)
        return smooth_tree_mask.astype(int) if cast_to_int else smooth_tree_mask

    first, second = _ellipse(upper_ellipse), _ellipse(lower_ellipse)
    if first is None or second is None:
        return finish("invalid_ellipse")
    centers = np.array([first[0], second[0]])
    normals = np.array([first[1][2], second[1][2]])
    if junction is None:
        # Parallel offset axes have no unique intersection. The least-squares
        # closest points keep the target between the bases, not far along them.
        matrix = np.column_stack((normals[0], -normals[1]))
        distances = np.linalg.lstsq(matrix, centers[1] - centers[0], rcond=1e-3)[0]
        target = (
            (centers[0] + distances[0] * normals[0])
            + (centers[1] + distances[1] * normals[1])
        ) / 2
    else:
        target = np.asarray(junction, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("junction must be a finite three-coordinate point")
    tubes = []
    for center, frame, radii in (first, second):
        delta = target - center
        axial = float(delta @ frame[2])
        lateral = delta @ frame[:2].T
        limit = max(2.0, 2.0 * radii[1])
        if abs(axial) > limit + 1e-8:
            return finish(
                "extension_too_long", requested_length=abs(axial), limit=limit
            )
        # Require target inside each ellipse's transverse support, with only a
        # subvoxel allowance for voxelized centers. No lateral tube translation.
        if np.sum((lateral / (radii + 0.5)) ** 2) > 1.0:
            return finish("junction_outside_tube")
        overlap = min(0.75, max(0.5, 0.5 * radii[1]))
        axial_low, axial_high = min(0.0, axial) - overlap, max(0.0, axial) + overlap
        tubes.append((center, frame, radii, axial_low, axial_high))

    minima, maxima = [], []
    for center, frame, radii, low, high in tubes:
        transverse_extent = np.sqrt(np.sum((radii[:, None] * frame[:2]) ** 2, axis=0))
        ends = np.array([center + low * frame[2], center + high * frame[2]])
        minima.append(ends.min(axis=0) - transverse_extent)
        maxima.append(ends.max(axis=0) + transverse_extent)
    padding = closing_radius + 1
    lo = np.maximum(0, np.floor(np.min(minima, axis=0)).astype(int) - padding)
    hi = np.minimum(
        smooth_tree_mask.shape,
        np.ceil(np.max(maxima, axis=0)).astype(int) + padding + 1,
    )
    if np.any(hi <= lo):
        return finish("outside_image")
    shape = hi - lo
    if np.prod(shape, dtype=np.int64) > 2_000_000:
        return finish("join_roi_too_large")
    coordinates = np.indices(tuple(shape), dtype=float).reshape(3, -1).T + lo
    generated = np.zeros(len(coordinates), dtype=bool)
    for center, frame, radii, low, high in tubes:
        local = (coordinates - center) @ frame.T
        generated |= (
            (local[:, 2] >= low)
            & (local[:, 2] <= high)
            & (np.sum((local[:, :2] / radii) ** 2, axis=1) <= 1 + 1e-9)
        )
    generated = generated.reshape(tuple(shape))
    before_closing = int(generated.sum())
    if closing_radius:
        axis = np.arange(-closing_radius, closing_radius + 1)
        grid = np.meshgrid(axis, axis, axis, indexing="ij")
        footprint = sum(a * a for a in grid) <= closing_radius**2
        generated |= ndimage.binary_closing(generated, structure=footprint)
    region = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    generated &= np.asarray(walls_mask[region], dtype=bool)
    destination = smooth_tree_mask[region]
    added = int(np.count_nonzero(generated & ~destination.astype(bool)))
    destination[generated] = True
    info.update(
        reason="accepted",
        added_voxels=added,
        extension_voxels_before_closing=before_closing,
        wall_supported_voxels=int(generated.sum()),
        target_zyx=target.tolist(),
        roi_shape=shape.tolist(),
    )
    return smooth_tree_mask.astype(int) if cast_to_int else smooth_tree_mask
