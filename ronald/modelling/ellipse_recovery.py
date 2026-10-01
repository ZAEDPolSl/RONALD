"""Validity checks for locally fitted cylinder sections, in PCA voxel units."""

import numpy as np


def ellipse_is_valid(ellipse, points=None, *, eps=1e-10):
    """Reject numerical collapse, not anatomically small but supported ellipses.

    Supplying points also requires independent transverse data support. Omitting
    them checks geometry only, before accepting a fitted or borrowed section.
    """
    if ellipse is None:
        return False
    geometry = np.asarray(ellipse, dtype=float)
    if geometry.shape != (3, 3) or not np.all(np.isfinite(geometry)):
        return False
    lengths = np.linalg.norm(geometry[1:], axis=1)
    if np.any(lengths <= max(100 * eps, 1e-6)):
        return False
    axes = geometry[1:] / lengths[:, None]
    if abs(np.dot(axes[0], axes[1])) > 1e-6:
        return False
    if np.any(np.abs(axes[:, 0]) > 1e-6):
        return False
    if points is not None:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            return False
        projected = np.unique(points[:, 1:], axis=0)
        if len(projected) < 3:
            return False
        singular = np.linalg.svd(projected - projected.mean(axis=0), compute_uv=False)
        if singular[0] == 0 or singular[1] <= singular[0] * 1e-10:
            return False
    return True
