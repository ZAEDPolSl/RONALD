"""The accepted binary-wall Sato preprocessing and two-component GMM.

Keep this stage separate from tube modeling so changes to the filter can be
validated independently. Responses are computed on the mask, never on CT HU.
"""

import os

import numpy as np
import SimpleITK as sitk
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from ronald.modelling.prepare_graph import expand_bbox, foreground_bbox
from ronald.segmentation.airways_segmentation import _parallel_sato

SIGMAS = (0.5, 1, 2, 3, 5, 8, 12)
# A Hessian uses two derivative convolutions at sigma/sqrt(2), truncate=8.
PADDING = int(np.ceil(2 * 8 * max(SIGMAS) / np.sqrt(2))) + 2
GMM_PARAMETERS = dict(
    n_components=2, random_state=42, n_init=3, max_iter=300, tol=1e-4, reg_covar=1e-6
)


def _weighted_upper_threshold(means, stds, weights):
    """Select one weighted-Gaussian crossing, never a disconnected MAP class.

    Prefer the crossing between ordered means. When a broad high-mean
    component also owns the low tail, use its upper crossing. If no crossing
    exists at/above the low mean, report the midpoint fallback explicitly.
    """
    means, stds, weights = (np.asarray(v, dtype=float) for v in (means, stds, weights))
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in (means, stds, weights)):
        raise ValueError("Expected two finite Gaussian components")
    if np.any(stds <= 0) or np.any(weights <= 0) or means[0] > means[1]:
        raise ValueError("Expected positive scales/weights and ordered means")
    # log p(low) - log p(high) = a*x*x + b*x + c.
    a = 0.5 / stds[1] ** 2 - 0.5 / stds[0] ** 2
    b = means[0] / stds[0] ** 2 - means[1] / stds[1] ** 2
    c = (0.5 * (means[1] / stds[1]) ** 2
         - 0.5 * (means[0] / stds[0]) ** 2
         + np.log(weights[0] * stds[1] / (weights[1] * stds[0])))
    roots = np.roots([a, b, c] if abs(a) > 1e-12 else [b, c])
    roots = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9)
    inside = [r for r in roots if means[0] <= r <= means[1]]
    if inside:
        # Preserve the existing, well-conditioned between-means calculation.
        threshold = brentq(lambda x: a*x*x + b*x + c, means[0], means[1])
        rule = "weighted intersection between means"
    else:
        upper = [r for r in roots if r > means[1]]
        threshold = min(upper) if upper else float(np.mean(means))
        rule = "weighted upper crossing" if upper else "midpoint fallback: no upper crossing"
    return float(threshold), rule, roots


def _highest_gmm_component(values):
    """Fit two Gaussians and return a monotone upper-tail threshold mask."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("GMM values must be a finite one-dimensional array")
    if len(values) < 2 or np.ptp(values) == 0:
        return np.zeros(values.shape, bool), dict(
            status="constant_response",
            threshold=None,
            components=[],
            map_threshold_disagreements=0,
        )
    gmm = GaussianMixture(**GMM_PARAMETERS).fit(values[:, None])
    if not gmm.converged_:
        raise RuntimeError("Sato GMM did not converge")
    order = np.argsort(gmm.means_.ravel())
    means = gmm.means_.ravel()[order]
    stds = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]

    def log_score(x, index):
        return np.log(weights[index]) + norm.logpdf(x, means[index], stds[index])

    high = (
        np.argmax(np.column_stack([log_score(values, i) for i in range(2)]), axis=1)
        == 1
    )
    threshold, threshold_rule, crossings = _weighted_upper_threshold(means, stds, weights)
    selected = values >= threshold
    report = dict(
        status="complete",
        threshold=threshold,
        threshold_rule=threshold_rule,
        crossings=crossings,
        mask_rule="response >= threshold; no MAP classification",
        domain="all walls_filled voxels, including zeros",
        map_threshold_disagreements=(
            int(np.count_nonzero(high != (values >= threshold)))
            if threshold is not None
            else None
        ),
        fit_parameters=dict(GMM_PARAMETERS),
        converged=True,
        iterations=int(gmm.n_iter_),
        components=[
            dict(
                component=i + 1,
                mean=float(m),
                standard_deviation=float(s),
                weight=float(w),
            )
            for i, (m, s, w) in enumerate(zip(means, stds, weights))
        ],
    )
    return selected, report


def calculate_sato_mask(walls_filled, *, max_workers=1):
    """Return ``(binary_sato_image, metadata)`` on the filled-wall image grid.

    Apply white-ridge Sato at the accepted voxel scales, take their maximum,
    normalize once over the padded ROI, and threshold using the GMM k=2 crossing.
    No airway protection, dilation, sheet removal, or component removal is used.
    Constant/empty responses return an empty mask, without inventing a class.
    """
    if not isinstance(walls_filled, sitk.Image):
        raise TypeError("walls_filled must be a SimpleITK.Image")
    if (
        walls_filled.GetDimension() != 3
        or walls_filled.GetNumberOfComponentsPerPixel() != 1
    ):
        raise ValueError("walls_filled must be a scalar 3D image")
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer or None")
    walls = sitk.GetArrayViewFromImage(walls_filled)
    for start in range(0, walls.shape[0], 16):
        block = walls[start : start + 16]
        if np.any((block != 0) & (block != 1)):
            raise ValueError("walls_filled must be binary (0 and 1)")
    mask = np.zeros(walls.shape, np.uint8)
    metadata = dict(
        input="binary walls_filled, not CT",
        sigmas_voxels=list(SIGMAS),
        padding_voxels=PADDING,
        airway_protection=False,
        dilation=False,
        walls_voxels=int(np.count_nonzero(walls)),
        k=2,
    )
    bbox = foreground_bbox(walls)
    if bbox is None:
        metadata.update(
            status="empty_walls", threshold=None, highest_component_voxels=0
        )
    else:
        bbox = expand_bbox(bbox, walls.shape, padding=PADDING)
        binary = (walls[bbox] > 0).astype(np.float32)
        response = _parallel_sato(
            binary, SIGMAS, max_workers=max_workers, eigen_chunk_depth=8
        )
        if not np.isfinite(response).all():
            raise ValueError("Sato response contains nonfinite values")
        minimum = float(response.min())
        response -= minimum
        maximum = float(response.max())
        if maximum > 0:
            response /= maximum
        else:
            response.fill(0)
        foreground = binary > 0
        high, fitted = _highest_gmm_component(response[foreground].astype(np.float64))
        mask[bbox][foreground] = high
        metadata.update(
            fitted,
            normalization=dict(raw_minimum=minimum, raw_range=maximum),
            highest_component_voxels=int(high.sum()),
        )
    result = sitk.GetImageFromArray(mask)
    result.CopyInformation(walls_filled)
    return result, metadata
