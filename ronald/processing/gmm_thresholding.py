import os
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn import mixture
from ronald.utils import solve, get_gmm_metadata


def get_thresholds(gmm_list, max_value):
    """Separate adjacent, mean-sorted Gaussians at their in-between crossing.

    Keep the existing unweighted density intersection. ``max_value`` remains
    accepted for caller compatibility, but an unrelated CT maximum must not
    decide which quadratic root separates two components. If their densities
    do not cross between the means, use the midpoint and report that fallback.
    """
    thresholds = []
    for left, right in zip(gmm_list[:-1], gmm_list[1:]):
        low, high = float(left["mean"]), float(right["mean"])
        std_left, std_right = float(left["std"]), float(right["std"])
        if not np.isfinite([low, high, std_left, std_right]).all():
            raise ValueError("Gaussian means and standard deviations must be finite")
        if std_left <= 0 or std_right <= 0:
            raise ValueError("Gaussian standard deviations must be positive")
        if low > high:
            raise ValueError("Gaussian components must be sorted by increasing mean")
        midpoint = low + (high - low) / 2
        # Work in a translated/scaled coordinate system to avoid cancellation
        # for close means with large offsets. The roots are converted back to HU.
        scale = max(high - low, std_left, std_right)
        roots = solve(0.0, (high - low) / scale,
                      std_left / scale, std_right / scale)
        candidates = [low + scale * float(root.real) for root in roots
                      if np.isreal(root) and np.isfinite(root)]
        between = [value for value in candidates if low <= value <= high]
        if between:
            threshold = min(between, key=lambda value: abs(value - midpoint))
        else:
            threshold = midpoint
            warnings.warn(
                f"Gaussian densities have no crossing between means {low:g} and "
                f"{high:g}; using midpoint {midpoint:g}.",
                RuntimeWarning,
                stacklevel=2,
            )
        thresholds.append(float(threshold))
    return thresholds


def create_thresholded_volumes(thresholds, image_seg_volume):
    image_thresholded = np.zeros_like(image_seg_volume)
    for i in range(len(thresholds) - 1):

        lower_threshold = thresholds[i]
        upper_threshold = thresholds[i + 1]

        thresholded = image_seg_volume.copy()
        idx_inside_threshold = np.where(
            (lower_threshold <= thresholded) & (thresholded < upper_threshold)
        )
        idx_outside_threshold = np.where(
            (thresholded < lower_threshold) | (upper_threshold <= thresholded)
        )

        thresholded[idx_inside_threshold] = i + 1
        thresholded[idx_outside_threshold] = 0

        thresholded[image_seg_volume == image_seg_volume.min()] = 0

        image_thresholded += thresholded
    return image_thresholded


def run_thresholding(
    sitk_image,
    sitk_mask=None,
    path_cache=None,
    number_of_gmms=3,
    return_thresholds=True,
    *,
    create_segments=True,
):
    """Fit CT intensity thresholds and optionally create their labelled volume.

    Threshold-only callers can disable ``create_segments`` to avoid allocating
    a full output volume. The returned segment image is then ``None``.
    """
    # segment the lung area
    if sitk_mask is not None:
        # get min
        stats = sitk.StatisticsImageFilter()
        stats.Execute(sitk_image)
        _min = stats.GetMinimum()
        # get lungs image
        sitk_image = sitk.Mask(sitk_image, sitk_mask, outsideValue=_min - 1)
    image = sitk.GetArrayFromImage(sitk_image)
    # Preserve C-order samples and the original exclusion rules, without
    # copying the whole volume repeatedly before fitting.
    flat = image.ravel()
    background_val = flat.min()
    X = flat[(flat != background_val) & ~(flat > 500)][:, np.newaxis]

    gmm = mixture.GaussianMixture(n_components=number_of_gmms)
    gmm.fit(X)

    gmm_list = get_gmm_metadata(gmm)
    thresholds = get_thresholds(gmm_list, X.max())
    thresholds.insert(0, np.min(image) - 1)
    thresholds.append(np.max(image) + 1)

    thresholds_df = pd.DataFrame(data=np.array(thresholds), columns=["threshold"])
    if path_cache is not None:
        thresholds_df.to_csv(os.path.join(path_cache, "thresholds.csv"), index=False)

    sitk_segments = None
    if create_segments:
        segments = create_thresholded_volumes(thresholds, image)
        sitk_segments = sitk.GetImageFromArray(segments)
        sitk_segments.CopyInformation(sitk_image)
    if return_thresholds:
        return sitk_segments, thresholds
    else:
        return sitk_segments
