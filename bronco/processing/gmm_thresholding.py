import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn import mixture
from bronco.utils import solve, get_gmm_metadata


def get_thresholds(gmm_list, max_value):

    thresholds = []

    for i in range(len(gmm_list) - 1):
        current_gauss_dict = gmm_list[i]
        next_gauss_dict = gmm_list[i + 1]

        threshold_candidates = solve(
            current_gauss_dict["mean"],
            next_gauss_dict["mean"],
            current_gauss_dict["std"],
            next_gauss_dict["std"],
        )

        if max(threshold_candidates) < max_value:
            threshold = max(threshold_candidates)
        else:
            threshold = min(threshold_candidates)

        thresholds.append(threshold)

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


def run_thresholding(sitk_image, sitk_mask=None, path_cache=None, number_of_gmms=3, return_thresholds=True):
    # segment the lung area
    if sitk_mask is not None:
        # get min
        stats = sitk.StatisticsImageFilter()
        stats.Execute(sitk_image)
        _min = stats.GetMinimum()
        # get lungs image
        sitk_image = sitk.Mask(sitk_image, sitk_mask, outsideValue=_min - 1)
    image = sitk.GetArrayFromImage(sitk_image)
    # image = np.swapaxes(image, 0, 2)

    nrrd_volume_seg_sq = image.copy()

    X = nrrd_volume_seg_sq.copy()
    X = X.flatten()

    # Remove Background Intensities Outside Patient
    background_val = X.min()
    background_idx = np.where(X == background_val)
    X = np.delete(X, background_idx)

    background_idx = np.where(X > 500)
    X = np.delete(X, background_idx)
    X = X[:, np.newaxis]
    # print("Generating Distplot...")
    # Generate distplot
    # sns_plot = sns.distplot(X)
    # dist_plot_path = os.path.join(output_path, 'dist_plot.png')

    # Model gmm
    # print("Running Gaussian modelling...")
    gmm = mixture.GaussianMixture(n_components=number_of_gmms)
    gmm.fit(X)

    # print("Sorting GMMs")
    gmm_list = get_gmm_metadata(gmm)
    thresholds = get_thresholds(gmm_list, X.max())
    thresholds.insert(0, np.min(image) - 1)
    thresholds.append(np.max(image) + 1)

    thresholds_df = pd.DataFrame(
        data=np.array(thresholds), columns=["threshold"]
    )
    if path_cache is not None:
        thresholds_df.to_csv(
            os.path.join(path_cache, "thresholds.csv"), index=False
        )

    # print("Generating thresholded volumes...")
    segments = create_thresholded_volumes(thresholds, image)
    # segments = np.swapaxes(segments, 0, 2)
    sitk_segments = sitk.GetImageFromArray(segments)
    sitk_segments.CopyInformation(sitk_image)
    if return_thresholds:
        return sitk_segments, thresholds
    else:
        return sitk_segments