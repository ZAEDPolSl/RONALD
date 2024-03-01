import cv2
import pickle
import numpy as np
import SimpleITK as sitk
from functools import wraps


def remove_borders(input_img, kernel_size=(5, 5), iterations=2):
    img = input_img.copy()
    kernel = np.ones(kernel_size, np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=iterations)
    return img_erosion


def remove_borders_from_lungs(input_img, kernel_size=(5, 5), iterations=2):
    img = input_img.copy()

    for i in range(img.shape[0]):
        img[i] = remove_borders(img[i], kernel_size=kernel_size, iterations=iterations)

    return img


def slicing_decorator(func):
    """
    A function decorator which extracts image slices of N-1 dimensions and calls func on each slice. The resulting
     images are then concatenated together with JoinSeries.

    :param func: A function which take a SimpleITK Image as it's first argument
    :return: The result of running func on each slice of image.
    """

    @wraps(func)
    def slice_by_slice(image, *args, **kwargs):
        size = list(image.GetSize())

        number_of_slices = size[-1]
        extract_size = size
        extract_index = [0] * image.GetDimension()

        img_list = []

        extract_size[-1] = 0
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(extract_size)

        for slice_idx in range(0, number_of_slices):
            extract_index[-1] = slice_idx
            extractor.SetIndex(extract_index)

            img_list.append(func(extractor.Execute(image), *args, **kwargs))

        return sitk.JoinSeries(
            img_list, image.GetOrigin()[-1], image.GetSpacing()[-1]
        )

    return slice_by_slice


@slicing_decorator
def erosion_by_slice(sitk_image, kernel_radius=1):
    min_f = sitk.MinimumMaximumImageFilter()
    min_f.Execute(sitk_image)
    min_val = min_f.GetMinimum()

    sitk_mask = sitk_image > min_val
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelType(sitk.sitkBox)
    erode_filter.SetKernelRadius(kernel_radius)
    sitk_mask_eroded = erode_filter.Execute(sitk_mask)
    return sitk_mask_eroded


@slicing_decorator
def dilation_by_slice(sitk_image, kernel_radius=1):
    min_f = sitk.MinimumMaximumImageFilter()
    min_f.Execute(sitk_image)
    min_val = min_f.GetMinimum()

    sitk_mask = sitk_image > min_val
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelType(sitk.sitkBox)
    dilate_filter.SetKernelRadius(kernel_radius)
    sitk_mask_dilated = dilate_filter.Execute(sitk_mask)
    return sitk_mask_dilated


@slicing_decorator
def opening_by_slice(sitk_image, kernel_radius=1):
    min_f = sitk.MinimumMaximumImageFilter()
    min_f.Execute(sitk_image)
    min_val = min_f.GetMinimum()

    sitk_mask = sitk_image > min_val
    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetKernelType(sitk.sitkBox)
    opening_filter.SetKernelRadius(kernel_radius)
    sitk_mask_opened = opening_filter.Execute(sitk_mask)
    return sitk_mask_opened


@slicing_decorator
def closing_by_slice(sitk_image, kernel_radius=1, kernel_type=sitk.sitkBox):
    min_f = sitk.MinimumMaximumImageFilter()
    min_f.Execute(sitk_image)
    min_val = min_f.GetMinimum()

    sitk_mask = sitk_image > min_val
    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetKernelType(kernel_type)
    opening_filter.SetKernelRadius(kernel_radius)
    sitk_mask_opened = opening_filter.Execute(sitk_mask)
    return sitk_mask_opened


@slicing_decorator
def opening_reconstruction_by_slice(sitk_image, kernel_radius=1):
    min_f = sitk.MinimumMaximumImageFilter()
    min_f.Execute(sitk_image)
    min_val = min_f.GetMinimum()

    sitk_mask = sitk_image > min_val
    opening_filter = sitk.BinaryOpeningByReconstructionImageFilter()
    opening_filter.SetKernelType(sitk.sitkBox)
    opening_filter.SetKernelRadius(kernel_radius)
    sitk_mask_opened = opening_filter.Execute(sitk_mask)
    return sitk_mask_opened



def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def get_gmm_metadata(gmm):

    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_

    weights = list(weights)
    means = list(means.squeeze())
    covars = list(covars.squeeze())
    stds = list(np.sqrt(covars))

    gmm_list = [
        {"mean": mean, "weight": weight, "std": std}
        for mean, std, weight in zip(means, stds, weights)
    ]

    gmm_list = sorted(gmm_list, key=lambda x: x["mean"])

    return gmm_list


# Cell
def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = (
        m1 ** 2 / (2 * std1 ** 2)
        - m2 ** 2 / (2 * std2 ** 2)
        - np.log(std2 / std1)
    )
    return np.roots([a, b, c])
