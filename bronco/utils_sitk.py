import numpy as np
import SimpleITK as sitk
from functools import wraps


def sitk_resample(itk_image, out_spacing=(2.0, 2.0, 2.0), is_label=False):
    """Source: https://gist.github.com/mrajchl/ccbd5ed12eb68e0c1afc5da116af614a"""
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


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
def fill_holes_by_slice(sitk_image):
    min_f = sitk.MinimumMaximumImageFilter()
    min_f.Execute(sitk_image)
    min_val = min_f.GetMinimum()

    sitk_mask = sitk_image > min_val
    erode_filter = sitk.BinaryFillholeImageFilter()
    # erode_filter.SetKernelType(sitk.sitkBox)
    # erode_filter.SetKernelRadius(kernel_radius)
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
