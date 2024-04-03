import os
import cv2
import pickle
import numpy as np
import SimpleITK as sitk
from functools import wraps
import matplotlib.pyplot as plt


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


def display(text, verbose):
    if verbose > 0:
        print(text)


def get_image2roi(image, mask):
    non_zero_indices = np.argwhere(mask > 0)
    if len(non_zero_indices) > 0:
        min_coords = non_zero_indices.min(axis=0)
        max_coords = non_zero_indices.max(axis=0)
    else:
        return image

    roi = image[min_coords[0]:max_coords[0] + 1,
                min_coords[1]:max_coords[1] + 1]
    return roi


def plot_sum_image(sitk_image, path_save=None, category=None, name=None):
    if type(sitk_image) is sitk.Image:
        im = sitk.GetArrayFromImage(sitk_image)
    else:
        im = sitk_image
    im_sum = np.where(im > 0, 1, 0).sum(axis=1)
    im_sum = np.rot90(np.rot90(im_sum))
    im_sum = get_image2roi(im_sum, np.where(im_sum > 0, 1, 0))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(im_sum, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if category is not None:
        plt.title(category, fontsize=23)
    if path_save is not None and name is not None:
        if category is None:
            path_save_fig = os.path.join(path_save, f"{name}.png")
        else:
            path_save_fig = os.path.join(path_save, category)
            os.makedirs(path_save_fig, exist_ok=True)
            path_save_fig = os.path.join(path_save_fig, f"{name}.png")
        fig.savefig(path_save_fig)
    else:
        plt.show()
    plt.close(fig)


def plot_sum_subplots_image(images, path_save=None, x_titles=None, y_titles=None, category=None, name=None, axis=None,
                            cmap='gray'):
    n = len(images)
    num_rows = int(np.ceil(n / 5))
    num_cols = int(min(n, 5))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows))
    axes = axes.flatten()
    # fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 8, 10))
    for enum, im_sum in enumerate(images):
        if axis is not None:
            im_sum = np.where(im_sum > 0, 1, 0).sum(axis=axis)
            if axis == 1:
                im_sum = np.rot90(np.rot90(im_sum))
            else:
                im_sum = np.rot90(im_sum)
        im_sum = get_image2roi(im_sum, np.where(im_sum > 0, 1, 0))
        if cmap is not None:
            axes[enum].imshow(im_sum, cmap='gray')
        else:
            axes[enum].imshow(im_sum)
        axes[enum].set_xticks([])
        axes[enum].set_yticks([])
        if x_titles is not None:
            axes[enum].set_title(str(x_titles[enum]), fontsize=23)
        if y_titles is not None:
            if not enum % 5:
                axes.set_ylabel(str(y_titles[enum // 5]))
    plt.tight_layout()
    if path_save is not None and name is not None:
        if category is None:
            path_save_fig = os.path.join(path_save, f"{name}.png")
        else:
            path_save_fig = os.path.join(path_save, category)
            os.makedirs(path_save_fig, exist_ok=True)
            path_save_fig = os.path.join(path_save_fig, f"{name}.png")
        fig.savefig(path_save_fig)
    else:
        plt.show()
    plt.close(fig)
