import itk
import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops


from bronco.external.sitk2itk import (
    ConvertItkImageToSimpleItkImage,
    ConvertSimpleItkImageToItkImage,
)
from bronco.segmentation.blobs_segmentation import blobs_segmentation
from bronco.segmentation.airways_segmentation import fast_marching


def vesselness_filter(sitk_image, sitk_lungs):
    sigma = 1.0
    alpha1 = 0.5
    alpha2 = 2.0
    threshold = 30

    float_sitk = sitk.Cast(sitk_image, sitk.sitkFloat32)
    itk_image = ConvertSimpleItkImageToItkImage(float_sitk, itk.F)
    itk_lungs = ConvertSimpleItkImageToItkImage(sitk_lungs, itk.F)

    itk_image = itk.cast_image_filter(
        itk_image, ttype=[type(itk_image), itk.Image[itk.F, 3]]
    )
    itk_lungs = itk.cast_image_filter(
        itk_lungs, ttype=[type(itk_lungs), itk.Image[itk.F, 3]]
    )

    multiply_filter = itk.MultiplyImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.F, 3], itk.Image[itk.F, 3]
    ].New()
    multiply_filter.SetInput1(itk_image)
    multiply_filter.SetInput2(itk_lungs)
    multiply_filter.Update()
    masked_image = multiply_filter.GetOutput()
    # Convert to float for further processing if needed
    input_image_float = itk.cast_image_filter(
        masked_image, ttype=[type(masked_image), itk.Image[itk.F, 3]]
    )
    # Compute Hessian with ITK
    hessian_image = itk.hessian_recursive_gaussian_image_filter(
        input_image_float, sigma=sigma
    )
    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[
        itk.ctype("float")
    ].New()
    vesselness_filter.SetInput(hessian_image)
    vesselness_filter.SetAlpha1(alpha1)
    vesselness_filter.SetAlpha2(alpha2)
    vesselness_filter.Update()
    itk_output = vesselness_filter.GetOutput()

    # Mask the vesselness response by the lungs mask (so we threshold only inside lungs)
    multiply_filter = itk.MultiplyImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.F, 3], itk.Image[itk.F, 3]
    ].New()
    multiply_filter.SetInput1(itk_output)
    multiply_filter.SetInput2(itk_lungs)
    multiply_filter.Update()
    output_image = multiply_filter.GetOutput()

    # Keep a SimpleITK copy of the (masked) vesselness response values
    direction = float_sitk.GetDirection()
    sitk_vesselness = ConvertItkImageToSimpleItkImage(
        output_image, sitk.sitkFloat32, direction
    )

    threshold_filter = itk.BinaryThresholdImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.UC, 3]
    ].New()
    threshold_filter.SetInput(output_image)
    threshold_filter.SetLowerThreshold(threshold)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.SetInsideValue(255)
    threshold_filter.Update()
    output_image = threshold_filter.GetOutput()

    # Convert binary ITK image back to SimpleITK
    sitk_vessels = ConvertItkImageToSimpleItkImage(
        output_image, sitk.sitkUInt8, direction
    )
    return sitk_vessels, sitk_vesselness


def vesselness_to_speed(
    sitk_vesselness, sitk_mask=None, min_speed=1e-3, max_speed=1.0, power=2.0
):
    """Simple mapping: normalize vesselness, invert, raise to `power`, map to [min_speed,max_speed].

    - All-zero vesselness -> uniform speed = max_speed (fast everywhere).
    - Optional `sitk_mask`: outside mask speeds set to min_speed.
    """

    arr = sitk.GetArrayFromImage(sitk_vesselness).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    m = arr.max()
    v_norm = arr / float(m) if m > 0.0 else arr * 0.0

    inv = 1.0 - np.clip(v_norm, 0.0, 1.0)
    mapped = np.power(inv, float(power))
    speed = float(min_speed) + (float(max_speed) - float(min_speed)) * mapped

    if sitk_mask is not None:
        mask = sitk.GetArrayFromImage(sitk_mask) > 0
        speed[~mask] = float(min_speed)

    sitk_speed = sitk.GetImageFromArray(speed.astype(np.float32))
    sitk_speed.CopyInformation(sitk_vesselness)
    return sitk_speed


def vessel_segmentation(
    sitk_image,
    sitk_lungs,
    sitk_lobes=None,
    sitk_mediastinum=None,
    binary=True,
):
    sitk_vessels, sitk_vesselness = vesselness_filter(sitk_image, sitk_lungs)

    sitk_speed = vesselness_to_speed(sitk_vesselness, sitk_mask=sitk_lungs, power=2.0)
    if sitk_mediastinum is None:
        from bronco.segmentation import mediastinum_segmentation

        sitk_mediastinum = mediastinum_segmentation(sitk_lungs)
    if sitk_lobes is None:
        from bronco.segmentation import lobes_segmentation

        sitk_lobes = lobes_segmentation(sitk_image)

    lungs = sitk.GetArrayFromImage(sitk_lungs)
    vessels = sitk.GetArrayFromImage(sitk_vessels)
    mediastinum = sitk.GetArrayFromImage(sitk_mediastinum)
    lobes = sitk.GetArrayFromImage(sitk_lobes)

    blood_system = np.logical_or(mediastinum, vessels).astype(int)
    blood_regions = regionprops(blood_system)
    sorted_regions = sorted(blood_regions, key=lambda x: x.area)
    first_region_mask = np.zeros_like(blood_system, dtype=int)
    first_region_mask[tuple(sorted_regions[0].coords.T)] = 1
    first_region_mask[mediastinum > 1] = 0
    first_region_mask[first_region_mask > 0] = lobes[first_region_mask > 0]
    first_region_mask[lungs == 0] = 0

    if binary:
        first_region_mask[first_region_mask > 0] = 1

    vessels_connected = sitk.GetImageFromArray(first_region_mask)
    vessels_connected.CopyInformation(sitk_vessels)
    vessels_connected = sitk.Cast(vessels_connected, sitk.sitkUInt8)

    blobs = blobs_segmentation(vessels_connected, sitk_lungs, sigma_max=12, steps=10)
    radius = (3,) * blobs.GetDimension()
    blobs = sitk.BinaryMorphologicalClosing(blobs, radius)

    blob_arr = sitk.GetArrayFromImage(blobs) > 0
    list_points = np.argwhere(blob_arr > 0).tolist()

    expanded_blobs_fm = fast_marching(
        sitk_speed, seed_point=list_points, stopping_value=1
    )
    expanded_blobs_mask = sitk.GetImageFromArray(
        (expanded_blobs_fm > 0).astype(np.uint8)
    )
    expanded_blobs_mask.CopyInformation(sitk_vessels)
    # Filter expanded components: keep only those that extend outside current vessels
    labeled_exp = sitk.ConnectedComponent(expanded_blobs_mask)
    labeled_np = sitk.GetArrayFromImage(labeled_exp).astype(np.int32)
    vessels_conn_np = sitk.GetArrayFromImage(vessels_connected) > 0

    # Fast vectorized check: for each label count voxels that are outside vessels_connected.
    # Keep labels that have at least one voxel outside (i.e. expansion reaches outside vessels).
    labels_flat = labeled_np.ravel().astype(np.int32)
    outside_mask_flat = (~vessels_conn_np).ravel().astype(np.uint8)
    if labels_flat.size == 0:
        filtered_np = np.zeros_like(labeled_np, dtype=np.uint8)
    else:
        max_label = labels_flat.max()
        # bincount with minlength ensures index = label
        outside_counts = np.bincount(
            labels_flat, weights=outside_mask_flat, minlength=max_label + 1
        )
        # labels to keep (exclude background 0)
        keep_labels = np.nonzero(outside_counts > 0)[0]
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size == 0:
            filtered_np = np.zeros_like(labeled_np, dtype=np.uint8)
        else:
            filtered_np = np.isin(labeled_np, keep_labels).astype(np.uint8)

    filtered_expanded_blobs_mask = sitk.GetImageFromArray(filtered_np.astype(np.uint8))
    filtered_expanded_blobs_mask.CopyInformation(expanded_blobs_mask)

    vessels_final = sitk.And(vessels_connected, sitk.Not(filtered_expanded_blobs_mask))

    return vessels_final
