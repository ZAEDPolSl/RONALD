import gc

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.measure import regionprops

from bronco.processing.gmm_thresholding import bic_gmm_foreground_mask

MRI_FRANGI_SIGMAS = (0.6, 0.9, 1.2, 1.8, 2.5, 3.5)
MRI_CROP_MARGIN_VOXELS = 12
MRI_FRANGI_CONTEXT_RADIUS = 2
MRI_AIRWAY_SUPPRESSION_RADIUS = 1
MRI_SURFACE_COMPONENT_BAND_RADIUS = 3


def _mask_bounding_box_xyz(sitk_mask, padding=0):
    mask = sitk.Cast(sitk_mask > 0, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if not stats.HasLabel(1):
        return None

    x, y, z, size_x, size_y, size_z = stats.GetBoundingBox(1)
    image_size = mask.GetSize()

    start = [
        max(0, int(x) - int(padding)),
        max(0, int(y) - int(padding)),
        max(0, int(z) - int(padding)),
    ]
    stop = [
        min(int(image_size[0]), int(x + size_x) + int(padding)),
        min(int(image_size[1]), int(y + size_y) + int(padding)),
        min(int(image_size[2]), int(z + size_z) + int(padding)),
    ]
    size = [stop[i] - start[i] for i in range(3)]
    return tuple(start), tuple(size)


def _crop_to_bbox(sitk_image, bbox_xyz):
    index_xyz, size_xyz = bbox_xyz
    return sitk.RegionOfInterest(sitk_image, size=size_xyz, index=index_xyz)


def _paste_crop_into_reference(sitk_crop, sitk_reference, bbox_xyz, pixel_id=None):
    pixel_id = sitk_crop.GetPixelID() if pixel_id is None else pixel_id
    output = sitk.Image(sitk_reference.GetSize(), pixel_id)
    output.CopyInformation(sitk_reference)
    return sitk.Paste(
        output,
        sitk_crop,
        sourceSize=sitk_crop.GetSize(),
        sourceIndex=(0, 0, 0),
        destinationIndex=bbox_xyz[0],
    )

def _vesselness_filter_ct(
    sitk_image,
    sitk_lungs,
):
    import itk

    from ctools import itk_to_sitk, sitk_to_itk

    sigma = 1.0
    alpha1 = 0.5
    alpha2 = 2.0
    threshold = 30

    float_sitk = sitk.Cast(sitk_image, sitk.sitkFloat32)
    itk_image = sitk_to_itk(float_sitk, itk.F)
    itk_lungs = sitk_to_itk(sitk_lungs, itk.F)

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

    input_image_float = itk.cast_image_filter(
        masked_image, ttype=[type(masked_image), itk.Image[itk.F, 3]]
    )

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

    multiply_filter = itk.MultiplyImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.F, 3], itk.Image[itk.F, 3]
    ].New()
    multiply_filter.SetInput1(itk_output)
    multiply_filter.SetInput2(itk_lungs)
    multiply_filter.Update()
    output_image = multiply_filter.GetOutput()

    direction = float_sitk.GetDirection()
    sitk_vesselness = itk_to_sitk(output_image, sitk.sitkFloat32, direction)

    threshold_filter = itk.BinaryThresholdImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.UC, 3]
    ].New()
    threshold_filter.SetInput(output_image)
    threshold_filter.SetLowerThreshold(threshold)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.SetInsideValue(255)
    threshold_filter.Update()
    output_image = threshold_filter.GetOutput()

    sitk_vessels = itk_to_sitk(output_image, sitk.sitkUInt8, direction)
    return sitk_vessels, sitk_vesselness


def _vesselness_filter_mri(
    sitk_image,
    sitk_lungs,
):
    bbox_xyz = _mask_bounding_box_xyz(sitk_lungs, padding=MRI_CROP_MARGIN_VOXELS)
    if bbox_xyz is None:
        zero_mask = sitk.Image(sitk_image.GetSize(), sitk.sitkUInt8)
        zero_mask.CopyInformation(sitk_image)
        zero_response = sitk.Image(sitk_image.GetSize(), sitk.sitkFloat32)
        zero_response.CopyInformation(sitk_image)
        return zero_mask, zero_response

    image_crop = _crop_to_bbox(sitk_image, bbox_xyz)
    lungs_crop = _crop_to_bbox(sitk_lungs, bbox_xyz)

    image_np = sitk.GetArrayFromImage(sitk.Cast(image_crop, sitk.sitkFloat32)).astype(
        np.float32
    )
    lungs_np = sitk.GetArrayFromImage(lungs_crop) > 0

    masked_np = np.where(lungs_np, image_np, 0.0)
    vesselness_np = frangi(
        masked_np,
        sigmas=MRI_FRANGI_SIGMAS,
        black_ridges=False,
    )
    vesselness_np = np.nan_to_num(
        vesselness_np, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    vesselness_np[~lungs_np] = 0.0

    inside = vesselness_np[lungs_np]
    if inside.size == 0:
        vessel_mask = np.zeros_like(vesselness_np, dtype=np.uint8)
    else:
        foreground_mask, _ = bic_gmm_foreground_mask(inside)
        vessel_mask = np.zeros_like(vesselness_np, dtype=np.uint8)
        vessel_mask[lungs_np] = foreground_mask.astype(np.uint8)

    sitk_vesselness_crop = sitk.GetImageFromArray(vesselness_np)
    sitk_vesselness_crop.CopyInformation(image_crop)

    sitk_vessels_crop = sitk.GetImageFromArray(vessel_mask.astype(np.uint8) * 255)
    sitk_vessels_crop.CopyInformation(image_crop)
    sitk_vessels_crop = sitk.Cast(sitk_vessels_crop, sitk.sitkUInt8)

    sitk_vesselness = _paste_crop_into_reference(
        sitk_vesselness_crop,
        sitk_image,
        bbox_xyz,
        pixel_id=sitk.sitkFloat32,
    )
    sitk_vessels = _paste_crop_into_reference(
        sitk_vessels_crop,
        sitk_image,
        bbox_xyz,
        pixel_id=sitk.sitkUInt8,
    )

    del image_crop, lungs_crop, image_np, lungs_np, masked_np, vesselness_np, inside
    del vessel_mask, sitk_vesselness_crop, sitk_vessels_crop
    gc.collect()
    return sitk_vessels, sitk_vesselness


def _ball(radius):
    coords = np.ogrid[
        -radius : radius + 1,
        -radius : radius + 1,
        -radius : radius + 1,
    ]
    return coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2 <= radius**2


def _threshold_mri_vesselness_inside_lungs(vesselness_np, lungs_np):
    vessels = np.zeros_like(lungs_np, dtype=bool)
    inside = vesselness_np[lungs_np]
    if inside.size == 0:
        return vessels
    foreground_mask, _ = bic_gmm_foreground_mask(inside)
    vessels[lungs_np] = foreground_mask.astype(bool)
    return vessels


def _rough_mri_airway_lumen_mask(sitk_image, sitk_lungs, min_component_voxels=15):
    """Return an exploratory dark-lumen mask for MRI vessel suppression only.

    This is not a validated airway segmentation. It is intentionally conservative
    and is used only to remove obvious dark-airway responses from MRI Frangi
    vessel candidates.
    """
    from bronco.segmentation.lungs_segmentation import _mri_air_roi_mask

    image_np = sitk.GetArrayFromImage(sitk.Cast(sitk_image, sitk.sitkFloat32)).astype(
        np.float32
    )
    airway = _mri_air_roi_mask(
        image_np,
        cavity_masks=None,
        min_component_voxels=min_component_voxels,
    )

    sitk_airway = sitk.GetImageFromArray(airway.astype(np.uint8))
    sitk_airway.CopyInformation(sitk_image)
    return sitk_airway


def _keep_mri_vessel_components_crossing_surface_band(
    vessels_np,
    lungs_np,
    sitk_lungs=None,
    surface_radius=MRI_SURFACE_COMPONENT_BAND_RADIUS,
):
    if sitk_lungs is not None:
        eroded = sitk.BinaryErode(
            sitk.Cast(sitk_lungs > 0, sitk.sitkUInt8),
            kernelRadius=(int(surface_radius),) * 3,
            foregroundValue=1,
        )
        eroded_np = sitk.GetArrayFromImage(eroded) > 0
    else:
        eroded_np = ndi.binary_erosion(
            lungs_np,
            structure=_ball(int(surface_radius)),
            iterations=1,
            border_value=0,
        )
    surface_band = lungs_np & ~eroded_np
    labels, n_labels = ndi.label(vessels_np, structure=np.ones((3, 3, 3), dtype=bool))
    if n_labels == 0:
        return vessels_np

    interior_vessels = vessels_np & ~surface_band
    labels_crossing_interior = np.unique(labels[interior_vessels])
    labels_crossing_interior = labels_crossing_interior[labels_crossing_interior != 0]
    return np.isin(labels, labels_crossing_interior)


def _vesselness_filter_mri_with_context_cleanup(
    sitk_image,
    sitk_lungs,
    sitk_airway_lumen=None,
):
    sitk_lungs = sitk.Cast(sitk_lungs > 0, sitk.sitkUInt8)
    sitk_context = sitk.BinaryDilate(
        sitk_lungs,
        kernelRadius=(MRI_FRANGI_CONTEXT_RADIUS,) * 3,
        foregroundValue=1,
    )
    _, sitk_vesselness_context = _vesselness_filter_mri(sitk_image, sitk_context)

    lungs_np = sitk.GetArrayFromImage(sitk_lungs) > 0
    vesselness_np = sitk.GetArrayFromImage(sitk_vesselness_context).astype(np.float32)
    vesselness_np = np.where(lungs_np, vesselness_np, 0.0)
    vessels_np = _threshold_mri_vesselness_inside_lungs(vesselness_np, lungs_np)

    if sitk_airway_lumen is None:
        sitk_airway_lumen = _rough_mri_airway_lumen_mask(sitk_image, sitk_lungs)
    airway_np = sitk.GetArrayFromImage(sitk_airway_lumen) > 0
    airway_band = ndi.binary_dilation(
        airway_np,
        structure=_ball(MRI_AIRWAY_SUPPRESSION_RADIUS),
        iterations=1,
    ) & lungs_np
    vessels_np = vessels_np & ~airway_band

    vessels_np = _keep_mri_vessel_components_crossing_surface_band(
        vessels_np,
        lungs_np,
        sitk_lungs=sitk_lungs,
        surface_radius=MRI_SURFACE_COMPONENT_BAND_RADIUS,
    )
    vesselness_np = np.where(vessels_np, vesselness_np, 0.0)

    sitk_vessels = sitk.GetImageFromArray(vessels_np.astype(np.uint8))
    sitk_vessels.CopyInformation(sitk_image)
    sitk_vesselness = sitk.GetImageFromArray(vesselness_np.astype(np.float32))
    sitk_vesselness.CopyInformation(sitk_image)
    return sitk_vessels, sitk_vesselness


def vesselness_filter(sitk_image, sitk_lungs, mode="ct"):
    if mode == "ct":
        return _vesselness_filter_ct(sitk_image, sitk_lungs)
    if mode == "mri":
        return _vesselness_filter_mri(sitk_image, sitk_lungs)
    raise ValueError(f"Unknown mode={mode!r}. Expected 'ct' or 'mri'.")


def vesselness_to_speed(
    sitk_vesselness, sitk_mask=None, min_speed=1e-3, max_speed=1.0, power=2.0
):
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


def _connect_to_mediastinum(sitk_vessels, sitk_lungs, sitk_mediastinum):
    lungs = sitk.GetArrayFromImage(sitk_lungs)
    vessels = sitk.GetArrayFromImage(sitk_vessels)
    mediastinum = sitk.GetArrayFromImage(sitk_mediastinum)

    blood_system = np.logical_or(mediastinum, vessels).astype(int)
    blood_regions = regionprops(blood_system)
    if len(blood_regions) == 0:
        first_region_mask = np.zeros_like(blood_system, dtype=np.uint8)
    else:
        sorted_regions = sorted(blood_regions, key=lambda x: x.area)
        first_region_mask = np.zeros_like(blood_system, dtype=int)
        first_region_mask[tuple(sorted_regions[0].coords.T)] = 1
        first_region_mask[mediastinum > 1] = 0
        first_region_mask[first_region_mask > 0] = lungs[first_region_mask > 0]
        first_region_mask[lungs == 0] = 0
        first_region_mask[first_region_mask > 0] = 1

    vessels_connected = sitk.GetImageFromArray(first_region_mask.astype(np.uint8))
    vessels_connected.CopyInformation(sitk_vessels)
    return sitk.Cast(vessels_connected, sitk.sitkUInt8)


def _remove_ct_blobs(sitk_vessels, sitk_vesselness, sitk_lungs):
    from bronco.segmentation.blobs_segmentation import blobs_segmentation
    from bronco.segmentation.airways_segmentation import fast_marching

    sitk_speed = vesselness_to_speed(sitk_vesselness, sitk_mask=sitk_lungs, power=2.0)
    vessels_connected = sitk.Cast(sitk_vessels, sitk.sitkUInt8)
    vessel_mask = sitk.GetArrayFromImage(vessels_connected)
    if np.count_nonzero(vessel_mask) == 0:
        return vessels_connected

    blobs = blobs_segmentation(vessels_connected, sitk_lungs, sigma_max=12, steps=10)
    radius = (3,) * blobs.GetDimension()
    blobs = sitk.BinaryMorphologicalClosing(blobs, radius)

    blob_arr = sitk.GetArrayFromImage(blobs) > 0
    list_points = np.argwhere(blob_arr > 0).tolist()
    if len(list_points) == 0:
        return vessels_connected

    expanded_blobs_fm = fast_marching(
        sitk_speed, seed_point=list_points, stopping_value=1
    )
    expanded_blobs_mask = sitk.GetImageFromArray(
        (expanded_blobs_fm > 0).astype(np.uint8)
    )
    expanded_blobs_mask.CopyInformation(sitk_vessels)

    labeled_exp = sitk.ConnectedComponent(expanded_blobs_mask)
    labeled_np = sitk.GetArrayFromImage(labeled_exp).astype(np.int32)
    vessels_conn_np = sitk.GetArrayFromImage(vessels_connected) > 0

    labels_flat = labeled_np.ravel().astype(np.int32)
    outside_mask_flat = (~vessels_conn_np).ravel().astype(np.uint8)
    if labels_flat.size == 0:
        filtered_np = np.zeros_like(labeled_np, dtype=np.uint8)
    else:
        max_label = labels_flat.max()
        outside_counts = np.bincount(
            labels_flat, weights=outside_mask_flat, minlength=max_label + 1
        )
        keep_labels = np.nonzero(outside_counts > 0)[0]
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size == 0:
            filtered_np = np.zeros_like(labeled_np, dtype=np.uint8)
        else:
            filtered_np = np.isin(labeled_np, keep_labels).astype(np.uint8)

    filtered_expanded_blobs_mask = sitk.GetImageFromArray(filtered_np.astype(np.uint8))
    filtered_expanded_blobs_mask.CopyInformation(expanded_blobs_mask)

    return sitk.And(vessels_connected, sitk.Not(filtered_expanded_blobs_mask))


def vessel_segmentation(
    sitk_image,
    sitk_lungs,
    sitk_mediastinum=None,
    mode="ct",
    check_mediastinum_connectivity=False,
    sitk_airway_lumen=None,
):
    if mode == "mri":
        sitk_vessels, sitk_vesselness = _vesselness_filter_mri_with_context_cleanup(
            sitk_image,
            sitk_lungs,
            sitk_airway_lumen=sitk_airway_lumen,
        )
    else:
        sitk_vessels, sitk_vesselness = vesselness_filter(
            sitk_image,
            sitk_lungs,
            mode=mode,
        )

    needs_connectivity = mode == "ct" or check_mediastinum_connectivity
    if needs_connectivity:
        if sitk_mediastinum is None:
            from bronco.segmentation import mediastinum_segmentation

            sitk_mediastinum = mediastinum_segmentation(sitk_lungs)
        sitk_vessels = _connect_to_mediastinum(
            sitk_vessels,
            sitk_lungs,
            sitk_mediastinum,
        )

    if mode == "ct":
        return _remove_ct_blobs(sitk_vessels, sitk_vesselness, sitk_lungs)
    if mode == "mri":
        return sitk_vessels
    raise ValueError(f"Unknown mode={mode!r}. Expected 'ct' or 'mri'.")
