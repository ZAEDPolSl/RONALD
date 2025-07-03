import itk
import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops


from bronco.external.sitk2itk import (
    ConvertItkImageToSimpleItkImage,
    ConvertSimpleItkImageToItkImage,
)
from bronco.segmentation.blobs_segmentation import blobs_segmentation
from bronco.modelling.prepare_graph import prepare_graph

def endblobs(vessel_mask, blob_mask, mediastinum_mask):
    blood_system = sitk.Or(vessel_mask, mediastinum_mask)
    airways_graph = prepare_graph(blood_system)
    end_nodes = [node for node, degree in airways_graph.degree() if degree == 1]

    coords = np.array([airways_graph.nodes[node]['o'] for node in end_nodes])
    coords_int = np.round(coords).astype(int)
    zs = coords_int[:, 2]
    ys = coords_int[:, 1]
    xs = coords_int[:, 0]

    mediastinum = sitk.GetArrayFromImage(mediastinum_mask)
    zs = np.clip(zs, 0, mediastinum.shape[0] - 1)
    ys = np.clip(ys, 0, mediastinum.shape[1] - 1)
    xs = np.clip(xs, 0, mediastinum.shape[2] - 1)
    inside_mask = mediastinum[xs, ys, zs]
    outside_mask_indices = np.where(~inside_mask)[0]
    zs_out = zs[outside_mask_indices]
    ys_out = ys[outside_mask_indices]
    xs_out = xs[outside_mask_indices]

    labeled_mask = sitk.ConnectedComponent(blob_mask)
    labeled_np = sitk.GetArrayFromImage(labeled_mask)

    neighborhood = 2
    shape_x, shape_y, shape_z = labeled_np.shape
    offset_range = np.arange(-neighborhood, neighborhood + 1)
    dx, dy, dz = np.meshgrid(offset_range, offset_range, offset_range, indexing='ij')

    # Flatten offset arrays
    dz = dz.ravel()
    dy = dy.ravel()
    dx = dx.ravel()

    # Repeat endpoint coords for each offset
    z_neigh = zs[:, None] + dz[None, :]
    y_neigh = ys[:, None] + dy[None, :]
    x_neigh = xs[:, None] + dx[None, :]

    # Clip to valid indices
    z_neigh = np.clip(z_neigh, 0, shape_z - 1)
    y_neigh = np.clip(y_neigh, 0, shape_y - 1)
    x_neigh = np.clip(x_neigh, 0, shape_x - 1)

    # Now extract all labels in neighborhoods
    labels_in_neigh = labeled_np[x_neigh, y_neigh, z_neigh]  # shape (num_points, neighborhood_size)

    # Flatten and get unique labels > 0
    labels_to_remove = np.unique(labels_in_neigh)
    labels_to_remove = labels_to_remove[labels_to_remove > 0]
    mask_remove = np.isin(labeled_np, labels_to_remove).astype(np.uint8)

    output_mask = sitk.GetImageFromArray(mask_remove)
    output_mask.CopyInformation(blob_mask)
    radius = (1,) * output_mask.GetDimension()
    output_mask = sitk.BinaryDilate(
        output_mask,
        radius)
    return output_mask


def vesselness_filter(sitk_image, sitk_lungs):
    sigma = 1.0
    alpha1 = 0.5
    alpha2 = 2.0

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

    multiply_filter = itk.MultiplyImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.F, 3], itk.Image[itk.F, 3]
    ].New()
    multiply_filter.SetInput1(itk_output)
    multiply_filter.SetInput2(itk_lungs)
    multiply_filter.Update()
    output_image = multiply_filter.GetOutput()

    threshold_filter = itk.BinaryThresholdImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.UC, 3]
    ].New()
    threshold_filter.SetInput(output_image)
    threshold_filter.SetLowerThreshold(30)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.SetInsideValue(255)
    threshold_filter.Update()
    output_image = threshold_filter.GetOutput()
    # Convert Hessian ITK image back to SimpleITK
    direction = float_sitk.GetDirection()
    sitk_vessels = ConvertItkImageToSimpleItkImage(output_image, 8, direction)
    return sitk_vessels


def vessel_segmentation(
    sitk_image, sitk_lungs, sitk_lobes=None, sitk_mediastinum=None, binary=True
):
    sitk_vessels = vesselness_filter(sitk_image, sitk_lungs)
    if sitk_mediastinum is None:
        from bronco.segmentation import mediastinum_segmentation

        sitk_mediastinum = mediastinum_segmentation(sitk_lungs)
    if sitk_lobes is None:
        from bronco.segmentation import lobe_segmentation

        sitk_lobes = lobe_segmentation(sitk_image)

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

    blobs = blobs_segmentation(vessels_connected, sitk_lungs)
    radius = (2,) * blobs.GetDimension()
    blobs = sitk.BinaryMorphologicalClosing(
        blobs,
        radius)

    final_blobs = endblobs(vessels_connected, blobs, sitk_mediastinum)
    blobs = sitk.Or(blobs, final_blobs)

    vessels_final = sitk.Subtract(vessels_connected, blobs)
    vessels_final = sitk.BinaryThreshold(vessels_final,
                                         lowerThreshold=1,
                                         upperThreshold=255,
                                         insideValue=255,
                                         outsideValue=0)

    return vessels_final
