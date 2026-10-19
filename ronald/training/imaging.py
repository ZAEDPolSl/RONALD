"""Physical-space I/O, thoracic bounding boxes, and bounded nodule region growing."""
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def read_image(path, series_uid=None):
    path = Path(path)
    if path.is_dir():
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or []
        if series_uid is None and len(ids) != 1:
            raise ValueError(f'{path}: expected one DICOM series, found {len(ids)}; set series_uid')
        if series_uid is not None and series_uid not in ids:
            raise ValueError(f'{path}: series_uid not found')
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_uid or ids[0]))
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f'{path}: expected scalar 3D image')
    return image


def binary_on_reference(path, reference, allow_instance_labels=False):
    image = read_image(path)
    values = np.unique(sitk.GetArrayViewFromImage(image))
    valid_values = (np.all(values >= 0) and np.all(values == np.floor(values))) if allow_instance_labels else np.isin(values, [0, 1, 255]).all()
    if not np.isfinite(values).all() or not valid_values:
        raise ValueError(f'{path}: expected binary 0/1 or 0/255 mask, not intensity/label map')
    mask = sitk.Resample(image > 0, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(mask).astype(bool)


def reference_grid(image, spacing_xyz):
    image = sitk.DICOMOrient(image, 'LPS')
    size = np.ceil((np.array(image.GetSize()) - 1) * image.GetSpacing() / np.array(spacing_xyz)).astype(int) + 1
    ref = sitk.Image([int(v) for v in size], sitk.sitkFloat32)
    ref.SetSpacing(tuple(spacing_xyz))
    ref.SetOrigin(image.GetOrigin())
    ref.SetDirection(image.GetDirection())
    return ref


def prepare_image(image, lungs_path, mediastinum_path, spacing_xyz, margin_mm, hu):
    ref = reference_grid(image, spacing_xyz)
    lungs = binary_on_reference(lungs_path, ref)
    if not lungs.any():
        raise ValueError('Lung mask is empty or outside CT physical extent')
    roi = lungs.copy()
    if mediastinum_path:
        mediastinum = binary_on_reference(mediastinum_path, ref)
        if not mediastinum.any():
            raise ValueError('Supplied mediastinum mask is empty or outside CT')
        roi |= mediastinum
    # A bounding box encloses BOTH lungs and the intervening mediastinum.
    # Do not multiply CT by a lung mask: that would erase central structures/nodules.
    coords = np.where(roi)
    margin = np.ceil(margin_mm / np.array(spacing_xyz)[::-1]).astype(int)
    start = np.maximum([a.min() for a in coords] - margin, 0)
    stop = np.minimum(np.array([a.max() for a in coords]) + margin + 1, roi.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    resampled = sitk.Resample(image, ref, sitk.Transform(), sitk.sitkLinear, -1024, sitk.sitkFloat32)
    ct = sitk.GetArrayFromImage(resampled)[box]
    if not np.isfinite(ct).all():
        raise ValueError('Nonfinite CT intensities')
    normalized = (np.clip(ct, *hu) - hu[0]) / (hu[1] - hu[0])
    return normalized.astype(np.float32), ref, box


def grow_nodules(image, annotations, lower_hu=-700., upper_hu=400., tolerance_hu=250., radius_scale=1.25):
    """Conservative heuristic, NOT a validated nodule segmentation.

    LUNA coordinates are physical XYZ mm. Work in a bounded local ROI; reject
    components reaching its spherical boundary rather than forcing vessels into BG.
    Failed seeds never silently become positive spherical masks.
    """
    ct = sitk.GetArrayFromImage(image)
    spacing = np.array(image.GetSpacing())[::-1]
    result = np.zeros(ct.shape, dtype=np.uint8)
    audit = []
    for row in annotations:
        xyz = [float(row[k]) for k in ('coordX', 'coordY', 'coordZ')]
        center = np.array(image.TransformPhysicalPointToContinuousIndex(xyz))[::-1]
        diameter = float(row['diameter_mm'])
        record = dict(center_xyz_mm=xyz, diameter_mm=diameter, accepted=False)
        if not np.isfinite(center).all() or not np.isfinite(diameter) or diameter <= 0:
            raise ValueError('Invalid LUNA annotation coordinate or diameter')
        if np.any(center < 0) or np.any(center > np.array(ct.shape) - 1):
            record['reason'] = 'seed_outside_ct'
            audit.append(record)
            continue
        radius = max(diameter * 0.5 * radius_scale, float(spacing.max()) * 2)
        lo = np.maximum(np.floor(center - radius / spacing - 1).astype(int), 0)
        hi = np.minimum(np.ceil(center + radius / spacing + 2).astype(int), ct.shape)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        patch = ct[box]
        grid = np.ogrid[tuple(slice(0, n) for n in patch.shape)]
        distance2 = sum(((g + lo[i] - center[i]) * spacing[i]) ** 2 for i, g in enumerate(grid))
        seed_zone = distance2 <= max(1.5, float(spacing.max())) ** 2
        candidates = seed_zone & (patch >= lower_hu) & (patch <= upper_hu)
        if not candidates.any():
            record['reason'] = 'no_eligible_seed'
            audit.append(record)
            continue
        seed = np.unravel_index(np.argmin(np.where(candidates, distance2, np.inf)), patch.shape)
        seed_hu = float(patch[seed])
        sphere = distance2 <= radius ** 2
        eligible = sphere & (patch >= lower_hu) & (patch <= upper_hu) & (abs(patch - seed_hu) <= tolerance_hu)
        components, _ = ndi.label(eligible)
        grown = components == components[seed]
        boundary = sphere & ~ndi.binary_erosion(sphere)
        if (grown & boundary).any() or any(grown.take(i, axis=a).any() for a in range(3) for i in (0, -1)):
            record['reason'] = 'reaches_roi_boundary'
        elif grown.sum() * np.prod(spacing) < 3:
            record['reason'] = 'too_small'
        else:
            result[box] |= grown.astype(np.uint8)
            record.update(accepted=True, voxels=int(grown.sum()), volume_mm3=float(grown.sum() * np.prod(spacing)))
        audit.append(record)
    mask = sitk.GetImageFromArray(result)
    mask.CopyInformation(image)
    return mask, audit
