import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image
from skimage.segmentation import random_walker
from sklearn.mixture import GaussianMixture


MRI_GMM_RANDOM_STATE = 17
MRI_GMM_MAX_TRAIN_VOXELS = 300_000
MRI_GMM_REG_COVAR = 1e-5
MRI_GMM_N_INIT = 5
MRI_GMM_MAX_ITER = 300
MRI_RANDOM_WALKER_BETA = 10.0
MRI_HILUM_DISTANCE_VOXELS = 6.0
MRI_HULL_DISTANCE_VOXELS = 10.0
MRI_AIR_ROI_HISTOGRAM_BINS = 512
MRI_AIR_ROI_HISTOGRAM_SMOOTH_SIGMA = 2.0


def _ball(radius):
    coords = np.ogrid[
        -radius : radius + 1,
        -radius : radius + 1,
        -radius : radius + 1,
    ]
    return coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2 <= radius**2


def _label_largest(mask_array, n_keep=2):
    labels, n_labels = ndi.label(mask_array, structure=np.ones((3, 3, 3), dtype=bool))
    out = np.zeros(mask_array.shape, dtype=bool)
    label_out = np.zeros(mask_array.shape, dtype=np.uint8)
    if n_labels == 0:
        return out, label_out

    counts = np.bincount(labels.ravel())
    keep = [i for i in np.argsort(counts)[::-1] if i != 0][:n_keep]
    out = np.isin(labels, keep)
    for display_label, component_label in enumerate(keep, start=1):
        label_out[labels == component_label] = display_label
    return out, label_out


def _label_largest_default_connectivity(mask_array, n_keep=2):
    labels, n_labels = ndi.label(mask_array)
    out = np.zeros(mask_array.shape, dtype=bool)
    if n_labels == 0:
        return out

    counts = np.bincount(labels.ravel())
    keep = [i for i in np.argsort(counts)[::-1] if i != 0][:n_keep]
    return np.isin(labels, keep)


def _crop_slices(mask_array, pad=8):
    coords = np.argwhere(mask_array)
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, mask_array.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _normalize_mri_for_random_walker(image_array, crop):
    data = np.log1p(np.clip(image_array[crop], a_min=0, a_max=None)).astype(np.float32)
    finite = data[np.isfinite(data)]
    lo, hi = np.percentile(finite, [1, 99.5])
    data = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    data = ndi.gaussian_filter(data, sigma=0.65)
    return data.astype(np.float32)


def _gmm_intensity_components(image_array, max_train_voxels=MRI_GMM_MAX_TRAIN_VOXELS):
    # Match the original MRI experiment chain, which fitted the GMM on nibabel
    # array ordering (x, y, z). The fixed random subsample depends on flatten
    # order, so fitting in SimpleITK's z, y, x order gives a slightly different
    # body mask.
    image_xyz = np.transpose(image_array, (2, 1, 0))
    foreground = np.isfinite(image_xyz) & (image_xyz > 0)
    data = np.log1p(np.clip(image_xyz[foreground], a_min=0, a_max=None))
    data = data.reshape(-1, 1).astype(np.float32)
    if data.shape[0] == 0:
        return np.zeros(image_array.shape, dtype=np.int16) - 1

    rng = np.random.default_rng(MRI_GMM_RANDOM_STATE)
    train = data
    if data.shape[0] > max_train_voxels:
        train = data[rng.choice(data.shape[0], size=max_train_voxels, replace=False)]

    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        reg_covar=MRI_GMM_REG_COVAR,
        random_state=MRI_GMM_RANDOM_STATE,
        n_init=MRI_GMM_N_INIT,
        max_iter=MRI_GMM_MAX_ITER,
    )
    model.fit(train)

    raw_labels = np.zeros(image_xyz.shape, dtype=np.int16) - 1
    raw_labels[foreground] = model.predict(
        np.log1p(np.clip(image_xyz[foreground], a_min=0, a_max=None))
        .reshape(-1, 1)
        .astype(np.float32)
    )
    labels = np.zeros(image_xyz.shape, dtype=np.int16) - 1
    for sorted_label, raw_label in enumerate(np.argsort(model.means_.reshape(-1))):
        labels[raw_labels == int(raw_label)] = int(sorted_label)
    return np.transpose(labels, (2, 1, 0))


def _mri_body_cavity_masks(image_array):
    labels = _gmm_intensity_components(image_array)
    body = labels == 1
    body_xyz = np.transpose(body, (2, 1, 0))
    body_largest_xyz = _label_largest_default_connectivity(body_xyz, n_keep=1)
    body_closed_xyz = ndi.binary_closing(
        body_largest_xyz,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=1,
    )
    body_filled_xyz = ndi.binary_fill_holes(body_closed_xyz)
    lung_base_xyz = _label_largest_default_connectivity(
        body_filled_xyz & ~body_closed_xyz,
        n_keep=2,
    )
    return {
        "body": np.transpose(body_closed_xyz, (2, 1, 0)),
        "body_envelope": np.transpose(body_filled_xyz, (2, 1, 0)),
        "body_closed_xyz": body_closed_xyz,
        "body_filled_xyz": body_filled_xyz,
        "lung_base": np.transpose(lung_base_xyz, (2, 1, 0)),
        "lung_base_xyz": lung_base_xyz,
        "air_component": labels == 0,
    }


def _equal_density_cutoff(values_inside, values_outside):
    values_inside = values_inside[np.isfinite(values_inside) & (values_inside > 0)]
    values_outside = values_outside[np.isfinite(values_outside) & (values_outside > 0)]
    if values_inside.size == 0 or values_outside.size == 0:
        return None

    lo = max(np.percentile(values_inside, 0.1), np.percentile(values_outside, 0.1))
    hi = min(np.percentile(values_inside, 99.9), np.percentile(values_outside, 99.9))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    edges = np.linspace(float(lo), float(hi), MRI_AIR_ROI_HISTOGRAM_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist_inside, _ = np.histogram(values_inside, bins=edges, density=True)
    hist_outside, _ = np.histogram(values_outside, bins=edges, density=True)
    hist_inside = ndi.gaussian_filter1d(
        hist_inside.astype(float),
        sigma=MRI_AIR_ROI_HISTOGRAM_SMOOTH_SIGMA,
    )
    hist_outside = ndi.gaussian_filter1d(
        hist_outside.astype(float),
        sigma=MRI_AIR_ROI_HISTOGRAM_SMOOTH_SIGMA,
    )
    diff = hist_inside - hist_outside
    crossings = np.flatnonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)
    if crossings.size == 0:
        return None

    inside_median = float(np.median(values_inside))
    outside_median = float(np.median(values_outside))
    target = (inside_median + outside_median) / 2.0
    candidates = []
    for index in crossings:
        x0 = centers[index]
        x1 = centers[index + 1]
        y0 = diff[index]
        y1 = diff[index + 1]
        if y1 == y0:
            candidates.append(float(x0))
        else:
            candidates.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return min(candidates, key=lambda value: abs(value - target))


def _mri_air_roi_mask(image_array, cavity_masks=None, min_component_voxels=15):
    if cavity_masks is None:
        cavity_masks = _mri_body_cavity_masks(image_array)

    air_component = cavity_masks["air_component"]
    body_envelope = cavity_masks["body_envelope"]
    lung_base = cavity_masks["lung_base"]
    cutoff = _equal_density_cutoff(
        image_array[air_component & body_envelope],
        image_array[air_component & ~body_envelope],
    )
    air_roi = lung_base & air_component
    if cutoff is not None:
        air_roi &= image_array < cutoff

    cc, n_labels = ndi.label(air_roi, structure=np.ones((3, 3, 3), dtype=bool))
    if n_labels > 0:
        counts = np.bincount(cc.ravel())
        keep = np.flatnonzero(counts >= int(min_component_voxels))
        keep = keep[keep != 0]
        air_roi = np.isin(cc, keep)
    return air_roi


def _fill_holes_2d(mask_array, axis):
    filled = np.zeros_like(mask_array, dtype=bool)
    if axis == 0:
        for i in range(mask_array.shape[0]):
            filled[i, :, :] = ndi.binary_fill_holes(mask_array[i, :, :])
    elif axis == 1:
        for i in range(mask_array.shape[1]):
            filled[:, i, :] = ndi.binary_fill_holes(mask_array[:, i, :])
    elif axis == 2:
        for i in range(mask_array.shape[2]):
            filled[:, :, i] = ndi.binary_fill_holes(mask_array[:, :, i])
    else:
        raise ValueError(axis)
    return filled


def _fill_2d_majority_per_component(mask_array):
    labels, n_labels = ndi.label(mask_array, structure=np.ones((3, 3, 3), dtype=bool))
    out = np.zeros_like(mask_array, dtype=bool)
    for component_id in range(1, n_labels + 1):
        component = labels == component_id
        votes = np.zeros(mask_array.shape, dtype=np.uint8)
        for axis in (0, 1, 2):
            votes += _fill_holes_2d(component, axis=axis).astype(np.uint8)
        out |= votes >= 2
    return _label_largest(out, n_keep=2)[0]


def _per_axial_slice_hull(mask_array):
    labels, n_labels = ndi.label(mask_array, structure=np.ones((3, 3, 3), dtype=bool))
    hull = np.zeros_like(mask_array, dtype=bool)
    for component_id in range(1, n_labels + 1):
        component = labels == component_id
        for z_index in range(component.shape[0]):
            if component[z_index].any():
                hull[z_index] |= convex_hull_image(component[z_index])
    return _label_largest(hull, n_keep=2)[0]


def _distance_limited_additions(seed, candidate, max_distance):
    dist = ndi.distance_transform_edt(~seed)
    return _label_largest(seed | (candidate & (dist <= float(max_distance))), n_keep=2)[0]


def _run_lung_random_walker(
    image_array,
    seed_lung,
    envelope,
    mediastinum,
    beta=MRI_RANDOM_WALKER_BETA,
    background_mode="outside_plus_far_mediastinum",
    hilar_distance=MRI_HILUM_DISTANCE_VOXELS,
):
    envelope = _label_largest(envelope | seed_lung, n_keep=2)[0]
    unknown = envelope & ~seed_lung
    if not unknown.any():
        return seed_lung.copy()

    roi = ndi.binary_dilation(envelope, structure=_ball(3), iterations=1) | seed_lung
    crop = _crop_slices(roi, pad=8)
    data = _normalize_mri_for_random_walker(image_array, crop)
    seed = seed_lung[crop]
    env = envelope[crop]
    med = mediastinum[crop]
    markers = np.zeros(seed.shape, dtype=np.int32)
    markers[seed] = 2

    if background_mode == "outside_plus_strong_mediastinum":
        background = (~env) | ndi.binary_dilation(med, structure=_ball(3), iterations=1)
    elif background_mode == "outside_plus_far_mediastinum":
        dist_to_seed = ndi.distance_transform_edt(~seed)
        far_mediastinum = med & (dist_to_seed > float(hilar_distance))
        background = (~env) | far_mediastinum
    elif background_mode == "outside_only":
        background = ~env
    else:
        raise ValueError(background_mode)

    markers[background & ~seed] = 1
    if not np.any(markers == 1) or not np.any(markers == 2):
        return seed_lung.copy()

    labels = random_walker(
        data,
        markers,
        beta=float(beta),
        mode="cg_j",
        tol=1e-3,
        prob_tol=1e-3,
    )
    accepted_crop = unknown[crop] & (labels == 2)
    accepted = np.zeros_like(seed_lung, dtype=bool)
    accepted[crop] = accepted_crop
    return _label_largest(seed_lung | accepted, n_keep=2)[0]


def _lungs_segmentation_mri(sitk_image, binary=True, return_air_roi=False):
    image_array = sitk.GetArrayFromImage(sitk.Cast(sitk_image, sitk.sitkFloat32)).astype(
        np.float32
    )

    cavity_masks = _mri_body_cavity_masks(image_array)
    lung_base_xyz = cavity_masks["lung_base_xyz"]
    lung_base = cavity_masks["lung_base"]
    body_filled_xyz = cavity_masks["body_filled_xyz"]
    lung_majority_xyz = _fill_2d_majority_per_component(lung_base_xyz) & body_filled_xyz
    lung_majority_xyz = _label_largest(lung_majority_xyz, n_keep=2)[0]
    lung_majority = np.transpose(lung_majority_xyz, (2, 1, 0))

    from bronco.segmentation.mediastinum_segmentation import mediastinum_segmentation

    sitk_base = sitk.GetImageFromArray(lung_base.astype(np.uint8))
    sitk_base.CopyInformation(sitk_image)
    mediastinum = sitk.GetArrayFromImage(mediastinum_segmentation(sitk_base)) > 0
    hull_axial = _per_axial_slice_hull(lung_majority)
    hull_dist10 = _distance_limited_additions(
        lung_majority,
        hull_axial,
        max_distance=MRI_HULL_DISTANCE_VOXELS,
    )
    mediastinum_r1 = ndi.binary_dilation(mediastinum, structure=_ball(1), iterations=1)
    hull_not_mediastinum = _label_largest(
        lung_majority | ((hull_dist10 & ~lung_majority) & ~mediastinum_r1),
        n_keep=2,
    )[0]

    current_rw = _run_lung_random_walker(
        image_array=image_array,
        seed_lung=lung_majority,
        envelope=hull_not_mediastinum,
        mediastinum=mediastinum,
        beta=MRI_RANDOM_WALKER_BETA,
        background_mode="outside_plus_strong_mediastinum",
    )

    dist_to_majority = ndi.distance_transform_edt(~lung_majority)
    near_hilum = mediastinum & (dist_to_majority <= MRI_HILUM_DISTANCE_VOXELS)
    envelope_hilum = hull_not_mediastinum | (hull_dist10 & near_hilum)
    lungs = _run_lung_random_walker(
        image_array=image_array,
        seed_lung=current_rw,
        envelope=envelope_hilum,
        mediastinum=mediastinum,
        beta=MRI_RANDOM_WALKER_BETA,
        background_mode="outside_plus_far_mediastinum",
        hilar_distance=MRI_HILUM_DISTANCE_VOXELS,
    )

    if binary:
        output = lungs.astype(np.uint8)
    else:
        output = _label_largest(lungs, n_keep=2)[1]
    sitk_lungs = sitk.GetImageFromArray(output)
    sitk_lungs.CopyInformation(sitk_image)
    if not return_air_roi:
        return sitk_lungs

    air_roi = _mri_air_roi_mask(image_array, cavity_masks=cavity_masks)
    sitk_air_roi = sitk.GetImageFromArray(air_roi.astype(np.uint8))
    sitk_air_roi.CopyInformation(sitk_image)
    return sitk_lungs, sitk_air_roi


def lungs_segmentation(
    sitk_image,
    config=None,
    binary=True,
    gpu_id=None,
    mode="ct",
    return_air_roi=False,
):
    if config is not None:
        mode = config.get("mode", config.get("image_mode", mode))

    if mode == "mri":
        return _lungs_segmentation_mri(
            sitk_image,
            binary=binary,
            return_air_roi=return_air_roi,
        )

    from lungmask import mask

    if config is not None:
        model = config["model_lungs_segmentation"]
    else:
        model = mask.get_model("R231")
    lungs = mask.apply(sitk_image, model, batch_size=5)
    if binary:
        _min = np.min(lungs)
        lungs[lungs > _min] = 1
        lungs[lungs == _min] = 0
        lungs = np.array(lungs, np.uint8)
    sitk_lungs = sitk.GetImageFromArray(lungs)
    sitk_lungs.CopyInformation(sitk_image)
    if return_air_roi:
        return sitk_lungs, None
    return sitk_lungs
