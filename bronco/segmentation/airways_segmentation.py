import os
import numpy as np
from tqdm import tqdm
from copy import copy
import SimpleITK as sitk
from skimage.filters import sato
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes

from bronco.utils import display, plot_sum_image
from bronco.preprocessing import run_thresholding
from bronco.io_local.ImageInstance import ImageInstance
from bronco.segmentation.trachea_segmentation import trachea_main_bronchus_segmentation
from bronco.preprocessing.connected_components import convex_hull_3d, find_largest_connected_component, \
    find_most_similar_connected_component
from bronco.utils import plot_sum_image, plot_sum_subplots_image

# from priv.image_viewer import image_viewer

path_temp = r"D:\msocha\PHD\2024-03-27 - test terminal bronchi filling\Abramik_Miroslaw"


def fast_marching(sitk_init, seed_point, stopping_value=60):
    if type(seed_point) is not list:
        seed_points = [seed_point]
    else:
        seed_points = seed_point
    # Apply level set segmentation
    fastMarching = sitk.FastMarchingImageFilter()
    for seed_point in seed_points:
        seed_point = [int(l) for l in list(seed_point)]
        fastMarching.AddTrialPoint([seed_point[2], seed_point[1], seed_point[0]])  # COS SITK !!!!
    fastMarching.SetStoppingValue(stopping_value)
    sitk_fast_marching = fastMarching.Execute(sitk_init)

    sitk_fast_marching = sitk.Threshold(sitk.Clamp(sitk_fast_marching, lowerBound=0, upperBound=10000),
                                        lower=0, upper=1000, outsideValue=-1) + 1
    image_fm = sitk.GetArrayFromImage(sitk_fast_marching)
    return image_fm


def per_slice_hole_removal(sitk_image, sitk_go_back=None, base_point=None):
    image = sitk.GetArrayFromImage(sitk_image)
    if sitk_go_back is not None:
        image_rough_airways = sitk.GetArrayFromImage(sitk_go_back)
    else:
        image_rough_airways = None
    for i in tqdm(range(image.shape[0]), desc="Filling airways..."):
        image_slice = image[i]
        if np.sum(image_slice) == 0:
            continue
        _image_slice = binary_fill_holes(image_slice)
        image[i] = _image_slice
    _image = image.copy()
    '''if base_point is None:
        for i in tqdm(range(1, image.shape[0]), desc="Extending terminal bronchioles..."):
            if i > 0:
                _image[i] = find_shared_blobs(image[i - 1], image_rough_airways[i])
    else:
        # freedom = [0, 2]
        # want to move per axial slice down and sagittal right and left
        ranges = [list(range(base_point[0], image.shape[0])), list(range(base_point[2] + 1, 0, -1)),
                  list(range(base_point[2], image.shape[2]))]
        j = [-1, 1, -1]  # because in one case we are decrementing
        for r in ranges:
            for i in r:
                if i > 0:
                    _image[i] = find_shared_blobs(image[i + j[i]], image_rough_airways[i])'''
    sitk_result = sitk.GetImageFromArray(_image)
    sitk_result.CopyInformation(sitk_image)
    return sitk_result


def forming_terminal_bronchioles(bronchi, bronchi_overgrown, base_point=None):
    # check input data format
    was_sitk_bronchi = False
    if type(bronchi) is sitk.Image:
        _bronchi = copy(bronchi)
        bronchi = sitk.GetArrayFromImage(bronchi)
        was_sitk_bronchi = True
    if type(bronchi_overgrown) is sitk.Image:
        bronchi_overgrown = sitk.GetArrayFromImage(bronchi_overgrown)

    # get base point
    if base_point is None:
        base_point = np.array(bronchi.shape) // 2  # center is our base point

    bronchi_supplemented = bronchi.copy()
    # want to move up and down from centre per axial, sagittal and coronal
    ranges = [list(range(base_point[0], bronchi.shape[0])), list(range(base_point[0] + 1, 0, -1)),
              list(range(base_point[1], bronchi.shape[1])), list(range(base_point[1] + 1, 0, -1)),
              list(range(base_point[2], bronchi.shape[2])), list(range(base_point[2] + 1, 0, -1))]
    list_perv_indexes = [-1, 1, -1, 1, -1, 1]  # because in some case we are decrementing during iterations
    axis = [0, 0, 1, 1, 2, 2]
    for j, r, a in zip(list_perv_indexes, ranges, axis):
        for i in r:
            if i > 0:
                if a == 0:
                    bronchi_supplemented[i] = find_shared_blobs(bronchi[i + j],
                                                                bronchi_overgrown[i])
                elif a == 1:
                    bronchi_supplemented[:, i] = find_shared_blobs(bronchi[:, i + j],
                                                                   bronchi_overgrown[:, i])
                else:
                    bronchi_supplemented[..., i] = find_shared_blobs(bronchi[..., i + j],
                                                                     bronchi_overgrown[..., i])


    if was_sitk_bronchi:
        bronchi_supplemented = sitk.GetImageFromArray(bronchi_supplemented)
        bronchi_supplemented.CopyInformation(_bronchi)
    return bronchi_supplemented


def find_shared_blobs(slice_1, slice_2):
    # Label connected components in both arrays
    labeled_1, num_features_1 = label(slice_1, return_num=True)
    labeled_2, num_features_2 = label(slice_2, return_num=True)

    shared_blobs = []
    flag = False
    # Check for overlaps
    for label_1 in range(1, num_features_1 + 1):
        # Extract coordinates of blob in array A
        coords_1 = np.argwhere(labeled_1 == label_1)
        size_1 = len(coords_1)
        # Check if any of the coordinates overlap with blobs in array B
        for label_2 in range(1, num_features_2 + 1):
            flag = True
            coords_2 = np.argwhere(labeled_2 == label_2)
            size_2 = len(coords_2)
            if np.any(np.in1d(coords_1, coords_2).reshape(coords_1.shape).all(axis=1)) and size_2 <= size_1:
                shared_blobs.append(coords_1)
                # break
    if flag:
        slice_shared = coordinates_to_image(shared_blobs, slice_1.shape)
    else:
        slice_shared = slice_1
    return slice_shared


def coordinates_to_image(coordinates, shape):
    image = np.zeros(shape, dtype=int)
    for coords in coordinates:
        image[tuple(coords.T)] = 1
    return image


def airways_segmentation(sitk_image, sitk_lungs, thresholds=None, path_visualisations=None, verbose=1, **kwargs):
    # GMM if not supplied
    if thresholds is None:
        display("GMM...", verbose)
        _, thresholds = run_thresholding(sitk_image, sitk_lungs, return_thresholds=True)

    # segment trachea
    display("Trachea Segmentation...", verbose)
    sitk_trachea = trachea_main_bronchus_segmentation(sitk_image, sitk_lungs)

    # get body
    display("Body Segmentation...", verbose)
    sitk_lungs_convex_hull = convex_hull_3d(sitk_lungs)
    sitk_body = sitk.Mask(sitk_image, sitk_lungs_convex_hull, outsideValue=8000)
    body = sitk.GetArrayFromImage(sitk_body)

    # based on the body and thresholds get rough airways mask
    print(thresholds)
    rough_vessels = np.where(body >= thresholds[2], 1, 0)

    # prepare rough vessels for later
    sitk_vessels_rough = sitk.GetImageFromArray(rough_vessels)
    sitk_vessels_rough.CopyInformation(sitk_image)

    # prepare airways scaffolding
    display("1st speed image component preparation...", verbose)
    sitk_bbv_scaffolding = sitk.GetImageFromArray(rough_vessels)
    sitk_bbv_scaffolding.CopyInformation(sitk_image)
    sitk_bbv_scaffolding = sitk.Cast(sitk.Abs(sitk_bbv_scaffolding - 1), sitk.sitkInt16) * sitk.Cast(sitk_body < 8000,
                                                                                                     sitk.sitkInt16)
    sitk_bbv_scaffolding = find_largest_connected_component(sitk_bbv_scaffolding)

    # create first speed image for level set segmentation
    sitk_initialization = sitk.GradientMagnitude(sitk_bbv_scaffolding)
    sitk_initialization = sitk.Mask(sitk.BoundedReciprocal(sitk_initialization), sitk_bbv_scaffolding)

    # scaffolding expansion
    sitk_trachea_processed = sitk_trachea * sitk_bbv_scaffolding
    '''sitk_dilated_trachea = sitk.BinaryDilate(sitk_trachea_processed, kernelRadius=(15, 15, 15))
    bbv_scaffolding = sitk.GetArrayFromImage(sitk_bbv_scaffolding)
    lungs = sitk.GetArrayFromImage(sitk.BinaryErode(sitk_lungs, kernelRadius=(1, 1, 1)))
    dilated_trachea = sitk.GetArrayFromImage(sitk_dilated_trachea)
    dilated_trachea[lungs == 1] = 0
    # contour_trachea = dilated_trachea - sitk.GetArrayFromImage(sitk_trachea_processed)
    airways_4_sato = np.ones_like(body)
    airways_4_sato[lungs == 1] = bbv_scaffolding[lungs == 1]
    airways_4_sato[dilated_trachea == 1] = bbv_scaffolding[dilated_trachea == 1]
    # airways_4_sato = airways_4_sato - contour_trachea
    sitk_airways_4_sato = sitk.GetImageFromArray(airways_4_sato)
    sitk_airways_4_sato.CopyInformation(sitk_image)'''

    #TODO temp saving
    ii = kwargs.get('ii', ImageInstance())
    ii.write(sitk_trachea_processed, os.path.join(path_temp, 'trachea.nrrd'))

    # create second speed image for level set segmentation
    display("2st speed image component preparation...", verbose)
    bbv = sitk.GetArrayFromImage(sitk_bbv_scaffolding)  # prev sitk_bbv_scaffolding
    bbv_sato = sato(bbv, [2, 3, 5], black_ridges=False)
    bbv_sato = ((bbv_sato - bbv_sato.min()) / (bbv_sato.max() - bbv_sato.min()))
    bbv_sato[bbv_sato < 0.05] = 0
    sitk_bbv_sato = sitk.GetImageFromArray(bbv_sato)
    sitk_bbv_sato.CopyInformation(sitk_image)

    # create joined speed image for level set segmentation
    display("Joined speed image component preparation...", verbose)
    sitk_speed_image = sitk_bbv_sato * sitk_initialization

    # get air, lungs + airways
    display("Seed points preparation...", verbose)
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(sitk_image, sitk_lungs)
    _min = stats.GetMinimum(1)
    sitk_air = sitk.Mask(sitk_image, sitk_bbv_scaffolding, outsideValue=8000)
    sitk_air = sitk.Cast(sitk_air, sitk.sitkFloat32)

    # there are small intensity differences, exploit them
    _sitk_air = sitk.CurvatureAnisotropicDiffusion(sitk_air)
    sitk_air_value_image = _sitk_air * sitk.Cast(sitk_trachea, sitk.sitkFloat32)
    air_value_image = sitk.GetArrayFromImage(sitk_air_value_image)
    air_values = air_value_image[air_value_image < -500]
    # threshold based on the mean intensity of voxels in the trachea
    thr = np.mean(air_values) # + np.std(air_values)
    # get rough points of the clean air
    sitk_airways_points = sitk.Threshold(_sitk_air, lower=int(_min), upper=int(thr))
    sitk_airways_points = sitk.Cast(sitk_airways_points < 0, sitk.sitkInt16)
    image_airways_points = sitk.GetArrayFromImage(sitk_airways_points)
    list_points = np.argwhere(image_airways_points > 0).tolist()

    # level set segmentation
    display("Fast marching airways segmentation...", verbose)
    image_level_set = fast_marching(sitk_speed_image, seed_point=list_points, stopping_value=10)
    # image_level_set_20 = np.where(fast_marching(sitk_speed_image, seed_point=list_points, stopping_value=20) > 0, 1, 0)
    # image_level_set_30 = np.where(fast_marching(sitk_speed_image, seed_point=list_points, stopping_value=30) > 0, 1, 0)
    # image_level_set_40 = np.where(fast_marching(sitk_speed_image, seed_point=list_points, stopping_value=40) > 0, 1, 0)
    sitk_airways = sitk.GetImageFromArray(image_level_set)
    sitk_airways.CopyInformation(sitk_image)
    sitk_airways = sitk.Cast(sitk_airways > 0, sitk.sitkInt16)




    '''image_level_set_50 = fast_marching(sitk_speed_image, seed_point=list_points, stopping_value=50)
    sitk_level_set_50 = sitk.GetImageFromArray(image_level_set_50)
    sitk_level_set_50.CopyInformation(sitk_image)
    sitk_level_set_50 = sitk.Cast(sitk_level_set_50 > 0, sitk.sitkInt16)'''

    # find the segmented airways in all elements
    display("Airways selection...", verbose)
    sitk_airways, iou = find_most_similar_connected_component(sitk_airways, sitk_trachea)

    # airways walls extraction
    sitk_airways = sitk.Cast(sitk_airways, sitk.sitkUInt8)
    sitk_vessels = sitk.Cast(sitk_vessels_rough > 0, sitk.sitkUInt8)
    sitk_airways_dilated = sitk.BinaryDilate(sitk_airways, kernelRadius=(3, 3, 3))
    sitk_walls = sitk_airways_dilated * sitk_vessels
    sitk_walls_closed = sitk.BinaryMorphologicalClosing(sitk_walls, (3, 6, 6))
    sitk_walls_filled = per_slice_hole_removal(sitk_walls_closed, sitk_airways)

    # remove walls from the filled image
    sitk_walls_filled = sitk_walls_filled - sitk_walls

    # TODO: TEMOPRARY SAVING
    '''sitk_image_level_set_40 = sitk.GetImageFromArray(image_level_set_40)
    sitk_image_level_set_40.CopyInformation(sitk_image)
    ii.write(sitk_image_level_set_40, os.path.join(path_temp, "level_set_40.nrrd"))
    ii.write(sitk_walls_filled, os.path.join(path_temp, "airways_after_filling.nrrd"))'''

    # plot_sum_image(sitk_walls_filled)
    # sitk_walls_filled = forming_terminal_bronchioles(sitk_walls_filled, image_level_set_20)
    # plot_sum_image(sitk_walls_filled)
    # sitk_walls_filled = forming_terminal_bronchioles(sitk_walls_filled, image_level_set_30)
    # plot_sum_image(sitk_walls_filled)
    # sitk_walls_filled = forming_terminal_bronchioles(sitk_walls_filled, image_level_set_40)
    # plot_sum_image(sitk_walls_filled)
    # sitk_walls_filled = per_slice_hole_removal(sitk_walls_filled, sitk_level_set_50)
    if path_visualisations is not None:
        images = []
        image = sitk.GetArrayFromImage(sitk_walls_filled)
        images.append(image)

    sitk_airways = sitk_walls_filled * sitk_airways

    sitk_airways = sitk_airways + 2 * sitk_walls

    return sitk_airways, sitk_trachea


if __name__ == "__main__":
    import pickle
    from bronco.io_local.ImageInstance import ImageInstance
    path_image = r"X:\Wojtek\MIRASLC\Dane surowe\Moltest 1\NRRD\Abramik_Miroslaw.nrrd"
    path_lungs = r"D:\msocha\PHD\2024-03-05 - bronco test\lungs.nrrd"
    path_thresholds = r"D:\msocha\PHD\2024-03-05 - bronco test\gmm.pkl"

    ii = ImageInstance()
    with open(path_thresholds, 'rb') as file:
        thresholds = pickle.load(file)
    sitk_image = ii.read(path_image)
    sitk_lungs = ImageInstance().read(path_lungs)
    sitk_airways = airways_segmentation(sitk_image, sitk_lungs, thresholds)

    plot_sum_image(sitk.Cast(sitk_airways == 1, sitk.sitkInt16), category='airways')
    plot_sum_image(sitk.Cast(sitk_airways == 2, sitk.sitkInt16), category='walls')

    sitk_airways = sitk.Cast(sitk_airways, sitk.sitkUInt8)
    sitk_series_255 = sitk.Cast(sitk.RescaleIntensity(sitk.IntensityWindowing(sitk_image,
                                                                              windowMinimum=-1100,
                                                                              windowMaximum=-300)), sitk.sitkUInt8)
    sitk_overlay = sitk.LabelOverlay(sitk_series_255, sitk_airways)
    image_viewer.Execute(sitk_overlay)

    airways = sitk.GetArrayFromImage(sitk_airways)
    plot_sum_image(airways)

