import os
import pickle
import SimpleITK as sitk
import numpy as np

from bronco.io_local.ImageInstance import ImageInstance
from bronco.preprocessing.get_thresholds import run_thresholding
from bronco.segmentation.lungs_segmentation import lungs_segmentation
from bronco.segmentation.vessels_segmentation import vessels_segmentation
from bronco.segmentation.airways_segmentation import airways_segmentation
from bronco.utils import display, plot_sum_image, plot_sum_subplots_image


def pipeline(path_series, path_save=None, path_visualisations=None, cache=False, series_uid=None,
             study_id=None, **kwargs):
    """
        Pipeline of lungs segmentation followed by the bronchovascular bundel modeling.

        Parameters
        ----------
        path_series : (str) path to the folder with DICOM series or the NRRD file,
        path_save : (str) path to the folder where results are going to be stored,
        path_visualisations : (str) path to the visualisations folder,
        cache: (bool) whether to use cached files,
        kwargs :
            - verbose: (int) default 1, whether to show messages during processing (>0 shows messages),
            - retain_main_bronchi : (bool) default True, whether to retain main broncho - time consuming operation,
            - return_binary : (bool) default True, whether to return binary or hierarchically labelled mask,
            - save_all: (bool) default False, whether to save all intermediate series - gmm results, skeleton etc.

        Returns
        -------
        sitk_bronchovascular_bundle : (sitk.Image) image of the bronchovascular bundle,
        sitk_thresholds : (sitk.Image) image of the raw (before the hierarchical clustering) bronchovascular bundle
        """
    # cache paths handling
    if path_save is not None:
        path_lungs = os.path.join(path_save, "lungs.nrrd")
        path_thresholds = os.path.join(path_save, "gmm.pkl")
        path_thresholds_image = os.path.join(path_save, "gmm.nrrd")
        path_vessels = os.path.join(path_save, "vessels.nrrd")
        path_vessels_rough = os.path.join(path_save, "vessels_rough.nrrd")
    else:
        path_lungs = ""
        path_thresholds = ""
        path_thresholds_image = ""
        path_vessels = ""
        path_vessels_rough = ""

    # reading
    display("Reading...", kwargs.get('verbose', 1))
    ii = ImageInstance()
    sitk_series = ii.read(path_series, series_uid=series_uid)

    # TODO: Debug
    kwargs['ii'] = ii
    kwargs['save_all'] = True

    # lung segmentation
    path_lungs = kwargs.get('path_lungs', path_lungs)
    if (os.path.isfile(path_lungs) or os.path.isdir(path_lungs)) and cache:
        display("Loading lungs segmentation...", kwargs.get('verbose', 1))
        sitk_lungs = ImageInstance().read(path_lungs)
        # del kwargs['path_lungs']
    else:
        display("Lungs segmentation...", kwargs.get('verbose', 1))
        sitk_lungs = lungs_segmentation(sitk_image=sitk_series)
        if path_save is not None:
            path_save_lungs = os.path.join(path_save, 'lungs.nrrd')
            ii.write(sitk_lungs, path_save_lungs, description="Lungs segmentation", forced_mode='file')

    # GMM processing
    if os.path.isfile(path_thresholds) and cache:
        display("GMM reading from cache...", kwargs.get('verbose', 1))
        with open(path_thresholds, 'rb') as file:
            thresholds = pickle.load(file)
        sitk_thresholds = ImageInstance().read(path_thresholds_image)
    else:
        display("GMM...", kwargs.get('verbose', 1))
        sitk_thresholds, thresholds = run_thresholding(sitk_series, sitk_lungs, return_thresholds=True)
        if len(path_thresholds) > 0:
            # saving if cache is enabled
            ii.write(sitk_thresholds, path_thresholds_image, forced_mode='file')
            with open(path_thresholds, 'wb') as file:
                pickle.dump(thresholds, file)

    # vessels segmentation
    if (os.path.isfile(path_vessels) or os.path.isdir(path_vessels)) and cache and False:
        display("Vessels reading from cache...", kwargs.get('verbose', 1))
        sitk_vessels = ImageInstance().read(path_vessels)
        # sitk_rough_vessels = ImageInstance().read(path_vessels_rough)
    else:
        display("Vessels segmentation...", kwargs.get('verbose', 1))
        sitk_vessels, sitk_rough_vessels = vessels_segmentation(sitk_series, sitk_lungs, sitk_thresholds,
                                                                path_visualisations=path_visualisations,
                                                                stuid=study_id, path_save=path_save, **kwargs)
        if len(path_vessels) > 0:
            ii.write(sitk_vessels, path_vessels, forced_mode='file')
            # ii.write(sitk_rough_vessels, path_vessels_rough)
    if study_id is not None:
        plot_sum_image(sitk_vessels, path_visualisations, category='vessels', name=study_id)

    # airways segmentation
    display("Airways segmentation...", kwargs.get('verbose', 1))
    sitk_airways, sitk_trachea = airways_segmentation(sitk_series, sitk_lungs, thresholds, path_visualisations=path_visualisations,
                                                      **kwargs)

    # TODO: Saving trachea
    ii.write(sitk_trachea, os.path.join(path_save, "trachea.nrrd"), forced_mode='file')

    if study_id is not None:
        plot_sum_image(sitk.Cast(sitk_airways == 1, sitk.sitkUInt8), path_visualisations,
                       category='airways', name=study_id)
        plot_sum_image(sitk.Cast(sitk_airways == 2, sitk.sitkUInt8), path_visualisations,
                       category='walls', name=study_id)

    # combine all into one mask
    vessels = sitk.GetArrayFromImage(sitk_vessels)
    airways = sitk.GetArrayFromImage(sitk_airways)
    vessels[airways == 1] = 2  # adding airways mask
    vessels[airways == 2] = 3  # adding airways walls mask
    sitk_bronchovascular_bundle = sitk.GetImageFromArray(vessels)
    sitk_bronchovascular_bundle.CopyInformation(sitk_series)

    if path_save is not None:
        path_save_bbv = os.path.join(path_save, 'bronchovascular_bundle')
        # os.makedirs(path_save_bbv, exist_ok=True)
        if study_id is not None:
            path_save_bbv = path_save_bbv + f"_{study_id}"
        ii.write(sitk_bronchovascular_bundle, path_save_bbv, description="Bronchovascular bundle segmentation",
                 forced_mode='file')

    # TODO: DEBUG
    from priv.image_viewer import image_viewer
    sitk_bronchovascular_bundle = sitk.Cast(sitk_bronchovascular_bundle, sitk.sitkUInt8)
    sitk_series_255 = sitk.Cast(sitk.RescaleIntensity(sitk.IntensityWindowing(sitk_series,
                                                                              windowMinimum=-1100,
                                                                              windowMaximum=-300)), sitk.sitkUInt8)
    sitk_overlay = sitk.LabelOverlay(sitk_series_255, sitk_bronchovascular_bundle)
    image_viewer.Execute(sitk_overlay)

    image = sitk.GetArrayFromImage(sitk_overlay)
    axial = image.shape[0]
    list_idx_slices = list(np.linspace(int(axial * 0.05), int(axial * 0.95), 26).astype(int))
    list_images = [image[idx] for idx in list_idx_slices]
    plot_sum_subplots_image(list_images, path_save=path_visualisations, category='slices', name=study_id, cmap=None)

    return sitk_bronchovascular_bundle


if __name__ == "__main__":
    path_image = r"X:\Wojtek\MIRASLC\Dane surowe\Moltest 1\NRRD\Abramik_Miroslaw.nrrd"
    path_save = r"D:\msocha\PHD\2024-03-05 - bronco test"
    pipeline(path_image, path_save=path_save, save_all=True, return_binary=True, cache=False, study_id="AbramikM")
