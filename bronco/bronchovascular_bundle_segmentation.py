import os
import SimpleITK as sitk
from .io_local.ImageInstance import ImageInstance
from bronco.preprocessing import preprocess_lungs
from bronco.preprocessing import run_thresholding
from bronco.preprocessing import skeleton_graph_analysis


def bronchovascular_bundle_segmentation(path_image, path_lungs, path_save=None, **kwargs):
    """
    Bronchovascular bundle segmentation function.
    
    Parameters
    ----------
    path_image : (str) path to the folder with DICOM series or the NRRD file,
    path_lungs : (str) path to the folder with DICOM series or the NRRD file,
    path_save : (str) path to the folder where results are going to be stored,
    kwargs :
        - retain_main_bronchi : (bool) default True, whether to retain main broncho - time consuming operation,
        - return_binary : (bool) default True, whether to return binary or hierarchically labelled mask,
        - save_all: (bool) default False, whether to save all intermediate series - gmm results, skeleton etc.
        - ii: (ImageInstance) default None, ImageInstance is a class object responsible for saving the results.

    Returns
    -------
    sitk_bronchovascular_bundle : (sitk.Image) image of the bronchovascular bundle,
    sitk_thresholds : (sitk.Image) image of the raw (before the hierarchical clustering) bronchovascular bundle
    """
    print("Reading the data...")
    if type(path_image) is str:
        ii = ImageInstance()
        sitk_image = ii.read(path_image)
    elif type(path_image) is sitk.Image:
        ii = kwargs.get('ii', None)
        sitk_image = path_image
    else:
        raise TypeError(f"Wrong dtype of path_image variable: {type(path_image)}, should be str or sitk.Image")

    if type(path_lungs) is str:
        sitk_lungs = ImageInstance().read(path_lungs)
    elif type(path_lungs) is sitk.Image:
        sitk_lungs = path_lungs
    else:
        raise TypeError(f"Wrong dtype of path_lungs variable: {type(path_lungs)}, should be str or sitk.Image")

    # preprocess lungs
    sitk_lungs, sitk_labelled_bronchi, sitk_mediastinum = preprocess_lungs(sitk_image, sitk_lungs,
                                                                           kwargs.get('retain_main_bronchi', True))

    # TODO: DEBUG saving
    # path_save = r"D:\msocha\PHD\2023-09-02 - testing airways segmentation\1"
    # ii.write(sitk.Cast(sitk_lungs, sitk.sitkUInt8), os.path.join(path_save, 'lungs'))
    # ii.write(sitk.Cast(sitk_labelled_bronchi, sitk.sitkUInt8), os.path.join(path_save, 'trachea'))
    # ii.write(sitk.Cast(sitk_mediastinum, sitk.sitkUInt8), os.path.join(path_save, 'mediastinum'))
    if sitk_labelled_bronchi is not None:
        pass
        # sitk_airways = get_airways(sitk_lungs, sitk_labelled_bronchi)

    # run thresholding using gmm
    sitk_thresholds = run_thresholding(sitk_image, sitk_lungs, number_of_gmms=3)

    # get main airways mask
    threshold = sitk.GetArrayFromImage(sitk_thresholds)
    threshold[threshold != 3] = 0
    threshold[threshold == 3] = 1
    sitk_thresholds = sitk.GetImageFromArray(threshold)
    sitk_thresholds.CopyInformation(sitk_image)

    # get lungs image
    stats = sitk.StatisticsImageFilter()
    stats.Execute(sitk_image)
    _min = stats.GetMinimum()
    sitk_lungs_image = sitk.Mask(sitk_image, sitk_lungs, outsideValue=_min - 1)

    # skeletonize and graph analysis
    if kwargs.get('save_all', False):
        _path_save = path_save
    else:
        _path_save = None
    sitk_bronchovascular_bundle = skeleton_graph_analysis(sitk_lungs_image, sitk_thresholds, ii=ii,
                                                          return_binary=kwargs.get('return_binary', True),
                                                          path_save=_path_save)
    sitk_bronchovascular_bundle.CopyInformation(sitk_image)

    # save results
    if path_save is not None and type(path_image) is str:
        print("Saving the results...")
        _path_save = os.path.join(path_save, 'bronchovascular_bundle')
        os.makedirs(_path_save, exist_ok=True)
        ii.write(sitk_bronchovascular_bundle, _path_save, description='Bronchovascular bundle')

        _path_save = os.path.join(path_save, 'raw_bronchovascular_bundle')
        os.makedirs(_path_save, exist_ok=True)
        ii.write(sitk_thresholds, _path_save, description='Raw Bronchovascular bundle - after GMM only')

    return sitk_bronchovascular_bundle, sitk_thresholds
