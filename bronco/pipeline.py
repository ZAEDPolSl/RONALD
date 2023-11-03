import os
from bronco.io_local.ImageInstance import ImageInstance
from bronco.lungs_segmentation import lungs_segmentation
from bronco.bronchovascular_bundle_segmentation import bronchovascular_bundle_segmentation


def pipeline(path_series, path_save=None, **kwargs):
    """
        Pipeline of lungs segmentation followed by the bronchovascular bundel modeling.

        Parameters
        ----------
        path_image : (str) path to the folder with DICOM series or the NRRD file,
        path_save : (str) path to the folder where results are going to be stored,
        kwargs :
            - retain_main_bronchi : (bool) default True, whether to retain main broncho - time consuming operation,
            - return_binary : (bool) default True, whether to return binary or hierarchically labelled mask,
            - save_all: (bool) default False, whether to save all intermediate series - gmm results, skeleton etc.

        Returns
        -------
        sitk_bronchovascular_bundle : (sitk.Image) image of the bronchovascular bundle,
        sitk_thresholds : (sitk.Image) image of the raw (before the hierarchical clustering) bronchovascular bundle
        """
    # reading
    ii = ImageInstance()
    sitk_series = ii.read(path_series)

    # lung segmentation\
    path_lungs = kwargs.get('path_lungs', None)
    if path_lungs is None:
        sitk_lungs = lungs_segmentation(sitk_image=sitk_series)
        if path_save is not None:
            path_save_lungs = os.path.join(path_save, 'lungs')
            os.makedirs(path_save_lungs, exist_ok=True)
            ii.write(sitk_lungs, path_save_lungs, description="Lungs segmentation")
    else:
        sitk_lungs = ImageInstance().read(path_lungs)
        del kwargs['path_lungs']

    # bronchovascular bundle segmentation
    kwargs['ii'] = ii
    sitk_bbv, sitk_raw_bbv = bronchovascular_bundle_segmentation(sitk_series, sitk_lungs, path_save, **kwargs)
    if path_save is not None:
        path_save_bbv = os.path.join(path_save, 'bronchovascular_bundle')
        os.makedirs(path_save_bbv, exist_ok=True)
        ii.write(sitk_bbv, path_save_bbv, description="Bronchovascular bundle segmentation")

    return sitk_lungs, sitk_bbv


if __name__ == "__main__":
    path_image = r"X:\Wojtek\MIRASLC\Dane surowe\Moltest 1\NRRD\Abramik_Miroslaw.nrrd"
    path_save = r"D:\msocha\PHD\2023-08-31 - bronchi test abr mir"
    pipeline(path_image, path_save=path_save, save_all=True, return_binary=False)