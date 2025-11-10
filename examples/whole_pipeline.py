import os
import SimpleITK as sitk
from bronco.segmentation import (
    airways_segmentation,
    lungs_segmentation,
    lobes_segmentation,
    mediastinum_segmentation,
    vessel_segmentation,
)
from bronco.modelling.smooth_tree import smooth_tree
from ctools import ImageInstance

opj = os.path.join
dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.split(dir_path)[0]

if __name__ == "__main__":
    path_data = opj(root, "data/patient1.nrrd")

    ii = ImageInstance()
    sitk_image = ii.read(path_data)
    sitk_lungs = lungs_segmentation(sitk_image)
    sitk_lobes = lobes_segmentation(sitk_image)
    sitk_mediastinum = mediastinum_segmentation(sitk_lungs)
    sitk_airways, _, __, sitk_rough = airways_segmentation(sitk_image, sitk_lungs)
    sitk_tracheobronchal = smooth_tree(sitk_rough, sitk_airways)
    sitk_vessels = vessel_segmentation(
        sitk_image, sitk_lungs, sitk_lobes, sitk_mediastinum
    )

    print("Done!")
