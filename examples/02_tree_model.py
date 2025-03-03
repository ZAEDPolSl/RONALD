import os
import SimpleITK as sitk
from bronco.modelling.smooth_tree import smooth_tree
from bronco.io_local import ImageInstance

opj = os.path.join
dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.split(dir_path)[0]

if __name__ == "__main__":
    path_input = opj(root, "data/output.nrrd")
    path_output_tree = opj(root, "data/tree.nrrd")
    path_output_smooth = opj(root, "data/smooth_tree.nrrd")

    ii = ImageInstance()
    sitk_data = ii.read(path_input)
    bronco_new, tree = smooth_tree(sitk_data)
    ii.write(bronco_new, path_output_smooth)
    ii.write(tree, path_output_tree)
    print("Done!")
