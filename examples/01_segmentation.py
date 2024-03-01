from bronco.segmentation.bronchovascular_bundle_segmentation import bronchovascular_bundle_segmentation


if __name__ == "__main__":
    path_input = "data/input/patient1.nrrd"
    path_supl = "data/input/patient1_lungs.nrrd"
    path_output = "data/output"

    bronchovascular_bundle_segmentation(path_input, path_supl, path_output)
