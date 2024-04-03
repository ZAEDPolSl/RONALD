from bronco.segmentation.vessels_segmentation import vessels_segmentation


if __name__ == "__main__":
    path_input = "data/input/patient1.nrrd"
    path_supl = "data/input/patient1_lungs.nrrd"
    path_output = "data/output"

    vessels_segmentation(path_input, path_supl, path_output)
