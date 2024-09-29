from bronco.run import run


if __name__ == "__main__":
    path_input = "data/input/patient1.nrrd"
    path_supl = "data/input/patient1_lungs.nrrd"
    path_output = "data/output"

    run(path_input, path_output, path_supl)
