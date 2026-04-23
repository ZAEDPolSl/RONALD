import os
import scipy
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label, regionprops_table
from skimage.morphology import binary_erosion, cube, ball
from ctools.ImageInstance import ImageInstance

# max_label do filtracji


def flood_fill_hull(image):
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull


def grade(sitk_bronchi):
    bronchi = sitk.GetArrayFromImage(sitk_bronchi)
    convex_hull_image, convex_hull = flood_fill_hull(bronchi)
    # show(convex_hull_image)
    convex_hull_image = convex_hull_image - binary_erosion(convex_hull_image, cube(10))
    # show(convex_hull_image)
    terminal_branches = bronchi * convex_hull_image
    # show(terminal_branches)
    terminal_branches = label(terminal_branches)
    # find and remove too small areas
    df = pd.DataFrame(
        regionprops_table(terminal_branches, properties=["label", "area"])
    )
    for i, row in df.iterrows():
        if row["area"] < 10:
            # without this there was an error in minor axis length
            terminal_branches[terminal_branches == row["label"]] = 0
    properties = [
        "label",
        "area",
        "axis_major_length",
        "axis_minor_length",
        "equivalent_diameter_area",
        "euler_number",
        "extent",
        "area_filled",
    ]  # 'feret_diameter_max',
    props = regionprops_table(terminal_branches, properties=properties)
    df_props = pd.DataFrame(props)
    df_summary = pd.DataFrame()
    for col in df_props.columns:
        df_summary.loc[0, f"max_{col}"] = np.max(df_props[col])
        df_summary.loc[0, f"min_{col}"] = np.min(df_props[col])
        df_summary.loc[0, f"mean_{col}"] = np.mean(df_props[col])
        df_summary.loc[0, f"std_{col}"] = np.std(df_props[col])
        df_summary.loc[0, f"median_{col}"] = np.median(df_props[col])
        # df_summary.loc[0, f"max_{col}"] = np.max(col)
    # df = df.apply(lambda x: np.median(), axis=1)
    return df_summary


if __name__ == "__main__":
    path_patient = "/mnt/pmanas/Ania/phd-data/moltest-1/masks/Wojtowicz_Henryk"
    path_airways = os.path.join(path_patient, "airways_final.nrrd")
    ii = ImageInstance()
    sitk_image = ii.read(path_airways)
    df = grade(sitk_image)
    print(df)
