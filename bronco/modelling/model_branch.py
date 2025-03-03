import numpy as np
import math

from bronco.modelling.oval_construction import find_ellipse
from bronco.modelling.cone_construction import is_point_in_cylinder


def smooth_branch(branch, image, mode="5percent", previous_oval=None):
    """
    Smooth a branch by fitting an ellipse to each point and checking if the point is in the cylinder.

    Parameters:
        branch: numpy array of shape (N, 3) - Points in the branch.
        image: numpy array of shape (I, J, K) - 3D binary image
        mode: str - Mode for smoothing the branch. Options are "5percent" or "full".
        previous_oval: tuple - Tuple of the previous ellipse major and minor axis

    Returns:
        smooth_branch: numpy array of shape (I, J, K) - 3D binary image
    """
    if previous_oval is not None:
        max_elipse_major = previous_oval[0]
        max_elipse_minor = previous_oval[1]
    if mode == "5percent":
        first_5_percent = math.ceil(branch.shape[0] * 0.05)
        last_5_percent = math.floor(branch.shape[0] * 0.95 - 1)
        point1 = branch[first_5_percent]
        point2 = branch[last_5_percent]

    else:
        point1 = branch[0]
        point2 = branch[-1]
    plane_normal = point2 - point1
    ellipse1 = find_ellipse(image, plane_normal, point1)
    ellipse2 = find_ellipse(image, plane_normal, point2)
    if previous_oval is not None:
        # check if the new ellipse is bigger than the previous one
        if np.linalg.norm(ellipse1[1]) > np.linalg.norm(
            max_elipse_major
        ) or np.linalg.norm(ellipse1[2]) > np.linalg.norm(max_elipse_minor):
            # normalize the new ellipse to the previous one
            new_major_axis = (
                ellipse1[1]
                / np.linalg.norm(ellipse1[1])
                * np.linalg.norm(max_elipse_major)
            )
            new_minor_axis = (
                ellipse1[2]
                / np.linalg.norm(ellipse1[2])
                * np.linalg.norm(max_elipse_minor)
            )
            ellipse1 = ellipse1[0], new_major_axis, new_minor_axis
        if np.linalg.norm(ellipse2[1]) > np.linalg.norm(
            max_elipse_major
        ) or np.linalg.norm(ellipse2[2]) > np.linalg.norm(max_elipse_minor):
            new_major_axis = (
                ellipse2[1]
                / np.linalg.norm(ellipse2[1])
                * np.linalg.norm(max_elipse_major)
            )
            new_minor_axis = (
                ellipse2[2]
                / np.linalg.norm(ellipse2[2])
                * np.linalg.norm(max_elipse_minor)
            )
            ellipse2 = ellipse2[0], new_major_axis, new_minor_axis
    x, y, z = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
        np.arange(image.shape[2]),
        indexing="ij",
    )
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    inside_cylinder = is_point_in_cylinder(
        points,
        ellipse1[0],
        ellipse2[0],
        ellipse1[1],
        ellipse1[2],
        ellipse2[1],
        ellipse2[2],
    )
    # inside cylinder back to 3D
    inside_cylinder = inside_cylinder.reshape(image.shape)
    return inside_cylinder, (ellipse2[1], ellipse2[2])
