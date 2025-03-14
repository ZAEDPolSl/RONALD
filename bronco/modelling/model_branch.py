import numpy as np
from skimage.util import label_points
from skimage.measure import label
from sklearn.decomposition import PCA

from bronco.modelling.cone_construction import is_point_in_cylinder
from bronco.modelling.oval_construction import find_ellipse
from bronco.modelling.ellipse import find_ellipse as f_el
from bronco.modelling.segment_branch import segment_branch
from bronco.modelling.densify import densify_point_cloud


def separate_branch(image, endpoints):
    checkpoints = np.argwhere(image == 1)
    # use the whole coordinate system
    branch_direction = endpoints[1] - endpoints[0]
    height = np.linalg.norm(branch_direction)
    axis_unit_vector = branch_direction / (height + 1e-10)

    # Project points onto cylinder's axis
    vector_to_points = checkpoints - endpoints[0]
    projection_lengths = np.dot(vector_to_points, axis_unit_vector)

    # Check if projections are within height bounds
    height_bounds_check = (0 <= projection_lengths) & (projection_lengths <= height)
    checkpoints_cut = checkpoints[height_bounds_check]
    mask = label_points(checkpoints_cut, image.shape).astype(bool).astype(int)
    labeled_image = label(mask)
    label_at_point = labeled_image[tuple(endpoints[0])]
    branch_points = np.argwhere(labeled_image == label_at_point)
    branch_points = densify_point_cloud(branch_points, factor=50)

    svd = PCA(n_components=3)
    svd.fit(branch_points)
    most_important_direction = svd.components_[0]
    transformed_points = svd.transform(branch_points)
    transformed_endpoints = svd.transform(endpoints)

    # Extract the first principal component values of endpoints
    first_endpoint_value = transformed_endpoints[0, 0]  # First endpoint on first axis
    second_endpoint_value = transformed_endpoints[1, 0]  # Second endpoint on first axis

    # trail_trans_ch = svd.transform(checkpoints)
    # trial_mask = np.isclose(trail_trans_ch[:, 0], first_endpoint_value, atol=1e-07)
    # Find points matching the same height on the first SVD axis
    tol = (transformed_points[:, 0].max() - transformed_points[:, 0].min()) / 100
    mask_first = np.isclose(transformed_points[:, 0], first_endpoint_value, atol=tol)
    mask_second = np.isclose(transformed_points[:, 0], second_endpoint_value, atol=tol)

    # Extract the corresponding original points
    points_same_as_first = transformed_points[mask_first]
    points_same_as_second = transformed_points[mask_second]

    ellipse1 = f_el(points_same_as_first, transformed_endpoints[0, :], svd)
    ellipse2 = f_el(points_same_as_second, transformed_endpoints[1, :], svd)
    return [ellipse1, ellipse2], svd


def check_ellipse_convergence(ellipse, ellipse_constraint, eps=1e-10, to_higher=True):
    """
    Check if the ellipse is converging to the constraint ellipse. If not, resize the ellipse to the constraint ellipse.

    Parameters:
        ellipse: numpy array of shape (3,3) - Ellipse parameters (center, major axis, minor axis)
        ellipse_constraint: numpy array of shape (3,3) - Ellipse constraint parameters (center, major axis, minor axis)
        eps: float - Small value to avoid division by zero.
    Returns:
        ellipse: tuple - Ellipse parameters (center, major axis, minor axis)
    """
    if (
        np.linalg.norm(ellipse[1]) > np.linalg.norm(ellipse_constraint[1])
        and np.linalg.norm(ellipse_constraint[2]) > 0
        and to_higher
    ):
        new_major_axis = (
            ellipse[1]
            / (np.linalg.norm(ellipse[1]) + eps)
            * (np.linalg.norm(ellipse_constraint[1]) + eps)
        )
        new_minor_axis = (
            ellipse[2]
            / (np.linalg.norm(ellipse[2]) + eps)
            * (np.linalg.norm(ellipse_constraint[2]) + eps)
        )
        ellipse = ellipse[0], new_major_axis, new_minor_axis
    elif np.linalg.norm(ellipse[1]) == 0 or np.linalg.norm(ellipse[2]) == 0:
        ellipse = ellipse[0], ellipse_constraint[1], ellipse_constraint[2]
    return ellipse


def analyse_segment(
    points, ellipses, branch, indices=[0, -1], previous_oval=None, eps=1e-10
):
    """
    Analyse a branch by fitting cylinder to each point and checking if the point is in the cylinder.

    Parameters:
        points: numpy array of shape (N, 3) - Points in the 3D image coordinate system.
        ellipse_fit_points: numpy array of shape (N, 3) - Points in the 3D image.
        branch: numpy array of shape (M, 3) - Points in the branch.
        indices: list - List of indices of the points to analyse.
        previous_oval: tuple - Tuple of the previous ellipse major and minor axis
        eps: float - Small value to avoid division by zero.
    Returns:
        inside_cylinder: numpy array of shape (N, 3) - Points inside the cylinder.
        ellipse2: tuple - Ellipse parameters (center, major axis, minor axis)
    """
    ellipse1 = ellipses[0]
    ellipse2 = ellipses[1]
    # point1 = branch[indices[0]]
    # # plane_normal1 = branch[indices[0] + 1] - point1
    # point2 = branch[indices[-1]]
    # # plane_normal2 = point2 - branch[indices[-1] - 1]

    # ellipse1 = find_ellipse(ellipse_fit_points, plane_normal1, point1)
    # print(ellipse1)
    # ellipse2 = find_ellipse(ellipse_fit_points, plane_normal2, point2)
    # print(ellipse2)
    ellipse2 = check_ellipse_convergence(ellipse2, ellipse1, eps=eps)
    ellipse1 = check_ellipse_convergence(ellipse1, ellipse2, eps=eps, to_higher=False)

    # if previous_oval is not None:
    #     # check if the new ellipse is bigger than the previous one
    #     ellipse1 = check_ellipse_convergence(ellipse1, previous_oval, eps=eps)
    #     ellipse2 = check_ellipse_convergence(ellipse2, previous_oval, eps=eps)
    inside_cylinder = is_point_in_cylinder(
        points,
        ellipse1[0],  # point1,  # ellipse1[0],
        ellipse2[0],  # point2,  # ellipse2[0],
        ellipse1[1],
        ellipse1[2],
        ellipse2[1],
        ellipse2[2],
    )
    return inside_cylinder, ellipse2


def smooth_branch(branch, image, segments=False, previous_oval=None, eps=1e-10):
    """
    Smooth a branch by fitting an ellipse to each point and checking if the point is in the cylinder.

    Parameters:
        branch: numpy array of shape (N, 3) - Points in the branch.
        image: numpy array of shape (I, J, K) - 3D binary image
        mode: str - Mode for smoothing the branch. Options are "5percent" or "full".
        previous_oval: tuple - Tuple of the previous ellipse major and minor axis
        eps: float - Small value to avoid division by zero.
    Returns:
        smooth_branch: numpy array of shape (I, J, K) - 3D binary image
        ellipse2: numpy array of shape (3,3) - Ellipse parameters (center, major axis, minor axis)
    """
    smooth_cylinder = np.zeros(image.shape, dtype=int)

    x, y, z = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
        np.arange(image.shape[2]),
        indexing="ij",
    )
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    ellipse_fit_points = np.argwhere(image == 1)

    if segments:
        indices = segment_branch(branch)
    else:
        indices = np.array([0, -1])
    # for each consecutive pair of indices, analyse the segment
    for i in range(len(indices) - 1):
        if np.all(branch[indices[i] + 1] == branch[indices[i + 1] - 1]):
            start_idx = indices[i]
            end_idx = indices[i + 1]
        else:
            start_idx = indices[i] + 1
            end_idx = indices[i + 1] - 1
        ellipses, svd = separate_branch(
            image, np.array([branch[start_idx], branch[end_idx]])
        )
        transformed_points = svd.transform(points)
        inside_cylinder, previous_oval = analyse_segment(
            transformed_points,
            ellipses,
            branch,
            indices=indices[i : i + 2],
            previous_oval=previous_oval,
            eps=eps,
        )
        # inside cylinder back to 3D
        inside_cylinder = inside_cylinder.reshape(image.shape)
        smooth_cylinder = np.logical_or(smooth_cylinder, inside_cylinder)
    return smooth_cylinder.astype(int), previous_oval
