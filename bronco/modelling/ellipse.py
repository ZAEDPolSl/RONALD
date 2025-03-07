import numpy as np
from sklearn.decomposition import PCA
from skimage.measure import EllipseModel


def points_in_plane(points, plane_normal, plane_point):
    """
    Check if points are in the plane defined by a normal vector and a point on the plane.

    Parameters:
        points: numpy array of shape (N, 3) - Array of 3D points.
        plane_normal: numpy array of shape (3,) - Normal vector of the plane.
        plane_point: numpy array of shape (3,) - Point on the plane.

    Returns:
        numpy array of bools - True if point is in the plane; False otherwise.
    """
    # Compute vectors from plane point to each point
    vectors = points - plane_point

    # Project onto plane normal
    projection_lengths = np.dot(vectors, plane_normal)

    return np.isclose(projection_lengths, 0)


def fit_ellipse_3d(points):
    """
    Fit an ellipse to a set of 3D points.
    :param points: Nx3 array of 3D points
    :return: center (3,), semi_axes (2,), axes_vectors (2,3), rotation_matrix (3,3)
    """
    # Step 1: Perform PCA to find the best fitting plane
    pca = PCA(n_components=3)
    pca.fit(points)
    plane_normal = pca.components_[2]  # Normal to the best fit plane
    
    # Step 2: Define a new coordinate system on the plane
    origin = np.mean(points, axis=0)
    x_axis, y_axis = pca.components_[:2]
    rotation_matrix = np.vstack([x_axis, y_axis, plane_normal])  # 3x3 matrix
    
    # Step 3: Transform 3D points to 2D
    transformed_points = (points - origin) @ rotation_matrix.T
    points_2d = transformed_points[:, :2]  # Only take X and Y coordinates
    mean_2d = np.mean(points_2d, axis=0)
    
    # Step 4: Fit an ellipse in 2D using skimage
    ellipse = EllipseModel()
    if not ellipse.estimate(points_2d):
        raise ValueError("Ellipse fitting failed.")
    xc, yc, a, b, theta = ellipse.params  # Extract parameters
    xc += mean_2d[0]
    yc += mean_2d[1]
    
    # Step 5: Transform the fitted ellipse back to 3D
    center_2d = np.array([xc, yc])
    center_3d = origin + center_2d[0] * x_axis + center_2d[1] * y_axis
    
    # Step 6: Compute the 3D major and minor axis vectors scaled by their lengths
    major_axis_3d = a * (np.cos(theta) * x_axis + np.sin(theta) * y_axis)
    minor_axis_3d = b * (-np.sin(theta) * x_axis + np.cos(theta) * y_axis)
    print("ellipse vectors 3d OG")
    print(major_axis_3d, minor_axis_3d)
    return center_3d, major_axis_3d, minor_axis_3d


def find_ellipse(image, plane_normal, point):
    points = np.argwhere(image == 1)
    # Check which points are in the plane
    in_plane = points_in_plane(points, plane_normal, point)
    params = fit_ellipse_3d(points[in_plane])
    return params
