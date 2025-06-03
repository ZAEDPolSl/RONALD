import numpy as np
from sklearn.decomposition import PCA

from bronco.modelling.cone_construction import is_point_in_cylinder
from bronco.modelling.densify import densify_point_cloud
from bronco.modelling.ellipse import find_ellipse as f_el
from bronco.modelling.fill_gaps import fill_gaps
from bronco.modelling.segment_branch import segment_branch


class BranchAnalyser:
    def __init__(self, eps=1e-10, segments=True):
        self.eps = eps
        self.segments = segments
        self.svd = None
        self.indices_options = []
        self.points = None
        self.transformed_points = None
        self.densified_transformed = None
        self.aggregated_gaps = []
        self.best_cylinder = None
        self.first_base = None
        self.second_base = None

    def between_endpoints(self, points, c1, c2):
        axis_vector = c2 - c1
        height = np.linalg.norm(axis_vector)
        axis_unit_vector = axis_vector / (height + self.eps)
        vector_to_points = points - c1
        projection_lengths = np.dot(vector_to_points, axis_unit_vector)
        height_bounds_check = (0 <= projection_lengths) & (projection_lengths <= height)
        return points[height_bounds_check]

    def prepare_branch_svd(self, points):
        branch_points = densify_point_cloud(points, factor=250)
        self.svd = PCA(n_components=3)
        self.svd.fit(branch_points)
        self.densified_transformed = self.svd.transform(branch_points)
        self.transformed_points = self.svd.transform(points)

    def separate_branch(self, transformed_endpoints):
        transformed_points = self.densified_transformed
        if self.segments:
            transformed_points = self.between_endpoints(
                transformed_points,
                transformed_endpoints[0, :],
                transformed_endpoints[1, :],
            )

        first_val = transformed_endpoints[0, 0]
        second_val = transformed_endpoints[1, 0]
        tol = (transformed_points[:, 0].max() - transformed_points[:, 0].min()) / 100

        mask_first = np.isclose(transformed_points[:, 0], first_val, atol=tol)
        mask_second = np.isclose(transformed_points[:, 0], second_val, atol=tol)

        points_first = transformed_points[mask_first]
        points_second = transformed_points[mask_second]

        ellipse1 = f_el(points_first, transformed_endpoints[0, :])
        ellipse2 = f_el(points_second, transformed_endpoints[1, :])

        return [ellipse1, ellipse2], [points_first, points_second]

    def check_ellipse_convergence(self, ellipse, constraint, to_higher=True):
        major_len, minor_len = np.linalg.norm(ellipse[1]), np.linalg.norm(ellipse[2])
        c_major_len, c_minor_len = np.linalg.norm(constraint[1]), np.linalg.norm(
            constraint[2]
        )

        if major_len > c_major_len and c_minor_len > 0 and to_higher:
            major_unit = ellipse[1] / (major_len + self.eps)
            minor_unit = ellipse[2] / (minor_len + self.eps)
            new_major = major_unit * (c_major_len + self.eps)
            new_minor = minor_unit * (c_minor_len + self.eps)
            return (ellipse[0], new_major, new_minor)

        if major_len == 0 or minor_len == 0:
            return (ellipse[0], constraint[1], constraint[2])

        return ellipse

    def analyse_segment(self, ellipses):
        ellipse1, ellipse2 = ellipses
        ellipse2 = self.check_ellipse_convergence(ellipse2, ellipse1, to_higher=True)
        ellipse1 = self.check_ellipse_convergence(ellipse1, ellipse2, to_higher=False)

        return is_point_in_cylinder(
            self.transformed_points,
            ellipse1[0],
            ellipse2[0],
            ellipse1[1],
            ellipse1[2],
            ellipse2[1],
            ellipse2[2],
        )

    def initialise(self, branch, image):
        self.points = np.argwhere(image == 1)
        self.prepare_branch_svd(self.points)

        if self.segments:
            self.indices_options = segment_branch(branch, adaptive=True)
        else:
            self.indices_options = [np.array([0, -1])]

    def analyse_indices_option(self, indices, branch, image):
        smooth_cylinder = np.zeros(image.shape, dtype=int)
        ellipse_pairs = []

        for i in range(len(indices) - 1):
            if (
                np.all(branch[indices[i] + 1] == branch[indices[i + 1] - 1])
                or indices[i] != 0
            ):
                start_idx, end_idx = indices[i], indices[i + 1]
            else:
                start_idx, end_idx = indices[i] + 1, indices[i + 1] - 1

            transformed_endpoints = self.svd.transform(
                np.array([branch[start_idx], branch[end_idx]])
            )
            ellipses, ellipse_points = self.separate_branch(transformed_endpoints)
            if i > 0:
                prev_lower = self.svd.inverse_transform(prev_upper)
                curr_upper = self.svd.inverse_transform(ellipse_points[0])
                ellipse_pairs.append((prev_lower, curr_upper))

            inside_cylinder = self.analyse_segment(ellipses)
            cyl = self.points[inside_cylinder]
            cyl_mask = np.zeros(image.shape, dtype=int)
            cyl_mask[cyl[:, 0], cyl[:, 1], cyl[:, 2]] = 1
            np.logical_or(smooth_cylinder, cyl_mask, out=smooth_cylinder)

            prev_upper = ellipse_points[1]

            if i == 0:
                on_first_base = ellipse_points[0]
            on_second_base = ellipse_points[1]

        _, major_axis, minor_axis = ellipses[1]
        major_len = np.linalg.norm(major_axis)
        minor_len = np.linalg.norm(minor_axis)
        thickness = min(major_len, minor_len)
        return (
            smooth_cylinder,
            [on_first_base, on_second_base],
            ellipse_pairs,
            thickness,
        )

    def smooth_branch(self, branch, image):
        self.initialise(branch, image)
        best_score = -1
        self.best_cylinder = np.ones(image.shape, dtype=int) * (-1)
        self.aggregated_gaps = []

        for indices in self.indices_options:
            smooth_cylinder, bases, gaps, thickness = self.analyse_indices_option(
                indices, branch, image
            )
            first_base, second_base = bases
            score = np.sum(smooth_cylinder)

            if score > best_score:
                best_score = score
                self.best_cylinder = smooth_cylinder
                self.first_base = self.svd.inverse_transform(first_base)
                self.second_base = self.svd.inverse_transform(second_base)
                self.aggregated_gaps = gaps
                self.thickness = thickness
            elif score == 0:
                continue
            else:
                break

        # Only fill gaps once using saved best-cylinder's ellipse pairs
        for lower, upper in self.aggregated_gaps:
            self.best_cylinder = fill_gaps(lower, upper, self.best_cylinder)

        return (
            self.best_cylinder.astype(int),
            self.first_base,
            self.second_base,
            self.thickness,
        )
