import numpy as np

from ronald.modelling.cone_construction import is_point_in_cylinder
from ronald.modelling.densify import densify_point_cloud
from ronald.modelling.ellipse import check_ellipse, find_ellipse
from ronald.modelling.fill_gaps import fill_gaps
from ronald.modelling.principal_axes import PrincipalAxes3D
from ronald.modelling.segment_branch import segment_branch

DENSIFICATION_FACTOR = 100
TRANSFORM_CHUNK_SIZE = 65_536


class BranchAnalyser:
    def __init__(self, eps=1e-10, segments=True, verbose=False):
        self.eps = eps
        self.segments = segments
        self.verbose = verbose
        self.principal_axes = None
        self.indices_options = []
        self.points = None
        self.transformed_points = None
        self.densified_transformed = None
        self.aggregated_gaps = []
        self.best_cylinder = None
        self.first_base = None
        self.second_base = None

    def between_endpoints(self, points, c1, c2):
        return points[self.between_endpoints_mask(points, c1, c2)]

    def between_endpoints_mask(self, points, c1, c2, chunk_size=65_536):
        axis_vector = c2 - c1
        height = np.linalg.norm(axis_vector)
        axis_unit_vector = axis_vector / (height + self.eps)
        selected = np.empty(len(points), dtype=bool)
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            vector_to_points = points[start:stop] - c1
            projection_lengths = np.dot(vector_to_points, axis_unit_vector)
            selected[start:stop] = (0 <= projection_lengths) & (
                projection_lengths <= height
            )
        return selected

    def prepare_principal_axes(self, points):
        if self.verbose:
            print("  Preparing branch PCA...")
        branch_points = densify_point_cloud(points, factor=DENSIFICATION_FACTOR)
        self.principal_axes = PrincipalAxes3D().fit(branch_points)
        for start in range(0, len(branch_points), TRANSFORM_CHUNK_SIZE):
            stop = min(start + TRANSFORM_CHUNK_SIZE, len(branch_points))
            branch_points[start:stop] = self.principal_axes.transform(
                branch_points[start:stop]
            )
        self.densified_transformed = branch_points
        self.transformed_points = self.principal_axes.transform(points)
        if self.verbose:
            print(f"  PCA prepared: {len(points)} points transformed")

    def separate_branch(self, transformed_endpoints):
        transformed_points = self.densified_transformed
        if self.verbose:
            print("  Separating branch at endpoints...")

        if self.segments:
            between_mask = self.between_endpoints_mask(
                transformed_points,
                transformed_endpoints[0, :],
                transformed_endpoints[1, :],
            )
            if self.verbose:
                print(
                    f"  Filtered to {np.count_nonzero(between_mask)} points between endpoints"
                )
        else:
            between_mask = np.ones(len(transformed_points), dtype=bool)

        first_val = transformed_endpoints[0, 0]
        second_val = transformed_endpoints[1, 0]
        first_axis = transformed_points[:, 0]
        first_axis_min = np.min(first_axis, where=between_mask, initial=np.inf)
        first_axis_max = np.max(first_axis, where=between_mask, initial=-np.inf)
        tol = (first_axis_max - first_axis_min) / 100

        mask_first = between_mask & np.isclose(
            transformed_points[:, 0], first_val, atol=tol
        )
        mask_second = between_mask & np.isclose(
            transformed_points[:, 0], second_val, atol=tol
        )

        points_first = transformed_points[mask_first]
        points_second = transformed_points[mask_second]

        if self.verbose:
            print(
                f"  First base: {len(points_first)} points, Second base: {len(points_second)} points"
            )

        ellipse1 = find_ellipse(points_first, transformed_endpoints[0, :])
        ellipse2 = find_ellipse(points_second, transformed_endpoints[1, :])

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

    def ellipse_point_mask(self, transformed_endpoints):
        """Select original base points when a segment contains no cylinder points."""
        first_val = transformed_endpoints[0, 0]
        second_val = transformed_endpoints[1, 0]
        tol = (
            self.transformed_points[:, 0].max() - self.transformed_points[:, 0].min()
        ) / 100

        mask_first = np.isclose(self.transformed_points[:, 0], first_val, atol=tol)
        mask_second = np.isclose(self.transformed_points[:, 0], second_val, atol=tol)

        base1_transformed_points = self.transformed_points[mask_first]
        base2_transformed_points = self.transformed_points[mask_second]

        ellipse1 = find_ellipse(base1_transformed_points, transformed_endpoints[0, :])
        ellipse2 = find_ellipse(base2_transformed_points, transformed_endpoints[1, :])

        base1_in_ellipse = check_ellipse(
            base1_transformed_points, ellipse1[0], ellipse1[1], ellipse1[2], self.eps
        )
        base2_in_ellipse = check_ellipse(
            base2_transformed_points, ellipse2[0], ellipse2[1], ellipse2[2], self.eps
        )

        selected = np.zeros(len(self.points), dtype=bool)
        if np.any(base1_in_ellipse):
            selected[np.flatnonzero(mask_first)[base1_in_ellipse]] = True
        if np.any(base2_in_ellipse):
            selected[np.flatnonzero(mask_second)[base2_in_ellipse]] = True
        return selected

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

    def initialise(self, branch, image=None, points=None):
        if self.verbose:
            print("  Initializing branch analysis...")
        if points is None:
            if image is None:
                raise ValueError("Either image or points must be provided")
            points = np.argwhere(image == 1)
        self.points = np.asarray(points)
        if self.verbose:
            print(f"  Found {len(self.points)} points in branch mask")
        self.prepare_principal_axes(self.points)

        if self.segments:
            if self.verbose:
                print("  Segmenting branch...")
            self.indices_options = segment_branch(branch, adaptive=True)
            if self.verbose:
                print(f"  Generated {len(self.indices_options)} segmentation options")
        else:
            self.indices_options = [np.array([0, -1])]
            if self.verbose:
                print("  Using single segment for branch (endpoints only)")

    def analyse_indices_option(self, indices, branch):
        if self.verbose:
            print(f"  Analyzing branch option with {len(indices)} segments...")

        smooth_cylinder = np.zeros(len(self.points), dtype=bool)
        gaps = []

        for i in range(len(indices) - 1):
            if self.verbose and len(indices) > 2:
                print(f"    Processing segment {i+1}/{len(indices)-1}...")

            if (
                np.all(branch[indices[i] + 1] == branch[indices[i + 1] - 1])
                or indices[i] != 0
            ):
                start_idx, end_idx = indices[i], indices[i + 1]
            else:
                start_idx, end_idx = indices[i] + 1, indices[i + 1] - 1

            transformed_endpoints = self.principal_axes.transform(
                np.array([branch[start_idx], branch[end_idx]])
            )
            ellipses, ellipse_points = self.separate_branch(transformed_endpoints)
            if i > 0:
                prev_lower = self.principal_axes.inverse_transform(prev_upper)
                curr_upper = self.principal_axes.inverse_transform(ellipse_points[0])
                gaps.append((prev_lower, curr_upper))

            cylinder_mask = self.analyse_segment(ellipses)

            if not np.any(cylinder_mask):
                cylinder_mask = self.ellipse_point_mask(transformed_endpoints)
                base_points_added = np.count_nonzero(cylinder_mask)
                if self.verbose:
                    print(
                        f"    Segment {i+1}: no cylinder points; "
                        f"added {base_points_added} base points"
                    )

            np.logical_or(smooth_cylinder, cylinder_mask, out=smooth_cylinder)

            if self.verbose:
                total_added = np.count_nonzero(cylinder_mask)
                print(f"    Segment {i+1}: Added {total_added} points to cylinder")

            prev_upper = ellipse_points[1]

            if i == 0:
                on_first_base = ellipse_points[0]
            on_second_base = ellipse_points[1]

        _, major_axis, minor_axis = ellipses[1]
        major_len = np.linalg.norm(major_axis)
        minor_len = np.linalg.norm(minor_axis)
        thickness = min(major_len, minor_len)

        if self.verbose:
            print(
                f"  Branch thickness: {thickness:.2f}, total points: {np.count_nonzero(smooth_cylinder)}"
            )

        return (
            smooth_cylinder,
            [on_first_base, on_second_base],
            gaps,
            thickness,
        )

    def smooth_branch_points(self, branch, points):
        """Model a branch using coordinates only, without full-volume temporaries."""
        if self.verbose:
            print("  Starting branch smoothing...")

        self.initialise(branch, points=points)
        best_score = -1
        self.best_cylinder = np.zeros(len(self.points), dtype=bool)
        self.aggregated_gaps = []

        no_improve_count = 0

        for option_index, indices in enumerate(self.indices_options):
            if self.verbose:
                print(
                    f"  Trying branch option {option_index + 1}/"
                    f"{len(self.indices_options)}..."
                )

            smooth_cylinder, bases, gaps, thickness = self.analyse_indices_option(
                indices, branch
            )
            first_base, second_base = bases
            score = np.count_nonzero(smooth_cylinder)

            if self.verbose:
                print(f"  Option {option_index + 1} score: {score} points")

            if score > best_score:
                if self.verbose:
                    print(
                        f"  Found better option! Score improved from {best_score} to {score}"
                    )
                best_score = score
                self.best_cylinder = smooth_cylinder
                self.first_base = self.principal_axes.inverse_transform(first_base)
                self.second_base = self.principal_axes.inverse_transform(second_base)
                self.aggregated_gaps = gaps
                self.thickness = thickness
                no_improve_count = 0
            elif score == 0:
                if self.verbose:
                    print("  Skipping option with zero score")
                continue
            else:
                no_improve_count += 1
                if no_improve_count >= 2:
                    if self.verbose:
                        print(
                            f"  No improvement for {no_improve_count} iterations, stopping early"
                        )
                    break

        if self.verbose:
            print(
                f"  Branch smoothing complete: {np.count_nonzero(self.best_cylinder)} points before gap filling"
            )

        return (
            self.points[self.best_cylinder],
            self.first_base,
            self.second_base,
            self.thickness,
            self.aggregated_gaps,
        )

    def smooth_branch(self, branch, image):
        """Backward-compatible full-volume wrapper around coordinate processing."""
        selected_points, first_base, second_base, thickness, gaps = (
            self.smooth_branch_points(branch, np.argwhere(image == 1))
        )
        best_cylinder = np.zeros(image.shape, dtype=bool)
        if len(selected_points):
            best_cylinder[tuple(selected_points.T)] = True
        for lower, upper in gaps:
            fill_gaps(lower, upper, best_cylinder, cast_to_int=False)
        self.best_cylinder = best_cylinder.astype(int)
        return self.best_cylinder, first_base, second_base, thickness
