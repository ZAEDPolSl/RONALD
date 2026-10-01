from collections import Counter
import numpy as np
from ronald.modelling.cone_construction import is_point_in_cylinder
from ronald.modelling.densify import densify_point_cloud
from ronald.modelling.ellipse import check_ellipse
from ronald.modelling.ellipse_recovery import ellipse_is_valid
from ronald.modelling.principal_axes import PrincipalAxes3D
from ronald.modelling.segment_branch import segment_branch
from ronald.modelling.sato_envelope import fit_sato_guided_ellipse

DENSIFICATION_FACTOR = 100


class BranchAnalyser:
    """Fit branch tubes around Sato sections, expanding only into supported walls."""

    def __init__(
        self,
        eps=1e-10,
        segments=True,
        verbose=False,
        *,
        capture_geometry=False,
        local_cross_section_sampler,
        sato_cross_section_sampler,
    ):
        if local_cross_section_sampler is None or sato_cross_section_sampler is None:
            raise ValueError("Both wall and Sato cross-section samplers are required")
        self.local_cross_section_sampler = local_cross_section_sampler
        self.sato_cross_section_sampler = sato_cross_section_sampler
        self.candidate_scores = []
        self.capture_geometry = capture_geometry
        self.selected_segments = []
        self.selected_option_index = None
        self.selected_indices = []
        self.fit_diagnostics = Counter()
        self.eps = eps
        self.segments = segments
        self.verbose = verbose
        self.principal_axes = None
        self.indices_options = []
        self.points = None
        self.transformed_points = None
        self.aggregated_gaps = []
        self.best_cylinder = None
        self.first_base = None
        self.second_base = None

    @staticmethod
    def ellipse_boundary(ellipse, count=64):
        """Sample the fitted ellipse boundary for joins and geometry export."""
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        return (
            ellipse[0]
            + np.cos(angles)[:, None] * ellipse[1]
            + np.sin(angles)[:, None] * ellipse[2]
        )

    def prepare_principal_axes(self, points, *, pca_points):
        """Densify Sato direction support while retaining original wall candidates.

        Dense local mask sections supply ellipse fits later. Densifying the
        branch-owned wall cloud would bias those sections at ownership borders.
        """
        pca_points = np.asarray(pca_points)
        if (
            pca_points.ndim != 2
            or pca_points.shape[1] != 3
            or len(pca_points) < 2
            or not np.all(np.isfinite(pca_points))
        ):
            raise ValueError("pca_points must contain at least two finite 3D points")
        if not np.any(pca_points != pca_points[0]):
            raise ValueError("pca_points must contain at least two distinct points")
        if self.verbose:
            print("  Preparing branch PCA...")
        direction_points = densify_point_cloud(pca_points, factor=DENSIFICATION_FACTOR)
        self.principal_axes = PrincipalAxes3D().fit(direction_points)
        del direction_points
        self.transformed_points = self.principal_axes.transform(points)
        self._base_principal_axes = self.principal_axes
        self._base_transformed_points = self.transformed_points
        if self.verbose:
            print(f"  PCA prepared: {len(points)} points transformed")

    def _select_segment_pca_frame(self, direction):
        """P1: select an existing PCA eigenvector; never substitute a tangent."""
        try:
            frame = self._base_principal_axes.reordered_for_direction(direction)
        except ValueError:
            self.fit_diagnostics["pca_degenerate_guides"] += 1
            self._restore_base_pca_frame()
            return
        self.principal_axes = frame
        if frame.axis_order_[0] != 0:
            self.fit_diagnostics["pca_axis_reorders"] += 1
        if np.array_equal(frame.axis_order_, np.arange(3)) and np.all(
            frame.axis_signs_ == 1
        ):
            self.transformed_points = self._base_transformed_points
        else:
            self.transformed_points = (
                self._base_transformed_points[:, frame.axis_order_] * frame.axis_signs_
            )

    def _restore_base_pca_frame(self):
        self.principal_axes = self._base_principal_axes
        self.transformed_points = self._base_transformed_points

    def separate_branch(self, endpoints):
        return self._separate_local_branch(endpoints)

    def _separate_local_branch(self, endpoints):
        """Fit full local mask sections independently of voxel branch ownership.

        E2 samples up to one longitudinal voxel inward along the same skeleton
        segment. E1 borrows the opposite base only after local searches finish.
        Recovered dimensions return to the original skeleton endpoint; their
        containment applies to the sampled section, not the relocated section.
        """
        self._last_inscribed_fits = []
        statuses = [
            dict(status="unresolved", valid=False, width=0.5, shift=0.0, attempts=0)
            for _ in range(2)
        ]
        expansion_factors = [None, None]
        last_expansion = None

        def measure(anchor):
            nonlocal last_expansion
            last_expansion = None
            world = self.principal_axes.inverse_transform(np.asarray([anchor]))[0]
            points, sampling = self.local_cross_section_sampler.sample(
                world, self.principal_axes.components_
            )
            projected = self.principal_axes.transform(points)
            sato_world, sato_sampling = self.sato_cross_section_sampler.sample(
                world, self.principal_axes.components_
            )
            sato_projected = self.principal_axes.transform(sato_world)
            fitted, diagnostic = fit_sato_guided_ellipse(
                sato_projected,
                projected,
                anchor,
                wall_sample_pitch=self.local_cross_section_sampler.pitch,
            )
            self._last_inscribed_fits.append(
                dict(
                    diagnostic,
                    sample_center=anchor.tolist(),
                    local_cross_section=sampling,
                    sato_cross_section=sato_sampling,
                )
            )
            self.fit_diagnostics[
                "sato_guided_" + ("fitted" if fitted is not None else "failed")
            ] += 1
            if fitted is not None:
                last_expansion = diagnostic.get("expansion_factor")
            ellipse = (
                fitted
                if fitted is not None
                else (anchor.copy(), np.zeros(3), np.zeros(3))
            )
            return (ellipse, sato_projected)

        ellipses, bases = ([], [])
        for i, endpoint in enumerate(endpoints):
            ellipse, sample = measure(endpoint)
            ellipses.append(ellipse)
            bases.append(sample)
            if ellipse_is_valid(ellipse, sample, eps=self.eps):
                statuses[i].update(status="original", valid=True)
                expansion_factors[i] = last_expansion
        span = endpoints[1, 0] - endpoints[0, 0]
        limit = min(1.0, abs(span) / 2)
        if limit > self.eps:
            for i in range(2):
                if statuses[i]["valid"]:
                    continue
                for shift in sorted(set((min(limit, x) for x in (0.25, 0.5, 1.0)))):
                    fraction = shift / abs(span)
                    anchor = endpoints[i] + fraction * (endpoints[1 - i] - endpoints[i])
                    statuses[i]["attempts"] += 1
                    ellipse, sample = measure(anchor)
                    if not ellipse_is_valid(ellipse, sample, eps=self.eps):
                        continue
                    ellipses[i] = (endpoints[i].copy(), ellipse[1], ellipse[2])
                    bases[i] = sample + endpoints[i] - anchor
                    statuses[i].update(status="resampled", valid=True, shift=shift)
                    expansion_factors[i] = last_expansion
                    break
        for i in range(2):
            if statuses[i]["valid"] or not statuses[1 - i]["valid"]:
                continue
            other = ellipses[1 - i]
            ellipses[i] = (endpoints[i].copy(), other[1].copy(), other[2].copy())
            bases[i] = self.ellipse_boundary(ellipses[i])
            statuses[i].update(status="borrowed", valid=True, width=0.0)
            expansion_factors[i] = expansion_factors[1 - i]
        self._last_sato_expansion_factors = expansion_factors
        unresolved = [i for i, base in enumerate(statuses) if not base["valid"]]
        self._last_recovery_unresolved = bool(unresolved)
        self._last_inscribed_recovery = dict(
            bases=statuses,
            unresolved=unresolved,
            counts={
                name: sum((b["status"] == name for b in statuses))
                for name in ("original", "resampled", "borrowed", "unresolved")
            },
        )
        for base in statuses:
            self.fit_diagnostics["ellipse_" + base["status"]] += 1
        return (ellipses, bases)

    def ellipse_point_mask(self, transformed_endpoints, *, ellipses):
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
        ellipse1, ellipse2 = ellipses
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
        """Select supported wall voxels inside the interpolated Sato-guided bases.

        Do not cap an ellipse by its predecessor: each base must enclose its
        measured Sato section. Walls are clipped after tube construction.
        """
        first, second = ellipses
        self._last_capped_ellipses = (first, second)
        return is_point_in_cylinder(
            self.transformed_points,
            first[0],
            second[0],
            first[1],
            first[2],
            second[1],
            second[2],
        )

    def initialise(self, branch, points, *, pca_points):
        if self.verbose:
            print("  Initializing branch analysis...")
        self.points = np.asarray(points)
        self._sato_candidate_mask = (
            self.sato_cross_section_sampler.mask[tuple(self.points.T)] > 0
        )
        self.fit_diagnostics.clear()
        if self.verbose:
            print(f"  Found {len(self.points)} points in branch mask")
        self.prepare_principal_axes(self.points, pca_points=pca_points)
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

    def _refine_sato_stations(self, indices, branch, *, max_added=8, max_depth=3):
        """Refine missing Sato locally, with optional four-station reserve.

        The reserve is used only for measured >5% (at least three voxels)
        misses at a normal depth or station limit. It permits one extra depth
        level, never a second complete fitting pass. Leaf checks report the
        remaining misses; these overlapping section counts are not a global
        unique-voxel coverage metric.
        """
        from scipy.spatial import KDTree

        added, checks, extra_added = ([], [], [])
        base_added = 0

        def refine(start, end, depth):
            nonlocal base_added
            at_limit = depth >= max_depth or base_added >= max_added
            # Adjacent stations cannot be refined: avoid two unused fits.
            if end - start <= 1:
                return [start, end]
            self._select_segment_pca_frame(branch[end] - branch[start])
            endpoints = self.principal_axes.transform(branch[[start, end]])
            low, high = np.sort(endpoints[:, 0])
            if high - low < 1.0:
                return [start, end]
            relevant = (
                self._sato_candidate_mask
                & (self.transformed_points[:, 0] >= low)
                & (self.transformed_points[:, 0] <= high)
            )
            total = int(relevant.sum())
            if total == 0:
                return [start, end]
            ellipses, _ = self.separate_branch(endpoints)
            check = dict(
                start=int(start),
                end=int(end),
                sato_voxels=total,
                missed_sato=None,
                depth=depth,
                inserted_index=None,
                normal_limit_reached=bool(at_limit),
                extra_station=False,
                unresolved=bool(self._last_recovery_unresolved),
            )
            checks.append(check)
            if self._last_recovery_unresolved:
                return [start, end]
            first, second = ellipses
            covered = is_point_in_cylinder(
                self.transformed_points,
                first[0],
                second[0],
                first[1],
                first[2],
                second[1],
                second[2],
            )
            missing = relevant & ~covered
            count = int(missing.sum())
            check["missed_sato"] = count
            if count < 3 or count / total <= 0.05:
                return [start, end]
            if at_limit and (len(extra_added) >= 4 or depth >= max_depth + 1):
                check["stop_reason"] = "refinement_limit"
                return [start, end]
            closest = KDTree(branch[start + 1 : end]).query(self.points[missing])[1]
            station = (
                start
                + 1
                + int(np.argmax(np.bincount(closest, minlength=end - start - 1)))
            )
            added.append(station)
            check["inserted_index"] = station
            if at_limit:
                extra_added.append(station)
                check["extra_station"] = True
            else:
                base_added += 1
            left = refine(start, station, depth + 1)
            right = refine(station, end, depth + 1)
            return left[:-1] + right

        result = []
        # The endpoint-only option uses -1 for the final skeleton point.
        indices = np.asarray(indices, dtype=int) % len(branch)
        for start, end in zip(indices[:-1], indices[1:]):
            part = refine(int(start), int(end), 0)
            result.extend(part if not result else part[1:])
        residual = [
            check
            for check in checks
            if check["inserted_index"] is None
            and (
                check["unresolved"]
                or (
                    check["missed_sato"] >= 3
                    and check["missed_sato"] / check["sato_voxels"] > 0.05
                )
            )
        ]
        self._last_sato_refinement = dict(
            added_indices=added,
            checks=checks,
            max_added=max_added,
            max_depth=max_depth,
            miss_fraction=0.05,
            targeted=True,
            extra_indices=extra_added,
            normal_limit_checks=sum((c["normal_limit_reached"] for c in checks)),
            residual_sections=residual,
        )
        self.fit_diagnostics["sato_added_stations"] += len(added)
        self.fit_diagnostics["sato_targeted_extra_stations"] += len(extra_added)
        self.fit_diagnostics["sato_refinement_residual_sections"] += len(residual)
        return np.asarray(result, dtype=int)

    @staticmethod
    def _expansion_roughness(samples):
        """Log-growth total variation per skeleton voxel, including frame seams.

        Absolute growth is not penalized: a uniformly expanded tube is as
        smooth as an unexpanded one. Missing/nonfinite samples cannot enter
        the score. Separate valid runs do not bridge unresolved sections.
        """
        variation = span = 0.0
        previous = None
        for sample in samples:
            if sample is None:
                previous = None
                continue
            position, factor = sample
            if factor is None or not np.isfinite(factor) or factor <= 0:
                previous = None
                continue
            current = (float(position), float(np.log(factor)))
            if previous is not None:
                variation += abs(current[1] - previous[1])
                span += max(0.0, current[0] - previous[0])
            previous = current
        return float(variation / max(span, 1.0))

    def analyse_indices_option(self, indices, branch):
        indices = self._refine_sato_stations(indices, branch)
        self._last_actual_indices = np.asarray(indices).copy()
        if self.verbose:
            print(f"  Analyzing branch option with {len(indices)} segments...")
        if self.capture_geometry:
            self._option_segments = []
        smooth_cylinder = np.zeros(len(self.points), dtype=bool)
        self._option_expansion_samples = []
        arc_positions = np.r_[
            0.0,
            np.cumsum(
                np.linalg.norm(np.diff(np.asarray(branch, dtype=float), axis=0), axis=1)
            ),
        ]
        gaps = []
        on_first_base = np.empty((0, 3), dtype=float)
        on_second_base = np.empty((0, 3), dtype=float)
        previous_world_base = None
        last_valid_thickness = 0.0
        for i in range(len(indices) - 1):
            if self.verbose and len(indices) > 2:
                print(f"    Processing segment {i + 1}/{len(indices) - 1}...")
            start_idx, end_idx = (indices[i], indices[i + 1])
            self.fit_diagnostics["segments_evaluated"] += 1
            self._select_segment_pca_frame(branch[end_idx] - branch[start_idx])
            transformed_endpoints = self.principal_axes.transform(
                np.array([branch[start_idx], branch[end_idx]])
            )
            ellipses, ellipse_points = self.separate_branch(transformed_endpoints)
            if self._last_recovery_unresolved or not all(
                (ellipse_is_valid(e, eps=self.eps) for e in ellipses)
            ):
                if self.capture_geometry:
                    self._option_segments.append(
                        dict(
                            status="unresolved",
                            segment_index=i,
                            start_index=int(start_idx),
                            end_index=int(end_idx),
                            centered_inscribed_ellipse=getattr(
                                self, "_last_inscribed_fits", []
                            ),
                            centered_inscribed_recovery=getattr(
                                self, "_last_inscribed_recovery", None
                            ),
                        )
                    )
                self._option_expansion_samples.append(None)
                self.fit_diagnostics["segments_unresolved"] += 1
                previous_world_base = None
                on_second_base = np.empty((0, 3), dtype=float)
                last_valid_thickness = 0.0
                continue
            factors = getattr(self, "_last_sato_expansion_factors", [None, None])
            self._option_expansion_samples.extend(
                zip([arc_positions[start_idx], arc_positions[end_idx]], factors)
            )
            cylinder_mask = self.analyse_segment(ellipses)
            ellipse_points = [
                self.ellipse_boundary(e) for e in self._last_capped_ellipses
            ]
            world_bases = [
                self.principal_axes.inverse_transform(base) for base in ellipse_points
            ]
            if previous_world_base is not None:
                gaps.append((previous_world_base, world_bases[0]))
            used_endpoint_fallback = not np.any(cylinder_mask)
            if used_endpoint_fallback:
                cylinder_mask = self.ellipse_point_mask(
                    transformed_endpoints, ellipses=self._last_capped_ellipses
                )
                base_points_added = np.count_nonzero(cylinder_mask)
                if self.verbose:
                    print(
                        f"    Segment {i + 1}: no cylinder points; added {base_points_added} base points"
                    )
            if self.capture_geometry:
                centers = np.asarray([e[0] for e in self._last_capped_ellipses])
                world_centers = self.principal_axes.inverse_transform(centers)
                majors = np.asarray([e[1] for e in self._last_capped_ellipses])
                minors = np.asarray([e[2] for e in self._last_capped_ellipses])
                self._option_segments.append(
                    dict(
                        status="valid",
                        segment_index=i,
                        start_index=int(start_idx),
                        end_index=int(end_idx),
                        used_endpoint_fallback=bool(used_endpoint_fallback),
                        sato_guided_refinement=getattr(
                            self, "_last_sato_refinement", None
                        )
                        if i == 0
                        else None,
                        selected_point_count=int(np.count_nonzero(cylinder_mask)),
                        centers_zyx=world_centers.copy(),
                        major_vectors_zyx=majors @ self.principal_axes.components_,
                        minor_vectors_zyx=minors @ self.principal_axes.components_,
                        pca_direction_zyx=self.principal_axes.components_[0].copy(),
                        centered_inscribed_ellipse=getattr(
                            self, "_last_inscribed_fits", []
                        ),
                        centered_inscribed_recovery=getattr(
                            self, "_last_inscribed_recovery", None
                        ),
                        symmetry_ellipse=None,
                        robust_ellipse=None,
                        terminal_flare=None,
                        blend_ellipse_zyx=None,
                    )
                )
            np.logical_or(smooth_cylinder, cylinder_mask, out=smooth_cylinder)
            if self.verbose:
                total_added = np.count_nonzero(cylinder_mask)
                print(f"    Segment {i + 1}: Added {total_added} points to cylinder")
            previous_world_base = world_bases[1]
            if i == 0:
                on_first_base = world_bases[0]
            on_second_base = world_bases[1]
            last_valid_thickness = min(
                np.linalg.norm(self._last_capped_ellipses[1][1]),
                np.linalg.norm(self._last_capped_ellipses[1][2]),
            )
        # All section boundaries were saved in world coordinates before changing
        # frames; return them in the original branch frame for the caller.
        self._restore_base_pca_frame()
        on_first_base = self.principal_axes.transform(on_first_base)
        on_second_base = self.principal_axes.transform(on_second_base)
        thickness = last_valid_thickness
        if self.verbose:
            print(
                f"  Branch thickness: {thickness:.2f}, total points: {np.count_nonzero(smooth_cylinder)}"
            )
        return (smooth_cylinder, [on_first_base, on_second_base], gaps, thickness)

    def smooth_branch_points(self, branch, points, *, pca_points):
        """Fit wall candidates using separate Sato direction and section support."""
        if self.verbose:
            print("  Starting branch smoothing...")
        self.initialise(branch, points=points, pca_points=pca_points)
        if self.capture_geometry:
            self.selected_segments = []
            self.selected_option_index = None
            self.selected_indices = []
        best_score = None
        self.candidate_scores = []
        self.selected_option_index = None
        self.best_cylinder = np.zeros(len(self.points), dtype=bool)
        self.aggregated_gaps = []
        no_improve_count = 0
        for option_index, indices in enumerate(self.indices_options):
            if self.verbose:
                print(
                    f"  Trying branch option {option_index + 1}/{len(self.indices_options)}..."
                )
            smooth_cylinder, bases, gaps, thickness = self.analyse_indices_option(
                indices, branch
            )
            first_base, second_base = bases
            wall_count = int(np.count_nonzero(smooth_cylinder))
            sato_mask = getattr(self, "_sato_candidate_mask", None)
            use_coverage = sato_mask is not None and np.any(sato_mask)
            sato_count = (
                int(np.count_nonzero(smooth_cylinder & sato_mask))
                if sato_mask is not None
                else None
            )
            roughness = self._expansion_roughness(
                getattr(self, "_option_expansion_samples", [])
            )
            # Preserve lumen first, then prefer consistent growth. Supported wall
            # volume is only a final tie-break; uniform wall growth is allowed.
            score = (
                (sato_count, -roughness, wall_count) if use_coverage else (wall_count,)
            )
            self.candidate_scores.append(
                dict(
                    option_index=option_index,
                    sato_retained=sato_count,
                    wall_retained=wall_count,
                    expansion_roughness=roughness,
                    coverage_first=bool(use_coverage),
                    actual_station_count=len(
                        getattr(self, "_last_actual_indices", indices)
                    ),
                    refinement=getattr(self, "_last_sato_refinement", None),
                    selected=False,
                )
            )
            if self.verbose:
                print(f"  Option {option_index + 1} score: {score} points")
            if best_score is None or score > best_score:
                if self.verbose:
                    print(
                        f"  Found better option! Score improved from {best_score} to {score}"
                    )
                self.selected_option_index = option_index
                if self.capture_geometry:
                    self.selected_segments = self._option_segments
                    self.selected_option_index = option_index
                    self.selected_indices = self._last_actual_indices.tolist()
                best_score = score
                self.best_cylinder = smooth_cylinder
                self.first_base = self.principal_axes.inverse_transform(first_base)
                self.second_base = self.principal_axes.inverse_transform(second_base)
                self.aggregated_gaps = gaps
                self.thickness = thickness
                no_improve_count = 0
            elif wall_count == 0:
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
        for candidate in self.candidate_scores:
            candidate["selected"] = (
                candidate["option_index"] == self.selected_option_index
            )
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
