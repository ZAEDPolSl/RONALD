"""Candidate selection keeps supported walls while prioritizing Sato coverage."""

import json
from types import SimpleNamespace
import numpy as np
import pytest
from ronald.modelling.model_branch import BranchAnalyser
from ronald.modelling.principal_axes import PrincipalAxes3D


def analyser(**kwargs):
    sampler = SimpleNamespace(mask=np.ones((50, 50, 50), np.uint8))
    return BranchAnalyser(
        local_cross_section_sampler=sampler,
        sato_cross_section_sampler=sampler,
        **kwargs,
    )


def run_options(monkeypatch, masks, factors=None, sato=None, **kwargs):
    a = analyser(**kwargs)
    points = np.column_stack((np.arange(len(masks[0])), np.zeros((len(masks[0]), 2))))
    calls = []

    def initialise(*args, **kw):
        a.points = points
        a.principal_axes = PrincipalAxes3D().fit(points)
        a.indices_options = [np.array([0, len(points) - 1]) for _ in masks]
        a._sato_candidate_mask = np.asarray(
            sato if sato is not None else [1, 1, 0, 0, 0], bool
        )

    def analyse(indices, branch):
        i = len(calls)
        calls.append(i)
        a._last_actual_indices = indices.copy()
        a._option_expansion_samples = (
            [(0.0, 1.0), (10.0, 1.0)] if factors is None else factors[i]
        )
        return np.asarray(masks[i], bool), [points[:1], points[-1:]], [], 1.0

    monkeypatch.setattr(a, "initialise", initialise)
    monkeypatch.setattr(a, "analyse_indices_option", analyse)
    a.smooth_branch_points(points, points, pca_points=points)
    json.dumps(a.candidate_scores, allow_nan=False)
    return a, calls


def test_sato_coverage_beats_larger_wall_volume(monkeypatch):
    a, _ = run_options(monkeypatch, [[1, 0, 1, 1, 1], [1, 1, 0, 0, 0]])
    assert a.selected_option_index == 1
    assert a.candidate_scores[1]["sato_retained"] == 2
    assert a.candidate_scores[1]["selected"]


def test_same_coverage_prefers_smoother_growth_not_smaller_growth(monkeypatch):
    masks = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    factors = [[(0.0, 1.0), (10.0, 2.0)], [(0.0, 3.0), (10.0, 3.0)]]
    a, _ = run_options(monkeypatch, masks, factors=factors)
    assert a.selected_option_index == 1
    assert a.candidate_scores[1]["expansion_roughness"] == 0


def test_equal_coverage_and_smoothness_retains_supported_walls(monkeypatch):
    a, _ = run_options(monkeypatch, [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    assert a.selected_option_index == 1
    assert a.best_cylinder.sum() == 5


def test_two_nonimproving_candidates_still_stop_search(monkeypatch):
    a, calls = run_options(
        monkeypatch,
        [[1, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 1, 1, 0], [1, 1, 1, 1, 1]],
    )
    assert calls == [0, 1, 2]
    assert a.selected_option_index == 0


def test_roughness_is_finite_and_does_not_penalize_absolute_growth():
    assert BranchAnalyser._expansion_roughness([(0, 4), (10, 4)]) == 0
    assert np.isfinite(
        BranchAnalyser._expansion_roughness([(0, np.nan), (1, 2), (2, np.inf)])
    )
    assert BranchAnalyser._expansion_roughness([(0, 1), None, (2, 10)]) == 0


def synthetic_refinement(monkeypatch, *, covered=False):
    a = analyser()
    frame = PrincipalAxes3D()
    frame.components_ = np.eye(3)
    frame.mean_ = np.zeros(3)
    a.principal_axes = frame
    monkeypatch.setattr(a, "_select_segment_pca_frame", lambda direction: None)
    monkeypatch.setattr(a, "_restore_base_pca_frame", lambda: None)
    a._last_recovery_unresolved = False
    branch = np.column_stack((np.arange(33), np.full(33, 15), np.full(33, 15)))
    a.points = np.array([(z, 15 + dy, 15) for z in range(33) for dy in (-2, 0, 2)])
    a.transformed_points = a.points.astype(float)
    a._sato_candidate_mask = np.ones(len(a.points), bool)

    def fit(ends):
        a._last_recovery_unresolved = False
        radius = 3.0 if covered else 0.5
        return [
            (e.copy(), np.array([0.0, radius, 0.0]), np.array([0.0, 0.0, radius]))
            for e in ends
        ], []

    monkeypatch.setattr(a, "separate_branch", fit)
    indices = a._refine_sato_stations(
        np.array([0, 32]), branch, max_added=1, max_depth=1
    )
    return a, indices


def test_extra_refinement_only_after_limit_and_measured_miss(monkeypatch):
    a, indices = synthetic_refinement(monkeypatch)
    diag = a._last_sato_refinement
    assert diag["extra_indices"]
    assert len(diag["extra_indices"]) <= 4
    assert len(diag["added_indices"]) <= 5
    assert len(indices) <= 7
    for c in diag["checks"]:
        if c["extra_station"]:
            assert c["normal_limit_reached"]
            assert c["missed_sato"] >= 3 and c["missed_sato"] / c["sato_voxels"] > 0.05
    assert diag["residual_sections"]
    assert max(c["depth"] for c in diag["checks"]) <= 2


def test_no_extra_refinement_without_residual_miss(monkeypatch):
    a, indices = synthetic_refinement(monkeypatch, covered=True)
    assert indices.tolist() == [0, 32]
    assert not a._last_sato_refinement["extra_indices"]
    assert not a._last_sato_refinement["residual_sections"]


def test_adjacent_stations_do_not_trigger_unusable_fits(monkeypatch):
    a = analyser()
    branch = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint16)

    def unexpected(*args, **kwargs):
        pytest.fail("Adjacent stations have no room for refinement")

    monkeypatch.setattr(a, "separate_branch", unexpected)
    np.testing.assert_array_equal(
        a._refine_sato_stations(np.array([0, 1]), branch), [0, 1]
    )
    assert a._last_sato_refinement["checks"] == []


def test_unsigned_decreasing_skeleton_uses_physical_arc_length(monkeypatch):
    a = analyser()
    frame = PrincipalAxes3D()
    frame.components_ = np.eye(3)
    frame.mean_ = np.zeros(3)
    a.principal_axes = frame
    monkeypatch.setattr(a, "_select_segment_pca_frame", lambda direction: None)
    monkeypatch.setattr(a, "_restore_base_pca_frame", lambda: None)
    a._last_recovery_unresolved = False
    branch = np.array([[10, 5, 5], [8, 5, 5], [5, 5, 5]], dtype=np.uint16)
    a.points = branch.copy()
    monkeypatch.setattr(a, "_refine_sato_stations", lambda indices, branch: indices)

    def separate(endpoints):
        a._last_sato_expansion_factors = [2.0, 2.0]
        ellipses = [
            (point, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
            for point in endpoints
        ]
        return ellipses, []

    def segment(ellipses):
        a._last_capped_ellipses = ellipses
        return np.ones(len(a.points), bool)

    monkeypatch.setattr(a, "separate_branch", separate)
    monkeypatch.setattr(a, "analyse_segment", segment)
    a.analyse_indices_option(np.array([0, 2]), branch)
    assert a._option_expansion_samples == [(0.0, 2.0), (5.0, 2.0)]


def test_no_sato_candidates_use_wall_coverage(monkeypatch):
    a, _ = run_options(monkeypatch, [[1, 1, 0, 0, 0], [1, 0, 1, 1, 1]], sato=[0] * 5)
    assert a.selected_option_index == 1
    assert not a.candidate_scores[0]["coverage_first"]


def test_samplers_are_required():
    with pytest.raises(TypeError):
        BranchAnalyser()
    with pytest.raises(ValueError, match="samplers"):
        BranchAnalyser(
            local_cross_section_sampler=None, sato_cross_section_sampler=None
        )
