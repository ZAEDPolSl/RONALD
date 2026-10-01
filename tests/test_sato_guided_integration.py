import numpy as np
import pytest
from ronald.modelling.model_branch import BranchAnalyser
from ronald.modelling.principal_axes import PrincipalAxes3D


class EllipseSampler:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.mask = np.ones((40, 40, 40), np.uint8)
        self.pitch = 0.5

    def sample(self, center, frame):
        y, x = np.meshgrid(
            np.arange(-self.a, self.a + 0.01, 0.5),
            np.arange(-self.b, self.b + 0.01, 0.5),
        )
        inside = (y / self.a) ** 2 + (x / self.b) ** 2 <= 1
        points = np.column_stack((np.zeros(inside.sum()), y[inside], x[inside]))
        return center + points @ frame, {"reason": "accepted"}


def make_analyser():
    a = BranchAnalyser(
        local_cross_section_sampler=EllipseSampler(9, 9),
        sato_cross_section_sampler=EllipseSampler(3, 1),
    )
    frame = PrincipalAxes3D()
    frame.components_ = np.eye(3)
    frame.mean_ = np.zeros(3)
    a.principal_axes = frame
    a._base_principal_axes = frame
    a._select_segment_pca_frame = lambda direction: None
    return a


def test_local_sections_preserve_sato_proportions_through_caps():
    a = make_analyser()
    ends = np.array([[5.0, 15.0, 15.0], [10.0, 15.0, 15.0]])
    ellipses, _ = a.separate_branch(ends)
    for e in ellipses:
        assert np.linalg.norm(e[1]) / np.linalg.norm(e[2]) == pytest.approx(3, rel=1e-5)
        assert np.linalg.norm(e[1]) > 8
    assert all(x["status"] == "sato_guided" for x in a._last_inscribed_fits)


def test_refines_existing_skeleton_stations_at_uncovered_bulge(monkeypatch):
    a = make_analyser()
    branch = np.column_stack((np.arange(11), np.full(11, 15), np.full(11, 15)))
    points = np.array(
        [
            (z, y, x)
            for z in range(11)
            for y in range(12, 19)
            for x in range(12, 19)
            if (y - 15) ** 2 + (x - 15) ** 2 <= (3 if 4 <= z <= 6 else 1) ** 2
        ]
    )
    a.points = points
    a.transformed_points = points.astype(float)
    a._sato_candidate_mask = np.ones(len(points), bool)

    def fit(endpoints):
        a._last_recovery_unresolved = False
        ellipses = []
        for e in endpoints:
            r = 3.1 if 4 <= e[0] <= 6 else 1.1
            ellipses.append(
                (e.copy(), np.array([0.0, r, 0.0]), np.array([0.0, 0.0, r]))
            )
        return ellipses, [points, points]

    monkeypatch.setattr(a, "separate_branch", fit)
    refined = a._refine_sato_stations(
        np.array([0, 10]), branch, max_added=3, max_depth=2
    )
    assert refined[0] == 0 and refined[-1] == 10
    assert len(refined) <= 9 and len(refined) > 2
    assert np.all(np.diff(refined) > 0)
    assert any(4 <= p <= 6 for p in refined)
    assert a._last_sato_refinement["added_indices"]


def test_unchanged_tube_is_not_subdivided(monkeypatch):
    a = make_analyser()
    branch = np.column_stack((np.arange(11), np.full(11, 15), np.full(11, 15)))
    a.points = branch.copy()
    a.transformed_points = branch.astype(float)
    a._sato_candidate_mask = np.ones(11, bool)
    indices = a._refine_sato_stations(np.array([0, 10]), branch)
    np.testing.assert_array_equal(indices, [0, 10])
