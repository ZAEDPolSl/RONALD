import numpy as np
from ronald.modelling.model_branch import BranchAnalyser
from ronald.modelling.principal_axes import PrincipalAxes3D


def analyser_for(sampler):
    a = BranchAnalyser(
        local_cross_section_sampler=sampler, sato_cross_section_sampler=sampler
    )
    frame = PrincipalAxes3D()
    frame.components_ = np.eye(3)
    frame.mean_ = np.zeros(3)
    a.principal_axes = frame
    return a


class RectangleSampler:
    pitch = 0.5

    def sample(self, center, frame):
        y, z = np.meshgrid(np.arange(-3.0, 3.01, 0.5), np.arange(-2.0, 2.01, 0.5))
        points = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
        return points @ frame + center, {"status": "accepted"}


def test_local_fit_not_clipped_by_branch_points():
    a = analyser_for(RectangleSampler())
    # Both branch ownership and longitudinal extent exclude most real support.
    ends = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ellipses, _ = a.separate_branch(ends)
    for ellipse, endpoint in zip(ellipses, ends):
        np.testing.assert_array_equal(ellipse[0], endpoint)
        np.testing.assert_allclose(
            [np.linalg.norm(v) for v in ellipse[1:]],
            np.sqrt(2) * np.array([3.0, 2.0]),
            rtol=1e-5,
        )
    assert a._last_inscribed_recovery["counts"]["original"] == 2


def test_actual_sampler_crosses_branch_assignment_boundary():
    from ronald.modelling.local_cross_section import LocalCrossSectionSampler

    z, y, x = np.indices((15, 31, 31))
    mask = (y - 15) ** 2 + (x - 15) ** 2 <= 8**2
    a = analyser_for(LocalCrossSectionSampler(mask))
    ellipses, _ = a.separate_branch(np.array([[5.0, 15.0, 15.0], [9.0, 15.0, 15.0]]))
    for e in ellipses:
        assert min(np.linalg.norm(v) for v in e[1:]) > 7
        assert max(np.linalg.norm(v) for v in e[1:]) < 9


def test_local_e2_samples_along_segment_and_restores_center():
    class MissingEndpoint(RectangleSampler):
        def sample(self, center, frame):
            if center[0] < 0.2:
                return np.empty((0, 3)), {"status": "empty"}
            return super().sample(center, frame)

    a = analyser_for(MissingEndpoint())
    ends = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    ellipses, _ = a.separate_branch(ends)
    np.testing.assert_array_equal(ellipses[0][0], ends[0])
    assert a._last_inscribed_recovery["bases"][0]["status"] == "resampled"
    np.testing.assert_allclose(
        a._last_inscribed_fits[-1]["sample_center"], [0.25, 0.5, 0.75]
    )


def test_local_e1_borrows_only_after_failed_search():
    class MissingFirstHalf(RectangleSampler):
        def sample(self, center, frame):
            if center[0] <= 1.0:
                return np.empty((0, 3)), {"status": "empty"}
            return super().sample(center, frame)

    a = analyser_for(MissingFirstHalf())
    ends = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ellipses, _ = a.separate_branch(ends)
    assert a._last_inscribed_recovery["bases"][0]["status"] == "borrowed"
    np.testing.assert_array_equal(ellipses[0][0], ends[0])
    np.testing.assert_allclose(ellipses[0][1:], ellipses[1][1:])


def test_both_missing_never_revert_to_branch_owned_support():
    class Empty:
        pitch = 0.5

        def sample(self, *a):
            return np.empty((0, 3)), {"status": "empty"}

    a = analyser_for(Empty())
    ellipses, _ = a.separate_branch(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert a._last_recovery_unresolved
    assert a._last_inscribed_recovery["unresolved"] == [0, 1]
    assert all(np.array_equal(e[1:], np.zeros((2, 3))) for e in ellipses)
