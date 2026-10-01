import numpy as np
import pytest

from ronald.processing.gmm_thresholding import get_thresholds


def component(mean, std):
    return {"mean": mean, "std": std, "weight": 0.5}


@pytest.mark.parametrize("left,right,expected", [
    (component(-578.0554821787466, 189.06853844229454),
     component(-5.608920378762586, 54.646701126343544), -155.12717121576978),
    (component(-596.2197830153882, 175.5175142380976),
     component(-4.018785451624684, 67.20722901509302), -186.47016333369837),
])
def test_patient_wall_threshold_uses_crossing_between_means(left, right, expected):
    for maximum in (500, 100, 4095):
        threshold, = get_thresholds([left, right], maximum)
        assert threshold == pytest.approx(expected)
        assert left["mean"] < threshold < right["mean"]
        # It is an actual density crossing, not an arbitrary HU clamp.
        log_density = lambda c: -np.log(c["std"]) - (threshold-c["mean"])**2/(2*c["std"]**2)
        assert log_density(left) == pytest.approx(log_density(right))


def test_normal_threshold_is_unchanged():
    threshold, = get_thresholds([
        component(-846.2338528379681, 55.693802047318805),
        component(-578.0554821787466, 189.06853844229454)], 500)
    assert threshold == pytest.approx(-746.0745953191114)


def test_equal_variances_and_adjacent_threshold_order():
    values = get_thresholds([component(-900, 50), component(-600, 50),
                             component(-100, 50)], 500)
    assert values == pytest.approx([-750, -350])


def test_no_between_mean_crossing_reports_midpoint_fallback():
    with pytest.warns(RuntimeWarning, match="using midpoint"):
        values = get_thresholds([component(-600, 1), component(-599.9, 100)], 500)
    assert values == pytest.approx([-599.95])


def test_identical_components_do_not_produce_an_invalid_root():
    with pytest.warns(RuntimeWarning, match="using midpoint"):
        assert get_thresholds([component(-600, 10), component(-600, 10)], 500) == [-600]


def test_nearly_equal_variances_with_large_offset():
    values = get_thresholds([component(1e8, 50), component(1e8+200, 50+1e-10)], 1e9)
    assert values == pytest.approx([1e8+100], abs=1e-5, rel=0)


@pytest.mark.parametrize("parts", [
    [component(-100, 10), component(-600, 10)],
    [component(-600, 0), component(-100, 10)],
    [component(float("nan"), 10), component(-100, 10)],
])
def test_invalid_parameters_are_rejected(parts):
    with pytest.raises(ValueError):
        get_thresholds(parts, 500)


def test_threshold_only_preserves_thresholds_and_skips_volume(monkeypatch):
    import SimpleITK as sitk
    from threadpoolctl import threadpool_limits
    from ronald.processing import gmm_thresholding as module

    rng = np.random.default_rng(42)
    image = sitk.GetImageFromArray(rng.integers(-1000, 1000, (12, 16, 20), dtype=np.int16))
    image.SetSpacing((0.7, 0.8, 1.2))
    image.SetOrigin((10.0, -20.0, 3.0))
    mask = sitk.GetImageFromArray(np.ones((12, 16, 20), dtype=np.uint8))
    mask.CopyInformation(image)
    with threadpool_limits(limits=1):
        np.random.seed(42)
        segments, expected = module.run_thresholding(image, mask)
        assert segments.GetSpacing() == image.GetSpacing()
        assert segments.GetOrigin() == image.GetOrigin()
        assert segments.GetDirection() == image.GetDirection()
        def unexpected_volume(*args, **kwargs):
            pytest.fail("Threshold-only path must not allocate a segmented volume")
        monkeypatch.setattr(module, "create_thresholded_volumes", unexpected_volume)
        np.random.seed(42)
        result, actual = module.run_thresholding(image, mask, create_segments=False)
    assert result is None
    assert actual == expected


def test_sample_selection_preserves_order_dtype_and_boundary_rules(monkeypatch):
    import SimpleITK as sitk
    from ronald.processing import gmm_thresholding as module

    image = np.array([[[-1000, 500, 501, -200],
                       [-900, -1000, 700, 499]]], dtype=np.int16)
    captured = {}
    class FakeGMM:
        def __init__(self, n_components):
            assert n_components == 3
        def fit(self, values):
            captured["values"] = values.copy()
    monkeypatch.setattr(module.mixture, "GaussianMixture", FakeGMM)
    monkeypatch.setattr(module, "get_gmm_metadata", lambda _: [
        component(-800, 50), component(-400, 50), component(100, 50)])
    _, thresholds = module.run_thresholding(sitk.GetImageFromArray(image), create_segments=False)
    expected = np.array([500, -200, -900, 499], dtype=np.int16)[:, None]
    np.testing.assert_array_equal(captured["values"], expected)
    assert captured["values"].dtype == image.dtype
    assert thresholds == [-1001, -600, -150, 701]
