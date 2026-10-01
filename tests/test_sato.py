"""Check extraction of the accepted binary-mask Sato/GMM preprocessing."""

import importlib
import numpy as np
import pytest
import SimpleITK as sitk
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

mod = importlib.import_module("ronald.modelling.sato")


def image(array):
    return sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))


def test_gmm_matches_original_weighted_map_and_crossing():
    rng = np.random.default_rng(42)
    values = np.clip(
        np.r_[np.zeros(40), rng.normal(0.13, 0.06, 600), rng.normal(0.75, 0.09, 400)],
        0,
        1,
    )
    with threadpool_limits(1):
        high, info = mod._highest_gmm_component(values)
        original = GaussianMixture(
            n_components=2,
            random_state=42,
            n_init=3,
            max_iter=300,
            tol=1e-4,
            reg_covar=1e-6,
        ).fit(values[:, None])
    order = np.argsort(original.means_.ravel())
    means = original.means_.ravel()[order]
    stds = np.sqrt(original.covariances_.ravel()[order])
    weights = original.weights_[order]

    def score(x, i):
        return np.log(weights[i]) + norm.logpdf(x, means[i], stds[i])

    expected = (
        np.argmax(np.column_stack([score(values, i) for i in range(2)]), axis=1) == 1
    )
    np.testing.assert_array_equal(high, expected)
    assert info["threshold"] == brentq(
        lambda x: score(x, 0) - score(x, 1), means[0], means[1]
    )


def test_scales_max_then_normalization_once_and_only_foreground_enters_gmm(monkeypatch):
    data = np.zeros((5, 6, 7), np.uint8)
    data[1:4, 2:5, 2:6] = 1
    reference = image(data)
    reference.SetSpacing((0.7, 0.8, 1.2))
    reference.SetOrigin((4, 5, 6))
    baseline = data.copy()
    responses = []
    calls = []
    sampled = []

    def filter(binary, sigmas, **kw):
        calls.append((sigmas, kw))
        assert binary.dtype == np.float32
        np.testing.assert_array_equal(binary, data)
        # Max is deliberately outside the mask to verify normalization domain.
        raw = np.arange(binary.size, dtype=np.float32).reshape(binary.shape) * (
            1 + sigmas[0]
        )
        responses.append(raw.copy())
        return raw

    def fit(values):
        sampled.append(values.copy())
        return values >= 0.5, {"threshold": 0.5}

    monkeypatch.setattr(mod, "_parallel_sato", filter)
    monkeypatch.setattr(mod, "_highest_gmm_component", fit)
    out, info = mod.calculate_sato_mask(reference, max_workers=2)
    combined = np.maximum.reduce(responses)
    combined -= combined.min()
    combined /= combined.max()
    np.testing.assert_array_equal(sampled[0], combined[data > 0].astype(np.float64))
    expected = (combined >= 0.5) & (data > 0)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(out), expected)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(reference), baseline)
    assert (
        out.GetSpacing() == reference.GetSpacing()
        and out.GetOrigin() == reference.GetOrigin()
    )
    assert [c[0][0] for c in calls] == list(mod.SIGMAS)
    assert all(c[1] == dict(max_workers=2, eigen_chunk_depth=8) for c in calls)
    assert info["padding_voxels"] == 138


def test_empty_and_constant_responses_do_not_fabricate_foreground(monkeypatch):
    monkeypatch.setattr(
        mod, "_parallel_sato", lambda binary, *a, **k: np.zeros_like(binary)
    )
    for data in (np.zeros((5, 5, 5)), np.ones((5, 5, 5))):
        out, info = mod.calculate_sato_mask(image(data))
        assert not sitk.GetArrayFromImage(out).any()
        assert info["threshold"] is None


@pytest.mark.parametrize("bad", [np.nan, -1, 2])
def test_nonbinary_or_nonfinite_wall_values_rejected(bad):
    data = np.ones((4, 4, 4), float)
    data[1, 1, 1] = bad
    with pytest.raises(ValueError, match="binary"):
        mod.calculate_sato_mask(sitk.GetImageFromArray(data))


def test_actual_small_filter_matches_original_scale_reduction():
    z, y, x = np.indices((13, 13, 13))
    data = (((y - 6) ** 2 + (x - 6) ** 2 <= 4) & (z > 1) & (z < 11)).astype(np.uint8)
    with threadpool_limits(1):
        result, info = mod.calculate_sato_mask(image(data), max_workers=1)
        response = np.zeros(data.shape, np.float32)
        # Original run used four workers; verify worker count changes no mask.
        for sigma in (0.5, 1, 2, 3, 5, 8, 12):
            score = mod._parallel_sato(
                data.astype(np.float32), (sigma,), max_workers=4, eigen_chunk_depth=8
            )
            np.maximum(response, score, out=response)
        response -= float(response.min())
        maximum = float(response.max())
        if maximum > 0:
            response /= maximum
        response *= data
        expected = np.zeros_like(data)
        membership, original = mod._highest_gmm_component(
            response[data > 0].astype(np.float64)
        )
        expected[data > 0] = membership
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), expected)
    assert info["threshold"] == original["threshold"]
