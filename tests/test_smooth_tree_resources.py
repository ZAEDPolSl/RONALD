import importlib

import numpy as np
import pytest
import SimpleITK as sitk


smooth_tree_module = importlib.import_module("ronald.modelling.smooth_tree")


def test_restore_connected_sato_only_adds_overlapping_components():
    airways = np.zeros((8, 8, 8), dtype=np.uint8)
    airways[1:3, 1:3, 1:3] = 1
    airways[5:7, 5:7, 5:7] = 1
    smooth = np.zeros_like(airways, dtype=bool)
    smooth[1, 1, 1] = True

    result = smooth_tree_module.restore_connected_sato(
        smooth, sitk.GetImageFromArray(airways)
    )

    expected = np.zeros_like(airways, dtype=bool)
    expected[1:3, 1:3, 1:3] = True
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, expected)


def test_native_thread_limit_is_restored():
    before = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()

    with smooth_tree_module._limited_native_threads(1):
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == 1

    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == before


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5])
def test_native_thread_limit_rejects_invalid_values(invalid_limit):
    with pytest.raises((TypeError, ValueError)):
        with smooth_tree_module._limited_native_threads(invalid_limit):
            pass


def test_smooth_tree_imports_boolean_working_mask(monkeypatch):
    source_array = np.ones((4, 4, 4), dtype=np.uint8)
    source = sitk.GetImageFromArray(source_array)
    monkeypatch.setattr(
        smooth_tree_module,
        "model_tree",
        lambda *args, **kwargs: source_array.astype(np.uint8),
    )

    monkeypatch.setattr(
        smooth_tree_module, "calculate_sato_mask", lambda *a, **k: (source, {})
    )
    result = smooth_tree_module.smooth_tree(source, max_threads=1)

    assert result.GetPixelID() == sitk.sitkUInt16
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), source_array)
