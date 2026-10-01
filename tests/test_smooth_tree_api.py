"""The single accepted pipeline: internal Sato and connected-only restoration."""

import importlib
import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage as ndi

mod = importlib.import_module("ronald.modelling.smooth_tree")


def image(array):
    return sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))


def test_public_computes_sato_inside_thread_limit_and_reports_it(monkeypatch):
    walls = image(np.ones((5, 5, 5)))
    walls.SetSpacing((0.7, 0.8, 1.2))
    walls.SetOrigin((5, 6, 7))
    sato = sitk.Image(walls)
    calls = []
    before = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()

    class Observer:
        def record_sato(self, mask, metadata):
            calls.append(("observe", mask, metadata))

    observer = Observer()

    def compute(mask, **kwargs):
        assert mask is walls
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == 1
        calls.append(("compute",))
        return sato, {"threshold": 0.53}

    def model(w, s, **kw):
        assert w is walls and s is sato
        assert kw["diagnostics"] is observer
        calls.append(("model",))
        return sitk.GetArrayFromImage(s)

    monkeypatch.setattr(mod, "calculate_sato_mask", compute)
    monkeypatch.setattr(mod, "model_tree", model)
    result = mod.smooth_tree(walls, max_threads=1, diagnostics=observer)
    assert [c[0] for c in calls] == ["compute", "observe", "model"]
    assert result.GetSpacing() == walls.GetSpacing()
    assert result.GetOrigin() == walls.GetOrigin()
    assert result.GetDirection() == walls.GetDirection()
    assert result.GetPixelID() == sitk.sitkUInt16
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == before


@pytest.mark.parametrize("field", ["size", "spacing", "origin", "direction"])
def test_bad_sato_grid_rejected_before_graph(monkeypatch, field):
    walls = image(np.ones((5, 5, 5)))
    sato = sitk.Image(walls)
    if field == "size":
        sato = image(np.ones((4, 5, 5)))
    elif field == "spacing":
        sato.SetSpacing((1, 1, 2))
    elif field == "origin":
        sato.SetOrigin((1, 0, 0))
    else:
        sato.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, 1))
    monkeypatch.setattr(
        mod, "prepare_graph", lambda *a, **k: pytest.fail("must reject before graph")
    )
    with pytest.raises(ValueError, match=field):
        mod.model_tree(walls, sato)


@pytest.mark.parametrize("bad", ["outside", "label", "nan"])
def test_invalid_foreground_rejected_before_graph(monkeypatch, bad):
    w = np.zeros((5, 5, 5), np.uint8)
    w[1:4, 1:4, 1:4] = 1
    s = w.astype(float)
    if bad == "outside":
        s[0, 0, 0] = 1
    elif bad == "label":
        s[2, 2, 2] = 2
    else:
        s[2, 2, 2] = np.nan
    monkeypatch.setattr(
        mod, "prepare_graph", lambda *a, **k: pytest.fail("must reject before graph")
    )
    with pytest.raises(ValueError):
        mod.model_tree(image(w), sitk.GetImageFromArray(s))


def test_connected_restoration_rejects_disjoint_source_and_keeps_fitted_walls():
    s = np.zeros((10, 10, 10), np.uint8)
    s[2:5, 2, 2] = 1
    s[7:9, 7, 7] = 1
    model = np.zeros_like(s)
    model[2, 2, 2] = 1
    model[2, 2, 3] = 1
    before = model.copy()
    out = mod.restore_connected_sato(model, image(s))
    assert np.all(out[2:5, 2, 2])
    assert not out[7:9, 7, 7].any()
    assert out[2, 2, 3]
    np.testing.assert_array_equal(model, before)


def test_pipeline_restores_sato_connects_supported_gap_and_excludes_remote_component():
    z, y, x = np.indices((48, 36, 36))
    s = ((y - 12) ** 2 + (x - 12) ** 2 <= 4) & (
        ((z >= 5) & (z <= 20)) | ((z >= 25) & (z <= 40))
    )
    s |= ((y - 29) ** 2 + (x - 29) ** 2 <= 1) & (z >= 12) & (z <= 17)
    w = (((y - 12) ** 2 + (x - 12) ** 2 <= 9) & (z >= 5) & (z <= 40)) | s
    before = w.copy()
    out = mod.model_tree(image(w), image(s)) > 0
    assert ndi.label(out, structure=np.ones((3, 3, 3)))[1] == 1
    assert out[8:38, 12, 12].all()
    assert not out[:, 28:31, 28:31].any()
    assert not (out & ~w).any()
    assert np.all(out[s & (y < 25)])
    np.testing.assert_array_equal(w, before)


def test_empty_sato_returns_empty_without_skeletonization(monkeypatch):
    w = image(np.ones((5, 5, 5)))
    s = image(np.zeros((5, 5, 5)))
    monkeypatch.setattr(
        mod, "calculate_sato_mask", lambda *a, **k: (s, {"threshold": None})
    )
    monkeypatch.setattr(
        mod, "prepare_graph", lambda *a, **k: pytest.fail("no skeleton for empty Sato")
    )
    assert not sitk.GetArrayFromImage(mod.smooth_tree(w)).any()
