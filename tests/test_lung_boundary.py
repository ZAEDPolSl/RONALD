"""Regression tests for wall growth across the lung/non-lung interface."""
import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage as ndi
from ronald.segmentation.lung_boundary import LungBoundaryGuard
from ronald.segmentation.airways_segmentation import constrained_airway_dilation, per_slice_hole_removal


def image(a):
    return sitk.GetImageFromArray(np.asarray(a,dtype=np.uint8))


def arrays():
    lung=np.zeros((19,35,43),np.uint8);lung[:,:,22:]=1
    airway=np.zeros_like(lung)
    # Genuine crossing at y=7; disconnected potential false crossing at y=26.
    airway[5:14,5:10,12:29]=1
    airway[5:14,24:29,12:20]=1
    airway[5:14,24:29,23:28]=1
    return lung,airway


def count(a):
    return ndi.label(a,structure=np.ones((3,3,3)))[1]


def test_dilation_cannot_create_distant_boundary_crossing_but_keeps_existing_entry():
    lung,airway=arrays();guard=LungBoundaryGuard(image(airway),image(lung),entry_margin=3)
    grown=constrained_airway_dilation(image(airway),image(lung),(3,3,3))
    before=sitk.GetArrayFromImage(grown)>0
    after=sitk.GetArrayFromImage(guard.apply(grown))>0
    assert count(before)==2
    assert count(after)==3
    assert after[9,7,21] and after[9,7,22]  # genuine entry retained
    assert not after[9,26,22]              # invented entry blocked
    assert not np.any(after&~before)


def test_same_guard_after_closing_and_filling_cannot_reopen_boundary():
    lung,airway=arrays();guard=LungBoundaryGuard(image(airway),image(lung))
    grown=guard.apply(constrained_airway_dilation(image(airway),image(lung),(3,3,3)))
    closed=sitk.BinaryMorphologicalClosing(grown,(3,6,6))
    assert sitk.GetArrayFromImage(closed)[9,26,22]
    closed=guard.apply(closed)
    filled=guard.apply(per_slice_hole_removal(closed))
    result=sitk.GetArrayFromImage(filled)>0
    assert not result[guard.blocked].any()
    assert result[9,7,22]


def test_diagonal_crossings_are_also_blocked():
    lung=np.zeros((7,7,7),np.uint8);lung[:,:,4:]=1
    reference=np.zeros_like(lung);reference[3,3,2]=1
    guard=LungBoundaryGuard(image(reference),image(lung),entry_margin=0)
    growth=reference.copy();growth[3,3,3]=1;growth[4,4,4]=1
    assert count(growth)==1
    result=sitk.GetArrayFromImage(guard.apply(image(growth)))>0
    assert not result[4,4,4]
    assert guard.metadata['status']=='no_reference_crossings'


def test_geometry_labels_and_input_immutability():
    lung,airway=arrays();reference=image(airway);reference.SetSpacing((.7,.8,1.3));reference.SetOrigin((10,20,30))
    lim=image(lung);lim.CopyInformation(reference)
    guard=LungBoundaryGuard(reference,lim)
    original=sitk.GetArrayFromImage(reference)
    out=guard.apply(reference)
    assert out.GetSpacing()==reference.GetSpacing() and out.GetOrigin()==reference.GetOrigin()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(reference),original)
    wrong=sitk.Image(reference);wrong.SetOrigin((0,0,0))
    with pytest.raises(ValueError,match='geometry'):guard.apply(wrong)


def test_preexisting_crossing_is_not_claimed_to_be_anatomically_validated():
    lung,airway=arrays();airway[9,26,19:24]=1
    guard=LungBoundaryGuard(image(airway),image(lung),entry_margin=0)
    assert not guard.blocked[9,26,22]  # explicit limitation: input leak survives


@pytest.mark.parametrize('margin',[-1,True,1.5])
def test_invalid_margin(margin):
    with pytest.raises(ValueError):LungBoundaryGuard(image(np.zeros((3,3,3))),image(np.zeros((3,3,3))),entry_margin=margin)
