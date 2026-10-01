"""Keep wall growth from inventing new lung/non-lung airway crossings."""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)


class LungBoundaryGuard:
    """A one-voxel inside-lung barrier with fixed airway-entry allowances.

    Entries are actual 26-neighbor contacts between the cleaned reference
    airway on opposite sides of the lung mask. Expand their allowance by
    ``entry_margin`` voxels per axis to accommodate the airway walls. All
    other inside-lung boundary voxels remain forbidden to subsequent growth.

    Construct once BEFORE wall dilation/closing/filling and reuse unchanged.
    This prevents new boundary crossings away from existing entries; it does
    not distinguish main bronchi from leaks already present in the reference.
    A reference without crossings creates no allowed entries (fail closed).
    """

    def __init__(self, reference_airways, lungs, *, entry_margin=3):
        if isinstance(entry_margin, bool) or not isinstance(entry_margin, (int, np.integer)) or entry_margin < 0:
            raise ValueError("entry_margin must be a nonnegative integer")
        self.reference = reference_airways
        self._check_grid(lungs)
        airway = sitk.GetArrayViewFromImage(reference_airways) > 0
        lung = sitk.GetArrayViewFromImage(lungs) > 0
        boundary = lung & ndi.binary_dilation(~lung, structure=_CONNECTIVITY)
        outside_airway = airway & ~lung
        entries = (airway & boundary) & ndi.binary_dilation(
            outside_airway, structure=_CONNECTIVITY
        )
        if entries.any() and entry_margin:
            # Separable box filtering is linear in volume size, unlike a dense
            # large-footprint dilation. This margin is explicitly in voxels.
            allowed = ndi.maximum_filter(
                entries, size=2 * int(entry_margin) + 1, mode="constant", cval=0
            )
        else:
            allowed = entries
        self.blocked = boundary & ~allowed
        self.metadata = dict(
            entry_margin_voxels=int(entry_margin),
            reference_entry_voxels=int(entries.sum()),
            allowed_boundary_voxels=int(np.count_nonzero(boundary & allowed)),
            blocked_boundary_voxels=int(self.blocked.sum()),
            connectivity=26,
            status="no_reference_crossings" if not entries.any() else "active",
        )

    def _check_grid(self, image):
        reference = self.reference
        for obj in (reference, image):
            if not isinstance(obj, sitk.Image):
                raise TypeError("Expected SimpleITK images")
            if obj.GetDimension() != 3 or obj.GetNumberOfComponentsPerPixel() != 1:
                raise ValueError("Expected scalar 3D images")
        if image.GetSize() != reference.GetSize():
            raise ValueError("Image size differs from boundary guard reference")
        for name in ("Spacing", "Origin", "Direction"):
            if not np.allclose(getattr(image, "Get" + name)(), getattr(reference, "Get" + name)(), atol=1e-6, rtol=0):
                raise ValueError("Image geometry differs from boundary guard reference")

    def apply(self, image):
        """Remove forbidden boundary voxels; preserve all other labels/values."""
        self._check_grid(image)
        values = sitk.GetArrayFromImage(image)
        values[self.blocked] = 0
        result = sitk.GetImageFromArray(values)
        result.CopyInformation(image)
        return result
