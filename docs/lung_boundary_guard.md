# Lung-boundary guard during wall construction

The airway leakage cleanup retains outside-lung components attached to the
trachea. That does not prevent lateral attachments. Separately clipped regional
dilations can meet at their shared boundary, and later global closing and axial
hole filling can create new crossings.

`LungBoundaryGuard` freezes permitted entry regions from the cleaned airway
mask before wall growth. It detects 26-neighbor airway contacts across the lung
boundary, permits a three-voxel margin along each axis around these entries,
and forbids all other inside-lung boundary voxels. A fixed guard is applied to
the cleaned airway seed, its dilation, closed walls, and filled walls. The final
airway mask uses the cleaned seed rather than the original uncleaned mask.

Only existing crossing neighborhoods are permitted. This is not anatomical
identification of the two main bronchi: a leak already present in the reference
can remain allowed. No reference crossings means no permitted crossings.
The margin is in voxels, matching the existing airway dilation, not millimeters.

The barrier is encoded in `walls_filled`; subsequent Sato masking and final
reconstruction clipping use that same support. No new input argument is needed
for `smooth_tree`.

Regression tests cover new connections from dilation and closing, diagonal
contacts, preservation of existing entries, repeated application after filling,
geometry checks, and the explicit pre-existing-leak limitation.

Two-patient inspection outputs are under each patient's
`wall_lung_boundary_20260929_122517` folder. These use saved final airways as a
reference proxy and apply the guard to existing wall/Sato masks. They are not
full airway-segmentation or Sato reruns. Original files are unchanged.
