# RONALD pseudo-mask distillation with a 3D UNet

Three output classes: **0 background (including nodules), 1 airway with wall,
2 vessel**. This is a standalone PyTorch implementation; no UNETR or nnUNet
installation is needed. The starter on `origin/amrukwa/bronconet` was inspected
(`train.py`, `dataset.py`, `bronco/training/*`). Its fixed-cube resizing,
four-class outputs, and sequential UNet/SwinUNETR runs are replaced here.
Existing classical RONALD code is unchanged.

For verified `/mnt/pmanas` roots, automatic inventory generation, and server commands,
see [PMANAS.md](PMANAS.md).

## Environment and quick start

Run from the repository root in Python 3.10+ (3.11 recommended). Install the
PyTorch build appropriate for the destination CUDA driver, then:

```bash
git submodule update --init --recursive
python -m pip install -r requirements-training.txt
python -m pytest tests/test_training_pipeline.py -q

python -m ronald.training.prepare \
  --inventory /data/inventory.csv \
  --output /scratch/ronald-cache \
  --luna-annotations /data/LUNA16/annotations.csv \
  --luna-candidates /data/LUNA16/candidates_V2.csv

python -m ronald.training.train \
  --manifest /scratch/ronald-cache/manifest.json \
  --config configs/training/unet.json \
  --output /runs/ronald-unet --device cuda:0

tensorboard --logdir /runs/ronald-unet/tensorboard
```

The requirements specify compatibility ranges, not an exact environment lock.
Every run records installed versions in `environment.json`. Archive the destination
machine's `pip freeze` with the experiment. CPU is supported for smoke tests.
CUDA performance and real-data convergence must be tested on the training machine.

## Inventory: explicit paths, no guessed patient identities

Copy `configs/training/inventory.example.csv` and replace its illustrative rows.
Use all available complete cases from Moltest 1, DUKE, Moltest 2, PLCS4 and LUNA16.
A row requires both final pseudo-masks; incomplete cases should not be included.
No fixed server paths or dataset discovery conventions are embedded in the code.

| Column | Meaning |
|---|---|
| `case_id` | Unique filename-safe identifier for a scan |
| `patient_id` | Global participant ID, shared by **all** scans of that person, including across datasets |
| `dataset` | Cohort name; used for split stratification and reporting |
| `image` | HU CT: NIfTI, NRRD, MHD, or a single-series DICOM directory |
| `airways` | Binary `airways_final`, already including the airway wall |
| `vessels` | Binary `vessels_final` |
| `lungs` | Binary mask containing both lungs; required |
| `mediastinum` | Optional binary mediastinum mask to extend the crop |
| `nodules` | Optional existing region-growing/nodule mask; takes precedence over LUNA generation |
| `luna_seriesuid` | Match to `seriesuid` in LUNA `annotations.csv`; only used with `--luna-annotations` |
| `series_uid` | Selected DICOM series UID; the inventory builder records this and its actual directory |
| `split` | Optional `training`, `validation`, or `test`; fill every row or leave every row empty |

DICOM discovery and loading use `ctools.ImageInstance`, including its nested-folder
search and most-slices selection policy. The inventory records the selected series
directory and UID, so preprocessing reads that same series. Review the selected
series against the pseudo-masks when multiple reconstructions exist.

Paths may be absolute or relative to the inventory file. Masks must be scalar,
binary 0/1 or 0/255 with valid physical coordinates. Nodule masks additionally
accept nonnegative integer instance labels; all positive IDs are unioned. Convert multi-label lung/lobe
maps to a binary lung mask first. HU-masked images are not binary masks. The
pipeline does not infer or thicken airway walls: they must be in `airways_final`.

Default splitting is 70/15/15 by patient, stratified by cohort membership and usable
nodule-mask availability where cohort size permits. Tiny strata stay in training;
use explicit splits for small cohorts or site-held-out experiments. Repeated
patient IDs and identical native voxel hashes cannot cross splits. This does not
replace checking a patient linkage table: different reconstructions or renamed
patient IDs cannot reliably be linked from image content. Reuse a fixed split for
all ablations. Test cases never enter training or checkpoint selection.

## Preprocessing and nodule masks

- Reorient to LPS and resample in physical space (default **0.8 mm XYZ**): linear
  interpolation for HU CT, nearest neighbor for labels. Inspect the audit before
  training. Set finer spacing if native small-vessel detail warrants it; finer
  interpolation cannot recover detail absent from the source scans.
- Crop to the bounding box of both lungs plus optional mediastinum, with 15 mm
  margin. The box contains the gap between the lungs. CT is **not multiplied by a
  lung mask**, so central tissue and nodules remain. To include mediastinal tissue
  outside the lung bounding box, supply its mask or increase `--margin-mm`.
- Clip HU to [-1000, 500] and scale to [0, 1]. Do not resize to a fixed whole-volume
  cube. Cropped arrays are cached on disk and memory-mapped for patch extraction.
- Airway/vessel overlap is ignored (`255`) rather than arbitrarily assigned.
  Nodule masks override either foreground class **and** overlap to background.
  Padding is also ignored. `audit.json` records conflicts, omitted foreground,
  nodule coverage and region-growing failures; exclusion of nodule voxels by the
  crop is an error. Supplied masks must not be empty for airways/vessels/lungs.
- Missing, empty, or entirely failed nodule masks are treated as unavailable
  supervision, never as proof that a scan has no nodules.

[LUNA16 annotations](https://luna16.grand-challenge.org/Data/) describe reference
nodules, not confirmed malignancy. With `--luna-candidates`, only `class=1` candidates select reference nodules for
region growing. Candidates are matched within the annotated nodule radius and
deduplicated; growth uses the reference center and diameter. `class=0` never
initiates growth. Missing or ambiguous matches fail explicitly. Selection counts
are written to `luna_selection.json`. Without the candidate file, all rows in the
standard `annotations.csv` are reference nodules; if a custom annotation CSV has
a `class` column, only class 1 is loaded. Candidate points alone are not contours
and lack diameters, so the reference annotation file is required. Malignant-only
selection requires additional externally linked labels and should not make benign
nodules become airway/vessel targets.

LUNA region growing runs on the native HU CT, using physical XYZ centers and
annotated diameters. It searches a small seed neighborhood, constrains connected
growth by HU range and seed-intensity tolerance, and bounds growth by a sphere
scaled from the annotated radius. Components touching that bound or the CT/ROI
edge are rejected. Default HU [-700, 400], tolerance 250 HU, radius factor 1.25 are
**initial heuristic settings**, configurable with `--grow-*`. Ground-glass,
pleural, calcified, or vessel-attached nodules can fail or be incompletely captured.
The result is not a validated nodule contour. Review the saved native
`nodules_region_grow.nii.gz` overlays and `audit.json`; provide corrected masks
through `nodules` when needed. No spherical fallback is silently labeled background.

Preparation starts in a fresh output directory and fails on missing files or invalid
geometry/identities. After interruption, repeat the exact preparation command with
`--resume`. Completed scans are reused; only an interrupted scan and remaining scans
are processed. `prepare_state.json` fingerprints configuration and inventory/LUNA
CSV content. Each scan has `complete.json` with case/audit metadata, source sizes
and modification times, and output sizes/times. Source metadata checks include
DICOM files and MHD payloads; they do not reread complete CT pixels. Do not edit
source or cache files while preparing. This is a change-detection check, not a
cryptographic integrity check of every voxel file.

Scan output is written into `.partial/<case_id>` and atomically renamed into place
only when all outputs and the completion record are written. A cache-directory lock
prevents concurrent preparation writers. Audit/manifest JSON writes are atomic.
Running `--resume` on an already finished cache validates and reuses it without
rewriting its manifest, preserving training checkpoint compatibility.

For a partial cache created **before preparation resume support**, use
`--resume --adopt-legacy-cache` once, with the original inventory and every original
preprocessing/region-growing setting. This option asserts that these are unchanged:
the old format did not record enough provenance to verify that assertion. It checks
old arrays and audit entries, reads each adopted native CT once for its hash, and
reuses the prepared arrays. An unfinished old scan is moved aside under
`.legacy-incomplete-*` and rebuilt. Later resumes only need `--resume`. Missing or
unreadable legacy `audit.json` prevents adoption. If original settings are uncertain,
start a fresh cache instead. The manifest and audit retain paths/IDs, so store them with
the same access controls as the research data.

## Training objective and sampling

For scan/patch i:

```text
L_i = w_i × [ CE + Tversky + 0.2 × soft-clDice + 2.0 × nodule_background_CE ]
w_i = 1.0 with a usable nodule mask, otherwise 0.5
```

The batch loss is `mean(L_i)`, not divided by the sum of weights. Consequently an
entire batch of unannotated scans still has less influence. Availability is a
scan-level flag, so a nodule-supervised scan keeps weight 1 even if one patch misses
its nodules. A scan with some accepted and some rejected LUNA seeds has partial,
not complete, nodule supervision; inspect rejection counts in the audit.

- CE class weights: background 0.2, airway 1.0, vessel 1.5.
- Tversky: equal mean over the two foreground classes, FP coefficient 0.3,
  FN coefficient 0.7 to emphasize thin-structure recall.
- Soft-clDice: separate airway/vessel channels; differentiable morphological
  skeletonization with 10 iterations. It encourages centerline overlap; it cannot
  repair wrong/disconnected pseudo-labels or guarantee anatomical correctness.
  See [Shit et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html).
- Nodule term: mean `-log p(background)` **over nodule voxels only**, zero when none
  occur in the patch. This avoids dilution by the much larger background. No
  nodule masks or overrides are used to modify model predictions at validation or
  inference time.
- Sampling probabilities: 15% uniform, 30% airway, 35% vessel, 20% nodule. Missing
  sampling categories fall back to uniform. Each scan contributes eight patches
  per epoch, so larger cohorts still contribute more. No outcome-based oversampling.
- Random flips in all three axes and mild intensity scale/shift. Patch sampling is
  deterministic from seed, epoch and patch index, independently of worker order.

Defaults: plain five-level UNet, GroupNorm, channels 24/48/96/192/320,
96³ patches, batch 2, AdamW, cosine learning rate, CUDA AMP, gradient clipping.
Down/up-sampling uses strided/transposed convolutions. Morphology uses separable
voxel extrema instead of CUDA max-pool backward so strict deterministic mode
does not encounter that unsupported operation. A CUDA-only regression test is
included and skipped automatically on CPU-only hosts.
These are starting settings, not tuned results. If memory is tight, reduce batch
size to 1, patch dimensions to 64³ (multiples of 16), channels, or skeleton
iterations. This implementation is single-device; no distributed training.

An ablation should compare CE+Tversky, adding clDice, adding the nodule penalty,
and changing unannotated weight. Freeze preprocessing/splits and compare nodule
foreground leakage alongside airway/vessel segmentation metrics. Pseudo-label
validation measures teacher agreement; manually checked structures and nodules
are needed to establish true segmentation quality.

## Logs, validation, checkpointing

`training.log` records progress/errors; `metrics.jsonl` records epoch losses,
learning rate, duration, and validation. TensorBoard contains loss components,
gradient norms, per-class/per-cohort metrics and three-plane CT overlays.
`validation/` contains per-case CSVs, summary JSONs and PNG overlays. Orange is
airway, cyan vessel, magenta nodule, gray ignored overlap. Cases with nodules use
slices through an actual nodule voxel. Visualization defaults: first three
validation cases at epoch 1 and every 10 epochs; reorder the manifest before the
first run if a particular QA subset is desired.

Validation uses full cropped volumes and Gaussian-blended sliding windows; only
patches occupy GPU memory. Full probability accumulation is on CPU and can need
several GB of RAM per volume. Voxel NPV, PPV, sensitivity and specificity are
reported before Dice, plus the fraction of nodule voxels incorrectly predicted as
foreground. These are **voxel segmentation metrics**, not patient-level diagnostic
NPV/PPV. Undefined values are null with contributing-case counts. Macro means
weight cases equally; TP/FP/FN/TN per case allow additional pooling.

`best.pt` is selected by mean airway/vessel Dice (excluding background); voxel NPV
alone would favor trivial background predictions. Early stopping counts 20
validation checks without improvement. Validation occurs at epoch 1, every five
epochs, and the final epoch. `last.pt` is saved atomically after every completed
epoch, plus periodic checkpoints. Checkpoints include model, optimizer, scheduler,
AMP scaler, RNG state, configuration, preprocessing, and manifest SHA256.

```bash
python -m ronald.training.train \
  --manifest /scratch/ronald-cache/manifest.json \
  --config configs/training/unet.json \
  --output /runs/ronald-unet --device cuda:0 \
  --resume /runs/ronald-unet/last.pt

# Run the held-out test only after model/configuration selection.
python -m ronald.training.train \
  --manifest /scratch/ronald-cache/manifest.json \
  --config configs/training/unet.json \
  --output /runs/ronald-unet-test --device cuda:0 \
  --evaluate /runs/ronald-unet/best.pt --split test

python -m ronald.training.predict \
  --checkpoint /runs/ronald-unet/best.pt \
  --image /data/new_ct.nii.gz --lungs /data/new_lungs.nii.gz \
  --mediastinum /data/new_mediastinum.nii.gz \
  --output /results/new_segmentation.nii.gz --device cuda:0
```

Resume starts at the next completed epoch with unchanged config/manifest; an
interrupted epoch is rerun. Use the same software/hardware for exact numerical
reproduction. Do not change cached files after training starts. Only load trusted
checkpoints (RNG restoration requires Python object deserialization). Predictions
are restored to the original CT size, spacing, direction, and origin, with
background outside the crop. Lung/mediastinum masks are still needed at inference;
end-to-end speed also depends on how these are obtained.

## Verification

`tests/test_training_pipeline.py` covers nodule gradients and override semantics,
missing annotation weights, ignored padding, patient/duplicate split leakage,
physical geometry, bounded region growing, UNet outputs, sliding-window blending,
undefined metrics, and a synthetic CLI integration run. The integration test
prepares images, trains, resumes, checks identical weights, checks TensorBoard
scalars/images and disk outputs, evaluates the test set, and checks native-grid
prediction export. No private CTs or real training are required for these tests.
