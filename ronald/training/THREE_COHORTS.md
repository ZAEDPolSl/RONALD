# DUKE + LUNA16 + Moltest 1

Run from the repository in the training environment. Reuse the completed inventory;
no DICOM discovery or raw CT copying is needed. Source inventory is unchanged.
Keep generated inventories/audits on the server, outside Git.

Default: 70/15/15 training/validation/test, seed 42. Group globally by supplied
patient_id and stratify by dataset membership. Fractions are rounded per identity
stratum; scan ratios can differ. Existing complete explicit splits are preserved
and checked. Partial splits and overlapping identities fail. Optional
--patient-map CSV (case_id,patient_id) provides verified global identities.

DUKE/LUNA default IDs are scan/series-based: reconcile repeat patients and
cross-cohort aliases before claiming patient independence. The audit records this
limitation. The selector validates required fields and duplicate case/image paths;
preparation checks files and native voxel hashes across splits. Inspect prepared
masks/geometry before training. LUNA class=1 means a nodule, not malignancy.

## Select and freeze

```bash
export WORK=/mnt/pmanas/Ania/bronco-pretrain/ronald-unet
export THREE="$WORK/duke-luna-moltest1-v1"
export LUNA=/mnt/pmanas/Ania/phd-data/luna-2016/raw

python -m ronald.training.cohort \
  --inventory "$WORK/inventory-ctools.csv" \
  --output "$THREE/cohort" --seed 42 \
  --validation-fraction 0.15 --test-fraction 0.15 --resume
```

Outputs: inventory.csv, training.csv, validation.csv, test.csv, audit.json.
Audit includes scan/identity counts per split, dataset counts, fingerprints and
limitations. Review before preparation. --resume verifies unchanged inputs and
outputs; use a new versioned directory if these change. The last reported total
was 3,623; actual counts come from the server inventory.

## Prepare / resume preparation

```bash
python -m ronald.training.prepare \
  --inventory "$THREE/cohort/inventory.csv" \
  --output "$THREE/cache" \
  --luna-annotations "$LUNA/annotations.csv" \
  --luna-candidates "$LUNA/candidates_V2.csv" \
  --spacing 0.8 0.8 0.8 --margin-mm 15 --hu -1000 500 \
  --seed 42 --resume
```

Preparation respects frozen splits. Repeat after interruptions. Use this new cache,
not the earlier five-cohort cache.

## Train

```bash
python -m ronald.training.train \
  --manifest "$THREE/cache/manifest.json" \
  --config configs/training/unet.json \
  --output "$THREE/run-01" --device cuda:0
```

## Resume training

```bash
python -m ronald.training.train \
  --manifest "$THREE/cache/manifest.json" \
  --config "$THREE/run-01/config.json" \
  --output "$THREE/run-01" --device cuda:0 \
  --resume "$THREE/run-01/last.pt"
```

## TensorBoard

```bash
tensorboard --logdir "$THREE/run-01/tensorboard" --port 6006
```

Existing disk logs, checkpoints and visualizations remain enabled. Select models
using validation only; keep test protected until analysis/model choices are fixed.
Moltest 2 and PLCS4 are excluded from all three splits.
