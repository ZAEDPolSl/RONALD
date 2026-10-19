# Linux server paths and commands

Branch: `amrukwa/ronald-unet`.

The roots below were checked against the NAS mounted locally at `/Volumes/pma`
and the existing dataset configuration. Linux uses `/mnt/pmanas` for the same share.
This checks directory layout, not completeness or quality of every case.
`configs/training/pmanas.json` stores these paths relative to the mount.

| Cohort | CT root under `/mnt/pmanas/` | Mask root under `/mnt/pmanas/` |
|---|---|---|
| Moltest 1 | `Wojtek/MIRASLC/Dane surowe/Moltest 1/NRRD` | `Ania/phd-data/moltest-1/masks` |
| Moltest 2 | `Wojtek/MIRASLC/Dane surowe/Moltest 2` | `Ania/phd-data/moltest-2/masks` |
| DUKE | `Ania/phd-data/DUKE/raw/DLCS_subset*` | `Ania/phd-data/DUKE/masks` |
| PLCS4 | `OPPRP/PLCS` and `OPPRP/PLCSv2` | `Ania/phd-data/OPPRP/masks` |
| LUNA16 | `Ania/phd-data/luna-2016/raw/subset*` | `Ania/phd-data/luna-2016/masks` |

Moltest 2 also checks `Ania/phd-data/moltest_2_400/masks` when the primary
mask directory lacks a complete set. Its annotation table points to patient
folders under the same Moltest 2 raw root. Complete primary masks take precedence.

PLCS4 uses `Ania/phd-data/OPPRP/plcs4_prediagnosis_selected_scans.csv` to identify
scans and group repeat scans by patient. Its Windows `X:\OPPRP\...` paths are
translated to `/mnt/pmanas/OPPRP/...`. The builder does not infer PLCS4 membership
from every scan in the much larger OPPRP directory.

Per-scan masks: `airways_final.nrrd`, `vessels_final.nrrd`, `lungs.nrrd`, optional
`mediastinum.nrrd`, optional `region_grow.nrrd`. The latter may contain positive
integer nodule instance IDs; preprocessing unions these to one nodule mask.
Other `region_grow*` variants are not assumed equivalent without checking their
meaning. Missing nodule masks retain the reduced training weight. LUNA always
uses class-1 generation in these commands.

LUNA annotation files:

```text
/mnt/pmanas/Ania/phd-data/luna-2016/raw/annotations.csv
/mnt/pmanas/Ania/phd-data/luna-2016/raw/candidates_V2.csv
```

## Get the branch

In an existing checkout:

```bash
git fetch origin
git switch amrukwa/ronald-unet
git pull --ff-only
git submodule update --init --recursive
source .venv/bin/activate
python -m pip install -r requirements-training.txt
```

For a new checkout, clone `--branch amrukwa/ronald-unet` and install
`requirements-training.txt` as described in `README.md`.

## Build the inventory and prepare

Run from the repository root. The work directory below is a **new chosen output
location**, not an existing dataset directory. A local SSD may be preferable for
cache throughput; change `WORK` before starting if needed.

```bash
export WORK=/mnt/pmanas/Ania/bronco-pretrain/ronald-unet
export LUNA=/mnt/pmanas/Ania/phd-data/luna-2016/raw
mkdir -p "$WORK"

python -m ronald.training.inventory \
  --config configs/training/pmanas.json \
  --output "$WORK/inventory.csv"

python -m ronald.training.prepare \
  --inventory "$WORK/inventory.csv" \
  --output "$WORK/cache-v1" \
  --luna-annotations "$LUNA/annotations.csv" \
  --luna-candidates "$LUNA/candidates_V2.csv" \
  --spacing 0.8 0.8 0.8 --margin-mm 15 --hu -1000 500 --seed 42
```

The inventory builder reports missing files and scans with no discoverable DICOM series in
`inventory.audit.json`. It only includes scans with both final masks and lungs. Nested DICOM folders are
resolved by `ctools.ImageInstance` using its most-slices policy. The actual series
directory and UID are saved in the inventory and audit, then reused at loading.
Review this selection when several CT reconstructions are present.
It does not copy or modify NAS source images. Duplicate scan identifiers fail
rather than silently choosing one image. LUNA discovery is restricted to CT
subsets so its separate lung-mask MHDs cannot be mistaken for CT scans.

**Patient linkage:** Moltest groups normalized patient-folder names across
Moltest 1/2; PLCS4 uses the selection table's patient key; DUKE uses its scan IDs;
LUNA uses series IDs. These are dataset conventions, not independently verified
cross-cohort identities. Review them before preparation. For known aliases or
repeat LUNA series, supply `--patient-map /path/to/mapping.csv` containing
`case_id,patient_id`; use case IDs from the inventory audit and the same global
patient ID for all linked cases. Rebuild the inventory under a new filename.
Preparation additionally detects identical CT arrays crossing splits.

## Train / resume

```bash
python -m ronald.training.train \
  --manifest "$WORK/cache-v1/manifest.json" \
  --config configs/training/unet.json \
  --output "$WORK/run-01" --device cuda:0
```

After interruption, activate the environment and set `WORK` again, then:

```bash
python -m ronald.training.train \
  --manifest "$WORK/cache-v1/manifest.json" \
  --config "$WORK/run-01/config.json" \
  --output "$WORK/run-01" --device cuda:0 \
  --resume "$WORK/run-01/last.pt"
```

```bash
tensorboard --logdir "$WORK/run-01/tensorboard" --port 6006
```

Use a fresh inventory filename and cache directory for a new preparation; keep
both unchanged when resuming training. Inspect preparation audit and region-grown
nodules before a full training run. Inventory/audit files contain private source
paths and stay on the server; do not commit them to Git.
