# MRI Vessel Reporting

This document describes the MRI lung-mask and vessel-reporting workflow.
Run commands from the repository root.

## Scope

Current biological scope:
- all vessels together
- no arterial vs venous split

Current processing scope:
- input is a JSON config with MRI image paths
- the pipeline generates the MRI lung mask automatically
- vessel segmentation runs in MRI mode
- outputs are written per study plus batch-level summary files

The MRI path is heuristic and was tuned on available example data. It should be
visually checked on new acquisition protocols before quantitative use.

## Input Assumption

For each case, you need one lung MRI image. The image may be a `.nii.gz`, `.nrrd`,
or a DICOM series directory readable by SimpleITK/CTools.

Important limitation:
- the MRI lung segmentation assumes the scan contains some surrounding body
- if the image is tightly cropped at the lungs, the body-versus-air model can fail
- outputs should be inspected, especially on new low-field MRI protocols

The pipeline saves the lung mask in the same image geometry as the input scan:
dimensions, voxel spacing, origin, and orientation/direction are copied from the
MRI image.

## Quick Start

### Python

```bash
python -m venv .venv-mri
source .venv-mri/bin/activate
pip install -r mri-requirements.txt
```

Copy and edit the example config:

```bash
cp mri_vessel_reporting_config.example.json my_mri_vessel_config.json
```

Set each study `image` to an MRI image file or DICOM series directory:

```json
{
  "output_dir": "/absolute/path/to/output",
  "caliber_thresholds_mm": {
    "small": 2.0,
    "large": 5.0
  },
  "studies": [
    {
      "name": "patient_001",
      "image": "/absolute/path/to/mri_image.nii.gz"
    }
  ]
}
```

Run:

```bash
python calculate_vesselness_stats.py --config my_mri_vessel_config.json
```

### Docker

Build locally:

```bash
docker build -f MRI-vesselness.dockerfile -t bronco-mri-vessels .
```

Run with a mounted data/config directory:

```bash
docker run --rm \
  -v /absolute/path/to/data:/data \
  bronco-mri-vessels \
  --config /data/config.json \
  --output-dir /data/output
```

Important:
- config paths must be valid inside the container, e.g. `/data/image.nii.gz`
- output should be inside the mounted directory if you want it on the host

## Config Format

Example config:
- [mri_vessel_reporting_config.example.json](mri_vessel_reporting_config.example.json)

Schema:

```json
{
  "output_dir": "data/mri_vessel_reports",
  "caliber_thresholds_mm": {
    "small": 2.0,
    "large": 5.0
  },
  "studies": [
    {
      "name": "case_001",
      "image": "/absolute/path/to/image.nii.gz"
    }
  ]
}
```

Notes:
- `studies` must be a non-empty list
- each study must contain `image`
- `name` is optional; if omitted, it is derived from the image filename
- `image` can point to a regular image file or a DICOM series directory
- if `image` points to DICOM, it should point to one directory containing one series
- all paths must be valid in the environment where the script runs

## Segmentation Workflow

Implemented in:
- [bronco/segmentation/lungs_segmentation.py](bronco/segmentation/lungs_segmentation.py)
- [bronco/segmentation/vessel_segmentation.py](bronco/segmentation/vessel_segmentation.py)

MRI lung segmentation:
- fits a 2-component GMM to separate bright body from dark air/background
- derives internal lung-cavity seeds from the filled body envelope
- fills hole-like lung gaps with 2D majority filling
- refines central/hilar regions with random walker constrained by mediastinum and hull priors
- saves an exploratory `air_roi` mask from the same body/cavity model

MRI vessel segmentation:
- runs Frangi vesselness in a small lung-context band
- thresholds vesselness with BIC-selected GMM foreground
- suppresses obvious airway/air-lumen responses using the generated `air_roi`
- keeps vessel components that extend beyond the surface band, reducing flat edge artifacts

The air ROI is only a rough helper for suppressing airway-like Frangi responses.
It is not a validated airway segmentation.

## Outputs

For each study, the script writes:
- `masks/lung_mask.nrrd`
- `masks/air_roi.nrrd`
- `masks/mediastinum_mask.nrrd`
- `masks/vessel_mask.nrrd`
- `centerlines/skeleton.nrrd`
- `centerlines/graph_nodes.csv`
- `centerlines/graph_node_points.csv`
- `centerlines/graph_edges.csv`
- `centerlines/graph_edge_points.csv`
- `metrics/vessel_metrics.json`
- `metrics/branch_metrics.csv`

At the batch level, the script writes:
- `reports.json`
- `study_metrics.csv`

## Reported Statistics

Per-study statistics include:
- vessel volume and lung volume
- total skeleton length
- thickness summaries
- tortuosity summaries
- curvature summaries
- branch, endpoint, junction, and component counts
- configurable small/medium/large caliber distribution
- small-vessel length summaries

The reporting bins are configured in the JSON config:

```json
"caliber_thresholds_mm": {
  "small": 2.0,
  "large": 5.0
}
```

This means:
- small: thickness `< 2.0 mm`
- medium: thickness `>= 2.0 mm` and `< 5.0 mm`
- large: thickness `>= 5.0 mm`

Branch numbering is central-to-peripheral. Branch `1` is closest to the
vessel-mask bounding-box center; larger branch ids are farther from that center.
This is not anatomical artery/vein labeling.

## Practical Recommendation

If you only need the final numbers for one or a few cases:
- use `metrics/vessel_metrics.json` for the full structured report
- use `study_metrics.csv` if you want a spreadsheet-friendly summary

If you want branch-level or centerline-level inspection:
- use `metrics/branch_metrics.csv`
- use the `centerlines/*.csv` files

## Citation

If you use this work, please cite:

> Mrukwa A, Polańska A, et al. Can Proper Vessel Segmentation Improve Early-Stage Lung Cancer Detection? Poster DATA-005, European Molecular Imaging Meeting (EMIM 2026), Ljubljana, Slovenia, 2026.

