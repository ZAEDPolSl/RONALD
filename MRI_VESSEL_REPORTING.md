# MRI Vessel Reporting

This document describes the currently implemented MRI vessel reporting workflow.

Run all commands below from the repository root directory.

## Scope

Current biological scope:
- all vessels together
- no arterial vs venous split

Current processing scope:
- input is a JSON config
- each study must provide:
  - path to MRI image
  - path to lung mask
- vessel segmentation runs in MRI mode
- outputs are written per study plus batch-level summary files
- the only supported MRI entry point is the batch config-driven runner

## What You Need Before You Start

For each case, you need:
- one lung MRI image
- one binary lung mask for that MRI

Important:
- the script does not create the lung mask for you
- the lung mask must already exist
- the MRI image can be a regular image file such as `.nii.gz`, or a DICOM series directory
- the lung mask should be a binary image file in the same physical space as the MRI
- the lung mask should cover both lungs

If you are not comfortable with Python environments, the Docker option below is usually the easiest way to run the pipeline.

## Quick Start

### Option 1. Python

1. Create and activate an environment:

```bash
python -m venv .venv-mri
source .venv-mri/bin/activate
pip install -r mri-requirements.txt
```

2. Copy the example config and edit it:

```bash
cp mri_vessel_reporting_config.example.json my_mri_vessel_config.json
```

Edit `my_mri_vessel_config.json` so that it contains:
- your MRI image path
- your lung mask path
- your desired output directory

If your image is stored as DICOM, set `image` to the directory containing that DICOM series.

3. Run the pipeline:

```bash
python calculate_vesselness_stats.py --config my_mri_vessel_config.json
```

4. Open the output folder listed in the config.

Main output files are:
- per-case `metrics/vessel_metrics.json`
- per-case `metrics/branch_metrics.csv`
- batch-level `study_metrics.csv`

### Option 2. Docker

1. Build the Docker image:

```bash
docker build -f MRI-vesselness.dockerfile -t bronco-mri-vessels .
```

2. Copy the example config and edit it so that the paths are valid inside the container:

```bash
cp mri_vessel_reporting_config.example.json my_mri_vessel_config.docker.json
```

Example:
- if you mount `/absolute/path/to/data` to `/data`
- then the config should use paths like `/data/case1/image.nii.gz`
- the output directory in the config should also be inside `/data` if you want the results on your host machine

3. Run the container:

```bash
docker run --rm \
  -v /absolute/path/to/data:/data \
  bronco-mri-vessels \
  --config /data/my_mri_vessel_config.docker.json
```

Optional:

```bash
docker run --rm \
  -v /absolute/path/to/data:/data \
  bronco-mri-vessels \
  --config /data/my_mri_vessel_config.docker.json \
  --output-dir /data/output
```

4. Open the output folder on your host machine.

## Requirements

Minimal Python dependencies for the MRI vessel reporting workflow are listed in:
- [mri-requirements.txt](mri-requirements.txt)
- [mri-requirements.lock.txt](mri-requirements.lock.txt)

Install with:

```bash
python -m venv .venv-mri
source .venv-mri/bin/activate
pip install -r mri-requirements.txt
```

For most users, `mri-requirements.txt` is the recommended install.

For an exact pinned environment matching the frozen MRI setup used in this worktree:

```bash
python -m venv .venv-mri
source .venv-mri/bin/activate
pip install -r mri-requirements.lock.txt
```

The MRI path uses the local `ctools` package for robust medical-image I/O in the batch runner.
This means the commands should be run from this repository, not from a copied standalone script.

## Entry Point

Main script:
- [calculate_vesselness_stats.py](calculate_vesselness_stats.py)

Run:

```bash
python calculate_vesselness_stats.py --config path/to/config.json
```

Optional:

```bash
python calculate_vesselness_stats.py --config path/to/config.json --output-dir path/to/output
```

If `--output-dir` is not provided, the script uses:
1. `output_dir` from the config, if present
2. otherwise `<config parent>/mri_vessel_reports`

All MRI runs should go through the config-driven batch runner, even for a single case.
Do not use the CT example scripts in `examples/` for this MRI workflow.

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
      "image": "/absolute/path/to/image.nii.gz",
      "lung_mask": "/absolute/path/to/lung_mask.nii.gz"
    }
  ]
}
```

Notes:
- `studies` must be a non-empty list
- each study must contain `image` and `lung_mask`
- `name` is optional; if omitted, it is derived from the image filename
- `image` can point to a regular image file or a DICOM series directory
- if `image` points to DICOM, it should point to one directory containing one study/series to be read
- `lung_mask` should point to a binary mask image file
- all paths inside the config must be valid in the environment where the script runs

Minimal MRI config example:

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
      "image": "/absolute/path/to/mri_image.nii.gz",
      "lung_mask": "/absolute/path/to/lung_mask.nii.gz"
    }
  ]
}
```

## Docker

Container file:
- [MRI-vesselness.dockerfile](MRI-vesselness.dockerfile)

Build:

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
- the JSON config paths must point to paths inside the container, for example `/data/image.nii.gz`
- mount the directory containing the MRI images, lung masks, config, and desired output location
- the Docker image was successfully tested on the example `b_state0` config in this worktree
- if you want outputs saved on your computer, make sure the output directory is inside the mounted `/data` tree

## Caliber Thresholds

The reporting bins are configurable in the JSON config.

For example:

```json
"caliber_thresholds_mm": {
  "small": 2.0,
  "large": 5.0
}
```

means:
- small: thickness `< 2.0 mm`
- medium: thickness `>= 2.0 mm` and `< 5.0 mm`
- large: thickness `>= 5.0 mm`

These are operational reporting bins for MRI, that can be adjusted.

## Segmentation Workflow

Implemented in:
- [bronco/segmentation/vessel_segmentation.py](bronco/segmentation/vessel_segmentation.py)

The MRI reporting script currently runs with:
- `mode="mri"`
- `check_mediastinum_connectivity=True`

## Reported Statistics

Metric helpers live in:
- [bronco/vessel_metrics.py](bronco/vessel_metrics.py)

Branch-wise measurements are computed on the true traced skeleton graph, not a display-adjusted graph with centroid-shifted node endpoints.

Per-study reported statistics currently include:

### 1. Volume
- vessel voxel count
- vessel volume in `mm^3`
- vessel volume in `mL`
- lung volume in `mm^3`
- lung volume in `mL`

### 2. Length
- total skeleton length in `mm`
- total skeleton length in `cm`

### 3. Thickness
- mean
- median
- std
- min
- max
- p10 / p25 / p75 / p90

### 4. Tortuosity
- mean
- median
- max

Definition:
- branch path length / straight endpoint-to-endpoint distance

### 5. Curvature
- mean
- median
- max

Definition:
- local bending estimated from discrete 3D centerline points

### 6. Branching Summary
- branch count
- endpoint count
- junction count
- connected components in vessel mask
- connected components in skeleton graph

### 7. Caliber Distribution
- configured thickness thresholds
- fraction of skeleton points in small / medium / large bins
- fraction of vessel voxels in small / medium / large bins

### 8. Small-Vessel Summary
- small-vessel skeleton length
- small-vessel skeleton length fraction
- large-vessel skeleton length
- large-vessel skeleton length fraction
- small-vessel skeleton length per mL of lung

## Outputs

For each study, the script writes:
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

`vessel_metrics.json` includes:
- input paths
- output paths
- computed statistics

`branch_metrics.csv` contains per-branch measurements such as:
- branch id
- node ids
- point count
- path length
- straight length
- tortuosity
- curvature
- thickness summary

The graph CSV files describe the true traced skeleton graph:
- `graph_nodes.csv`: one row per node with degree and node voxel count
- `graph_node_points.csv`: node voxel coordinates
- `graph_edges.csv`: one row per edge
- `graph_edge_points.csv`: traced edge voxel coordinates

`study_metrics.csv` is a flattened one-row-per-study summary for easier spreadsheet analysis.

## Practical Recommendation

If you only need the final numbers for one or a few cases:
- use `metrics/vessel_metrics.json` for the full structured report
- use `study_metrics.csv` if you want a spreadsheet-friendly summary

If you want branch-level or centerline-level inspection:
- use `metrics/branch_metrics.csv`
- use the `centerlines/*.csv` files
