# BRONCO
This repository contains Python tools for thoracic airway and vessel analysis.
The original workflow targets CT airway segmentation. The repository also
includes an MRI vessel-reporting workflow that generates an MRI lung mask,
segments vessel-like structures, and reports vessel metrics.

## CT Airway Segmentation

The CT airway algorithm segments airways from CT thorax images.
The algorithm utilizes the Fast Marching method, guided by two velocity maps: one based on image gradient and 
the other on vesselness filtering. This combination enhances the accuracy and reliability of airway segmentation,
making it suitable for medical imaging and research purposes.

Features:
- Segments airways from thoracic CT scans.
- Uses Fast Marching algorithm for region growing.
- Two velocity maps (gradient-based and vesselness) to guide segmentation.
- Written in Python 3.10 for easy integration and flexibility.

**Example output:**
![Image](/data/readme/airways_subplots.png)

Feel free to contribute or use this algorithm for research and development. Algorithm works best for low-dose CT.

## MRI Vessel Reporting

The MRI workflow is config-driven and requires only an MRI image per study. It
automatically creates and saves:

- `masks/lung_mask.nrrd`
- `masks/air_roi.nrrd`
- `masks/mediastinum_mask.nrrd`
- `masks/vessel_mask.nrrd`
- centerline files and vessel metrics

The MRI lung-mask algorithm is heuristic. It assumes the scan contains at least
some surrounding body, not only a tight crop around the lungs. New acquisition
protocols should be visually checked before quantitative use.

Run with Python:

```bash
python calculate_vesselness_stats.py --config mri_vessel_reporting_config.example.json
```

Run with Docker:

```bash
docker build -f MRI-vesselness.dockerfile -t bronco-mri-vessels .
docker run --rm \
  -v /absolute/path/to/data:/data \
  bronco-mri-vessels \
  --config /data/config.json \
  --output-dir /data/output
```

See [MRI_VESSEL_REPORTING.md](MRI_VESSEL_REPORTING.md) for the full MRI workflow.

# Installation

## Clone with Submodules
This project uses the [CTools](https://github.com/ZAEDPolSl/CTools) library as a git submodule. When cloning, use:

```bash
git clone --recurse-submodules https://github.com/ZAEDPolSl/BRONCO.git
```

Or if you've already cloned the repository:

```bash
git submodule update --init --recursive
```

## Install Dependencies

### Option 1: Using uv (recommended)
[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first, then:

```bash
uv sync
```

This will automatically install all dependencies including the CTools submodule.

### Option 2: Using pip
```bash
pip install -r requirements.txt
```

The `lungmask` package is used by the default CT lung-segmentation path; please
refer to the [original github](https://github.com/JoHof/lungmask) repository for citation.

## Running Scripts

With uv:
```bash
uv run python examples/01_segmentation.py
```

With pip (in activated virtual environment):
```bash
python examples/01_segmentation.py
```

MRI reporting example:

```bash
python examples/04_mri_vessel_reporting.py --image /absolute/path/to/mri_image.nii.gz
```

# Usage

Example usage can be found in `examples/whole_pipeline.py`, note that depending on your IDE configuration supplied
in the example path strings may not be correct, adjust for personal usage.

## Citation

If you use this work, please cite:

> Mrukwa A, Polańska A, et al. Can Proper Vessel Segmentation Improve Early-Stage Lung Cancer Detection? Poster DATA-005, European Molecular Imaging Meeting (EMIM 2026), Ljubljana, Slovenia, 2026.
