# RONALD
This repository contains a Python 3.8 implementation of an airways and vessels segmentation algorithm for CT thorax images. 
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

# Installation

## Clone with Submodules
This project uses the [CTools](https://github.com/ZAEDPolSl/CTools) library as a git submodule. When cloning, use:

```bash
git clone --recurse-submodules https://github.com/ZAEDPolSl/RONALD.git
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

The `lungmask` package is there for the lungs segmentation task, please refer to the
[original github](https://github.com/JoHof/lungmask) repository for citation.

## Running Scripts

With uv:
```bash
uv run python examples/01_segmentation.py
```

With pip (in activated virtual environment):
```bash
python examples/01_segmentation.py
```

# Usage

Example usage can be found in `examples/whole_pipeline.py`, note that depending on your IDE configuration supplied
in the example path strings may not be correct, adjust for personal usage.
