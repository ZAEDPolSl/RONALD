# BRONCO
This repository contains a Python 3.8 implementation of an airways segmentation algorithm for CT thorax images. 
The algorithm utilizes the Fast Marching method, guided by two velocity maps: one based on image gradient and 
the other on vesselness filtering. This combination enhances the accuracy and reliability of airway segmentation,
making it suitable for medical imaging and research purposes.
Features:

- Segments airways from thoracic CT scans.
- Uses Fast Marching algorithm for region growing.
- Two velocity maps (gradient-based and vesselness) to guide segmentation.
- Written in Python 3.8 for easy integration and flexibility.

**Example output:**
![Image](/data/readme/airways_subplots.png)

Feel free to contribute or use this algorithm for research and development. Algorithm works best for low-dose CT.

# Installation
Run command:

``pip install -r requirements.txt``

The `lungmask` package is there for the lungs segmentation task, please refer to the
[original github](https://github.com/JoHof/lungmask) repository for citation.
# Usage

Example usage can be found in `examples/01_segmentation.py`, note that depending on your IDE configuration supplied
in the example path strings may not be correct, adjust for personal usage.

Code be easily integrate in the production pipeline, the `run` function accepts both paths to the series and 
sitk.Image objects. For CT image reading we recommend the `ImageInstance` class.

```python
from bronco.run import run
from bronco.io_local import ImageInstance


if __name__ == "__main__":
    path_input = "data/input/patient1.nrrd"
    path_supl = "data/input/patient1_lungs.nrrd"
    path_output = "data/output.nrrd"
    
    ii_image = ImageInstance()
    ii_lungs = ImageInstance()
    sitk_image = ii_image.read(path_input)
    sitk_mask = ii_lungs.read(path_supl)

    run(sitk_image, path_output, sitk_mask)
```
