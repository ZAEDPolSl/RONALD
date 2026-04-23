# SWIN UNETR Training for Bronchial Tree Segmentation

This directory contains scripts for training a SWIN UNETR model for multi-class bronchial tree segmentation using MONAI.

## Overview

The pipeline segments CT scans into 4 classes:
- **Class 0**: Background
- **Class 1**: Airways (from airways.nrrd)
- **Class 2**: Airways final (from airways_final.nrrd, where airways is 0)
- **Class 3**: Vessels (from vessels_final.nrrd)

## Files

- `dataset.py`: Dataset preparation and data loading utilities
- `train_swin_unetr.py`: Training script for SWIN UNETR model
- `inference_swin_unetr.py`: Inference script for trained models
- `bronco_final_pipeline.py`: Script to create combined masks from individual segmentations

## Setup

### Requirements

```bash
pip install monai torch torchvision SimpleITK numpy scikit-learn pandas tqdm
```

### Recommended: Install with all optional dependencies

```bash
pip install 'monai[all]'
```

## Usage

### Step 1: Create Combined Masks (bronco_final.nrrd)

First, create the combined masks from individual segmentations:

```bash
python bronco_final_pipeline.py --dataset "Moltest 1" --start 0
```

This combines airways.nrrd, airways_final.nrrd, and vessels_final.nrrd into bronco_final.nrrd.

### Step 2: Prepare Dataset

Create a JSON file with train/val/test splits:

```bash
python dataset.py \
    --datasets "Moltest 1" "Luna 2016" \
    --output bronco_dataset.json \
    --val-split 0.2 \
    --test-split 0.1 \
    --seed 42 \
    --verify
```

**Arguments:**
- `--datasets`: List of datasets to include (space-separated)
- `--output`: Output JSON file path
- `--val-split`: Validation split fraction (default: 0.2)
- `--test-split`: Test split fraction (default: 0.1)
- `--seed`: Random seed for reproducibility
- `--mask-filename`: Name of mask file to use (default: bronco_final.nrrd)
- `--verify`: Verify data integrity after creation

**Available datasets:**
- "Moltest 1"
- "Moltest 2"
- "Luna 2016"
- "OPPRP"
- "NLST USA"
- "DUKE"
- "Moltest 2_400"

### Step 3: Train Model

Train the SWIN UNETR model:

```bash
python train_swin_unetr.py \
    --json-file bronco_dataset.json \
    --output-dir ./output \
    --max-epochs 500 \
    --batch-size 1 \
    --lr 1e-4 \
    --val-interval 5 \
    --feature-size 48 \
    --roi-size 96 96 96 \
    --spacing 1.5 1.5 2.0 \
    --cache-rate 1.0 \
    --num-workers 4 \
    --amp
```

**Key Training Arguments:**

**Data:**
- `--json-file`: Path to dataset JSON file
- `--output-dir`: Directory to save models and logs
- `--cache-rate`: Fraction of data to cache (1.0 = all)
- `--num-workers`: Number of workers for data caching

**Model:**
- `--feature-size`: SWIN UNETR feature size (24, 48, 96)
- `--roi-size`: ROI size for training crops (e.g., 96 96 96)
- `--spacing`: Target voxel spacing (e.g., 1.5 1.5 2.0)
- `--pretrained-weights`: Path to pretrained weights (optional)

**Training:**
- `--max-epochs`: Maximum training epochs
- `--batch-size`: Batch size (typically 1 for 3D medical imaging)
- `--lr`: Learning rate
- `--weight-decay`: Weight decay for optimizer
- `--val-interval`: Validate every N epochs
- `--checkpoint-interval`: Save checkpoint every N epochs
- `--sw-batch-size`: Sliding window batch size for validation
- `--amp`: Use automatic mixed precision

**Outputs:**
- `best_model.pth`: Best model based on validation Dice score
- `final_model.pth`: Model after final epoch
- `checkpoint_epoch_X.pth`: Periodic checkpoints
- `training_history.json`: Training and validation metrics

### Step 4: Run Inference

Run inference on test data:

```bash
python inference_swin_unetr.py \
    --json-file bronco_dataset.json \
    --model-path ./output/best_model.pth \
    --output-dir ./predictions \
    --feature-size 48 \
    --roi-size 96 96 96 \
    --spacing 1.5 1.5 2.0 \
    --sw-batch-size 4
```

**Or on a directory of images:**

```bash
python inference_swin_unetr.py \
    --input-dir /path/to/images \
    --model-path ./output/best_model.pth \
    --output-dir ./predictions \
    --feature-size 48 \
    --roi-size 96 96 96 \
    --spacing 1.5 1.5 2.0
```

**Inference Arguments:**
- `--json-file`: Use test split from dataset JSON
- `--input-dir`: Or specify directory with images
- `--model-path`: Path to trained model
- `--output-dir`: Directory to save predictions
- `--feature-size`: Must match training
- `--roi-size`: Must match training
- `--spacing`: Must match training
- `--sw-batch-size`: Sliding window batch size

## Model Architecture

SWIN UNETR is a hierarchical vision transformer-based segmentation model that combines:
- Swin Transformer encoder for feature extraction
- CNN-based decoder with skip connections
- Efficient self-attention mechanisms

**Reference:** Tang et al., "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis", CVPR 2022

## Data Format

### Input Images
Supported formats (via ImageInstance):
- DICOM series
- NIfTI (.nii, .nii.gz)
- NRRD (.nrrd)
- MetaImage (.mhd)

### Output Masks
- Format: NRRD
- Values: 0 (background), 1 (airways), 2 (airways_final), 3 (vessels)
- Spacing and orientation preserved from input

## Dataset JSON Format

```json
{
  "training": [
    {
      "image": "/path/to/image",
      "label": "/path/to/mask/bronco_final.nrrd",
      "patient_id": "patient_001",
      "dataset": "Moltest 1"
    }
  ],
  "validation": [...],
  "testing": [...]
}
```

## Training Tips

1. **Memory**: Start with smaller ROI size (64×64×64) if OOM errors occur
2. **Batch size**: Keep at 1 for 3D volumes, use `num_samples=4` in cropping
3. **Cache rate**: Use 1.0 if you have enough RAM, otherwise 0.5 or lower
4. **AMP**: Use `--amp` flag for faster training with mixed precision
5. **Pretrained weights**: Consider using self-supervised pretrained weights from MONAI
6. **Validation**: Monitor validation Dice scores to detect overfitting

## Computing Requirements

**Recommended:**
- GPU: NVIDIA GPU with 16GB+ VRAM
- RAM: 32GB+ for caching
- Storage: Fast SSD for data loading

**Minimum:**
- GPU: NVIDIA GPU with 8GB VRAM (reduce ROI size and cache rate)
- RAM: 16GB
- Storage: HDD acceptable but slower

## Monitoring Training

Training saves:
- Console output with loss and metrics
- `training_history.json` with all metrics
- Regular checkpoints for resuming

To resume training, load a checkpoint and continue from that epoch.

## Troubleshooting

**Out of Memory:**
- Reduce `--roi-size` (e.g., 64 64 64)
- Reduce `--cache-rate` (e.g., 0.5)
- Reduce `--sw-batch-size` (e.g., 2)
- Use `--amp` for mixed precision

**Slow Training:**
- Increase `--cache-rate` (more RAM usage)
- Use faster storage (SSD)
- Increase `--num-workers`
- Use `--amp` flag

**Poor Performance:**
- Increase training epochs
- Adjust learning rate
- Check data quality and preprocessing
- Try different `--feature-size` (24, 48, 96)

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{tang2022self,
  title={Self-supervised pre-training of swin transformers for 3d medical image analysis},
  author={Tang, Yucheng and Yang, Dong and Li, Wenqi and Roth, Holger R and Landman, Bennett and Xu, Daguang and Nath, Vishwesh and Hatamizadeh, Ali},
  booktitle={CVPR},
  year={2022}
}
```

## Additional Resources

- [MONAI Documentation](https://docs.monai.io/)
- [SWIN UNETR Paper](https://arxiv.org/abs/2201.01266)
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
