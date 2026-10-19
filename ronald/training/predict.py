"""Segment a new scan, restoring the original CT grid and physical metadata."""
import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from .imaging import read_image, prepare_image
from .model import UNet3D, sliding_predict
from .train import load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--lungs', required=True)
    p.add_argument('--mediastinum')
    p.add_argument('--series-uid')
    p.add_argument('--output', required=True, help='Native-grid three-class NIfTI/NRRD mask')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    cfg, prep = checkpoint['config'], checkpoint['preprocessing']
    model = UNet3D(cfg['channels']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    original = read_image(args.image, args.series_uid)
    image, ref, box = prepare_image(original, args.lungs, args.mediastinum, prep['spacing_xyz'], prep['margin_mm'], prep['hu'])
    probs = sliding_predict(model, image, cfg['patch_size'], device, cfg['overlap'], cfg['sw_batch_size'], cfg['amp'])
    array = np.zeros(tuple(reversed(ref.GetSize())), dtype=np.uint8)
    array[box] = probs.argmax(0).astype(np.uint8)
    result = sitk.GetImageFromArray(array)
    result.CopyInformation(ref)
    result = sitk.Resample(result, original, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result, args.output)


if __name__ == '__main__':
    main()
