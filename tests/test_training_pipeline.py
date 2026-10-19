"""Focused regression tests for loss semantics, geometry, leakage, and inference."""
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from ronald.training.imaging import prepare_image, grow_nodules
from ronald.training.losses import PseudoLabelLoss
from ronald.training.model import UNet3D, sliding_predict
from ronald.training.prepare import split_cases, validate_splits
from ronald.training.evaluation import segmentation_metrics


def test_nodule_penalty_overrides_foreground_and_has_correct_gradient():
    logits = torch.zeros(1, 3, 5, 5, 5, requires_grad=True)
    labels = torch.full((1, 5, 5, 5), 2, dtype=torch.long)
    nodules = torch.zeros_like(labels, dtype=torch.bool)
    nodules[:, 2, 2, 2] = True
    criterion = PseudoLabelLoss(ce_weight=0, tversky_weight=0, cldice_weight=0, nodule_weight=1)
    loss, _ = criterion(logits, labels, nodules, torch.ones(1))
    loss.backward()
    assert logits.grad[0, 0, 2, 2, 2] < 0
    assert (logits.grad[0, 1:, 2, 2, 2] > 0).all()
    assert torch.count_nonzero(logits.grad) == 3
    no_nodule, _ = criterion(logits, labels, torch.zeros_like(nodules), torch.ones(1))
    assert no_nodule == 0


def test_scan_weight_not_cancelled_and_padding_ignored():
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 5, 5, 5, requires_grad=True)
    target = torch.zeros(1, 5, 5, 5, dtype=torch.long)
    target[:, 1:3, 1:3, 1:3] = 2
    nodules = torch.zeros_like(target, dtype=torch.bool)
    criterion = PseudoLabelLoss(skeleton_iterations=2)
    full, _ = criterion(logits, target, nodules, torch.ones(1))
    half, _ = criterion(logits, target, nodules, torch.tensor([.5]))
    torch.testing.assert_close(half, full * .5)
    ignored = torch.full_like(target, 255)
    loss, _ = criterion(logits, ignored, nodules, torch.ones(1))
    assert loss == 0
    full.backward()
    assert torch.isfinite(logits.grad).all()


def test_patient_splits_and_duplicates():
    cases = [{'patient_id': f'p{i // 2}', 'case_id': f'c{i}', 'dataset': 'LUNA',
              'has_nodules': True, 'image': f'/ct/{i}'} for i in range(20)]
    a = split_cases(cases)
    assert a == split_cases(cases)
    assert all(a.values())
    validate_splits(a)
    broken = {'training': [dict(cases[0], image_sha256='same')],
              'validation': [dict(cases[-1], image_sha256='same')]}
    with pytest.raises(ValueError, match='Leakage'):
        validate_splits(broken)


def test_crop_includes_mediastinum_and_geometry(tmp_path):
    image = sitk.GetImageFromArray(np.zeros((20, 24, 32), dtype=np.float32))
    image.SetSpacing((.7, .8, 1.5))
    image.SetOrigin((23, -40, 12))
    image.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
    lungs = np.zeros((20, 24, 32), dtype=np.uint8)
    lungs[3:17, 4:20, 3:11] = 1
    lungs[3:17, 4:20, 21:29] = 1
    mask = sitk.GetImageFromArray(lungs)
    mask.CopyInformation(image)
    path = tmp_path / 'lungs.nii.gz'
    sitk.WriteImage(mask, str(path))
    crop, ref, box = prepare_image(image, path, None, [.7, .8, 1.5], 0, [-1000, 500])
    assert crop.shape[2] >= 25  # includes the gap between both lungs
    assert ref.GetDirection() == (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    assert np.isfinite(crop).all()


def test_grow_respects_world_coordinates_and_rejects_spill():
    ct = np.full((25, 25, 25), -950, dtype=np.float32)
    zz, yy, xx = np.ogrid[:25, :25, :25]
    lesion = (zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 12) ** 2 <= 3 ** 2
    ct[lesion] = -100
    image = sitk.GetImageFromArray(ct)
    image.SetOrigin((100., -50., 80.))
    image.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
    xyz = image.TransformIndexToPhysicalPoint((12, 12, 12))
    row = dict(zip(('coordX', 'coordY', 'coordZ'), xyz), diameter_mm=8)
    mask, audit = grow_nodules(image, [row])
    assert audit[0]['accepted']
    np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), lesion)
    dense = sitk.GetImageFromArray(np.zeros_like(ct))
    dense.CopyInformation(image)
    mask, audit = grow_nodules(dense, [row])
    assert not audit[0]['accepted']
    assert not sitk.GetArrayViewFromImage(mask).any()


def test_sliding_window_blending_and_unet_shape():
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x * 0, x * 0 + 1, x * 0 + 2], 1)
    image = np.zeros((13, 19, 11), dtype=np.float32)
    pred = sliding_predict(Constant(), image, (8, 8, 8), torch.device('cpu'), batch_size=3)
    assert pred.shape == (3, *image.shape)
    expected = torch.tensor([0., 1., 2.]).softmax(0).numpy()
    np.testing.assert_allclose(pred, np.broadcast_to(expected[:, None, None, None], pred.shape), atol=1e-6)
    model = UNet3D((4, 8, 16))
    assert model(torch.zeros(1, 1, 16, 16, 16)).shape == (1, 3, 16, 16, 16)


def test_metrics_do_not_report_missing_nodules_as_perfect():
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    result = segmentation_metrics(labels, labels, labels)
    assert result['nodule_foreground_leakage'] is None
    assert result['vessel_dice'] is None
    assert result['vessel_npv'] == 1


def test_end_to_end_train_resume_evaluate_predict(tmp_path):
    import csv
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from ronald.training.train import load_checkpoint
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(tmp_path / 'mpl'))
    def run(module, *args):
        result = subprocess.run([sys.executable, '-m', f'ronald.training.{module}', *map(str, args)],
                                cwd=repo, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    rows = []
    rng = np.random.default_rng(42)
    shape = (16, 16, 16)
    for i, split in enumerate(['training', 'validation', 'test']):
        case = tmp_path / f'case{i}'
        case.mkdir()
        arrays = {'image': rng.normal(-700, 20, shape).astype(np.float32),
                  'lungs': np.ones(shape, np.uint8), 'airways': np.zeros(shape, np.uint8),
                  'vessels': np.zeros(shape, np.uint8), 'nodules': np.zeros(shape, np.uint8)}
        arrays['airways'][:, 4:6, 4:6] = 1
        arrays['vessels'][:, 10:12, 10:12] = 1
        arrays['nodules'][7:9, 10:12, 10:12] = 1  # deliberately conflicting pseudo-label
        row = dict(case_id=f'c{i}', patient_id=f'p{i}', dataset='synthetic', split=split)
        for name, array in arrays.items():
            image = sitk.GetImageFromArray(array)
            image.SetOrigin((10., -20., 30.))
            path = case / f'{name}.nii.gz'
            sitk.WriteImage(image, str(path))
            row[name] = str(path)
        rows.append(row)
    inventory = tmp_path / 'inventory.csv'
    with inventory.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cache = tmp_path / 'cache'
    run('prepare', '--inventory', inventory, '--output', cache, '--spacing', 1, 1, 1, '--margin-mm', 0)
    labels = np.load(cache / 'c0' / 'label.npy')
    assert (labels[7:9, 10:12, 10:12] == 0).all()
    cfg = json.loads((repo / 'configs/training/unet.json').read_text())
    cfg.update(epochs=2, patch_size=[16]*3, channels=[4, 8], patches_per_case=1,
               batch_size=1, num_workers=0, amp=False, validation_interval=1,
               visualization_interval=1, visualization_cases=1, checkpoint_interval=1)
    cfg['loss']['skeleton_iterations'] = 1
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    common = ['--manifest', cache / 'manifest.json', '--config', config_path, '--device', 'cpu']
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    run('train', *common, '--output', full)
    run('train', *common, '--output', resumed, '--resume', full / 'epoch_0001.pt')
    a, b = load_checkpoint(full / 'last.pt'), load_checkpoint(resumed / 'last.pt')
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['global_step'] == b['global_step'] == 2
    assert (full / 'training.log').stat().st_size
    assert len((full / 'metrics.jsonl').read_text().splitlines()) == 2
    assert list((full / 'validation').glob('*.png'))
    events = EventAccumulator(str(full / 'tensorboard')).Reload()
    assert events.Tags()['scalars'] and events.Tags()['images']
    run('train', *common, '--output', tmp_path / 'evaluation', '--evaluate', full / 'best.pt', '--split', 'test')
    prediction = tmp_path / 'prediction.nii.gz'
    run('predict', '--checkpoint', full / 'best.pt', '--image', rows[2]['image'],
        '--lungs', rows[2]['lungs'], '--output', prediction, '--device', 'cpu')
    result = sitk.ReadImage(str(prediction))
    original = sitk.ReadImage(rows[2]['image'])
    assert result.GetSize() == original.GetSize()
    assert result.GetOrigin() == original.GetOrigin()
    assert result.GetSpacing() == original.GetSpacing()
    assert result.GetDirection() == original.GetDirection()
    assert set(np.unique(sitk.GetArrayFromImage(result))) <= {0, 1, 2}


def test_deterministic_morphology_matches_pooling():
    from torch.nn import functional as F
    from ronald.training.losses import soft_erode, soft_dilate
    torch.manual_seed(5)
    x = torch.rand(2, 2, 7, 8, 9, requires_grad=True)
    expected = torch.minimum(torch.minimum(-F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0)),
                                           -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))),
                             -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1)))
    torch.testing.assert_close(soft_erode(x), expected)
    torch.testing.assert_close(soft_dilate(x), F.max_pool3d(x, 3, 1, 1))
    soft_dilate(soft_erode(x)).sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA hardware')
def test_cuda_deterministic_training_step():
    from ronald.training.train import seed_everything
    seed_everything(42, True)
    model = UNet3D((4, 8)).cuda()
    x = torch.rand(1, 1, 16, 16, 16, device='cuda')
    y = torch.randint(0, 3, (1, 16, 16, 16), device='cuda')
    criterion = PseudoLabelLoss(skeleton_iterations=1).cuda()
    with torch.autocast('cuda'):
        logits = model(x)
    loss, _ = criterion(logits, y, y == 0, torch.ones(1, device='cuda'))
    loss.backward()
    assert torch.isfinite(loss)
