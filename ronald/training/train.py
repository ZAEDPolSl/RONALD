"""Train/evaluate a plain 3D UNet with TensorBoard, disk logs, and exact epoch resume."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .data import PatchDataset
from .evaluation import evaluate
from .losses import PseudoLabelLoss
from .model import UNet3D
from .prepare import validate_splits

LOG = logging.getLogger(__name__)


def validate_config(c):
    for key in ('epochs', 'patches_per_case', 'batch_size', 'validation_interval', 'visualization_interval',
                'checkpoint_interval', 'sw_batch_size', 'patience'):
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if c['num_workers'] < 0 or c['visualization_cases'] < 0:
        raise ValueError('Workers/visualization_cases must be nonnegative')
    if len(c['channels']) < 2 or any(v <= 0 for v in c['channels']):
        raise ValueError('Need at least two positive channel widths')
    factor = 2 ** (len(c['channels']) - 1)
    if len(c['patch_size']) != 3 or any(v < factor * 2 or v % factor for v in c['patch_size']):
        raise ValueError(f'Patch dimensions must be multiples of {factor}, at least {factor * 2}')
    if not 0 < c['unannotated_weight'] <= 1 or not 0 <= c['overlap'] < 1:
        raise ValueError('Invalid unannotated weight or overlap')
    probabilities = c['sampling_probabilities']
    if len(probabilities) != 4 or min(probabilities) < 0 or not np.isclose(sum(probabilities), 1):
        raise ValueError('Four sampling probabilities must be nonnegative and sum to one')
    if c['learning_rate'] <= 0 or c['gradient_clip'] <= 0 or c['weight_decay'] < 0:
        raise ValueError('Invalid optimizer configuration')
    loss = c['loss']
    if len(loss['class_weights']) != 3 or min(loss['class_weights']) <= 0:
        raise ValueError('Need three positive class weights')
    if any(loss[k] < 0 for k in ('ce_weight', 'tversky_weight', 'cldice_weight', 'nodule_weight')):
        raise ValueError('Loss weights cannot be negative')
    if loss['skeleton_iterations'] < 0 or min(loss['tversky_alpha'], loss['tversky_beta']) <= 0:
        raise ValueError('Invalid topology/Tversky parameters')


def seed_everything(seed, deterministic):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(state, path):
    temp = Path(str(path) + '.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    # Checkpoints contain Python/NumPy RNG state. Load only checkpoints you trust.
    return torch.load(path, map_location='cpu', weights_only=False)


def provenance():
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'SimpleITK', 'matplotlib', 'tensorboard'):
        packages[name] = importlib.metadata.version(name)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob('*.py'))}
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'source_sha256': source_hashes,
            'git_commit': commit, 'cuda': torch.version.cuda, 'argv': sys.argv}


def run(args):
    manifest_path = Path(args.manifest).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    validate_splits(manifest['splits'])
    root = manifest_path.parent
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    seed_everything(config['seed'], config['deterministic'])
    device = torch.device(args.device)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('Supported devices: cpu or cuda[:index]')
    out = Path(args.output).resolve()
    if not args.resume and not args.evaluate and out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new run directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / 'manifest.json').exists():
        if hashlib.sha256((out / 'manifest.json').read_bytes()).hexdigest() != manifest_hash:
            raise ValueError('Output directory belongs to a different manifest')
        if json.loads((out / 'config.json').read_text()) != config:
            raise ValueError('Output directory belongs to a different configuration')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'training.log')], force=True)
    model = UNet3D(config['channels']).to(device)
    if args.evaluate:
        checkpoint = load_checkpoint(args.evaluate)
        if checkpoint['config'] != config or checkpoint['manifest_sha256'] != manifest_hash:
            raise ValueError('Evaluation requires original config and manifest')
        model.load_state_dict(checkpoint['model'])
        metrics = evaluate(model, manifest['splits'][args.split], root, config, device,
                           out / args.split, checkpoint['epoch'] + 1, visualize=True)
        LOG.info('Evaluation complete: %s', metrics)
        return
    dataset = PatchDataset(manifest['splits']['training'], root, config)
    if not any(c['has_nodules'] for c in dataset.cases):
        LOG.warning('No training scan has usable nodule masks: nodule penalty will be zero throughout training')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'] and device.type == 'cuda')
    criterion = PseudoLabelLoss(**config['loss']).to(device)
    start, best, bad_checks, global_step = 0, -float('inf'), 0, 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint['config'] != config or checkpoint['manifest_sha256'] != manifest_hash:
            raise ValueError('Resume requires unchanged configuration and manifest')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'])
        start, best = checkpoint['epoch'] + 1, checkpoint['best']
        bad_checks, global_step = checkpoint['bad_checks'], checkpoint['global_step']
        LOG.info('Resuming after completed epoch %d', start)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'manifest.json').write_bytes(manifest_bytes)
    (out / 'environment.json').write_text(json.dumps(provenance(), indent=2))
    writer = SummaryWriter(str(out / 'tensorboard'), purge_step=global_step + 1 if args.resume else None)
    try:
        for epoch in range(start, config['epochs']):
            t0 = time.monotonic()
            dataset.epoch = epoch
            # Recreate workers each epoch. Sampling/shuffle does not depend on worker order.
            generator = torch.Generator().manual_seed(config['seed'] + epoch)
            loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['num_workers'], pin_memory=device.type == 'cuda',
                                generator=generator, persistent_workers=False)
            model.train()
            totals, count = {}, 0
            for batch in loader:
                x, y, n, weight = [batch[k].to(device, non_blocking=True) for k in ('image', 'label', 'nodules', 'weight')]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config['amp'] and device.type == 'cuda'):
                    logits = model(x)
                loss, parts = criterion(logits, y, n, weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss epoch={epoch + 1} step={global_step}')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                batch_n = len(x)
                count += batch_n
                for key, value in {'total': float(loss.detach()), **parts}.items():
                    totals[key] = totals.get(key, 0.) + value * batch_n
                    writer.add_scalar(f'train_step/{key}', value, global_step)
                writer.add_scalar('train_step/gradient_norm', float(grad_norm), global_step)
                if global_step % 20 == 0:
                    LOG.info('Epoch %d step %d loss=%.5f', epoch + 1, global_step, float(loss.detach()))
            metrics = {}
            improved = False
            if (epoch + 1) % config['validation_interval'] == 0 or epoch == 0 or epoch + 1 == config['epochs']:
                metrics = evaluate(model, manifest['splits']['validation'], root, config, device,
                                   out / 'validation', epoch + 1, writer,
                                   visualize=epoch == 0 or (epoch + 1) % config['visualization_interval'] == 0,
                                   log_step=global_step)
                # Foreground Dice selects segmentation checkpoint; NPV is reported but
                # dominated by background and unsuitable as the sole selection criterion.
                dice = [metrics[f'all/{name}_dice'] for name in ('airway', 'vessel')]
                if any(v is None for v in dice):
                    raise ValueError('Validation must contain both airway and vessel target classes')
                score = float(np.mean(dice))
                improved = score > best + config['min_delta']
                if improved:
                    best, bad_checks = score, 0
                else:
                    bad_checks += 1
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            record = {'epoch': epoch + 1, 'global_step': global_step, 'seconds': time.monotonic() - t0,
                      'learning_rate': lr, 'train': {k: v / count for k, v in totals.items()},
                      'validation': metrics, 'best_foreground_dice': best}
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            for key, value in record['train'].items():
                writer.add_scalar(f'train_epoch/{key}', value, global_step)
            writer.add_scalar('learning_rate', lr, global_step)
            writer.flush()
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'rng': rng_state(), 'epoch': epoch, 'global_step': global_step, 'best': best,
                     'bad_checks': bad_checks, 'config': config, 'manifest_sha256': manifest_hash,
                     'preprocessing': manifest['preprocessing']}
            atomic_save(state, out / 'last.pt')
            if improved:
                atomic_save(state, out / 'best.pt')
            if (epoch + 1) % config['checkpoint_interval'] == 0:
                atomic_save(state, out / f'epoch_{epoch + 1:04d}.pt')
            LOG.info('Epoch %d/%d loss=%.5f best_fg_dice=%.5f time=%.1fs', epoch + 1,
                     config['epochs'], record['train']['total'], best, record['seconds'])
            if bad_checks >= config['patience']:
                LOG.info('Early stopping after %d validation checks without improvement', bad_checks)
                break
    finally:
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--resume', help='Trusted checkpoint; resumes from next epoch')
    group.add_argument('--evaluate', help='Trusted checkpoint; no training')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    args = p.parse_args()
    try:
        run(args)
    except Exception:
        LOG.exception('Training/evaluation failed')
        raise


if __name__ == '__main__':
    main()
