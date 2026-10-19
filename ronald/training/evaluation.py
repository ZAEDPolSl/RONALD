"""Whole-crop metrics and class-colored three-plane overlays."""
import csv
import json
from pathlib import Path
import numpy as np
from .model import sliding_predict


def divide(a, b):
    return float(a / b) if b else None


def segmentation_metrics(prediction, label, nodules):
    valid = label != 255
    result = {}
    for value, name in [(1, 'airway'), (2, 'vessel')]:
        p, y = prediction == value, label == value
        tp = int((p & y & valid).sum())
        fp = int((p & ~y & valid).sum())
        fn = int((~p & y & valid).sum())
        tn = int((~p & ~y & valid).sum())
        for metric, score in {'npv': divide(tn, tn + fn), 'ppv': divide(tp, tp + fp),
                              'sensitivity': divide(tp, tp + fn), 'specificity': divide(tn, tn + fp),
                              'dice': divide(2 * tp, 2 * tp + fp + fn),
                              'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}.items():
            result[f'{name}_{metric}'] = score
    mask = nodules.astype(bool) & valid
    result['nodule_voxels'] = int(mask.sum())
    result['nodule_foreground_leakage'] = divide(((prediction > 0) & mask).sum(), mask.sum())
    return result


def overlay_figure(image, labels, prediction, nodules, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    center = [s // 2 for s in image.shape]
    if nodules.any():
        # Center on an actual nodule voxel, not the midpoint between separate nodules.
        points = np.argwhere(nodules)
        center = points[len(points) // 2].tolist()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    colors = {1: (1., .4, 0.), 2: (0., .7, 1.), 255: (.7, .7, .7)}
    for axis, plane in enumerate(['axial', 'coronal', 'sagittal']):
        ct = np.take(image, center[axis], axis=axis)
        n = np.take(nodules, center[axis], axis=axis)
        for column, mask in enumerate([None, labels, prediction]):
            ax = axes[axis, column]
            ax.imshow(ct, cmap='gray', vmin=0, vmax=1, origin='lower')
            if mask is not None:
                selected = np.take(mask, center[axis], axis=axis)
                rgba = np.zeros((*selected.shape, 4))
                for value, color in colors.items():
                    rgba[selected == value] = (*color, .55)
                ax.imshow(rgba, origin='lower')
            if n.any():
                rgba = np.zeros((*n.shape, 4))
                rgba[n.astype(bool)] = (1, 0, 1, .65)
                ax.imshow(rgba, origin='lower')
            ax.set_title(f'{plane}: {["CT", "pseudo-label", "prediction"][column]}')
            ax.axis('off')
    fig.suptitle(title)
    fig.legend(handles=[Patch(color=c, label=n) for c, n in [('tab:orange', 'airway + wall'),
               ('deepskyblue', 'vessel'), ('magenta', 'nodule → background'), ('gray', 'ignored overlap')]], loc='outside lower center', ncol=4)
    return fig


def evaluate(model, cases, root, config, device, out, epoch, writer=None, visualize=False, log_step=None):
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    log_step = epoch if log_step is None else log_step
    model.eval()
    for index, case in enumerate(cases):
        folder = Path(root) / case['cache']
        image = np.load(folder / 'image.npy', mmap_mode='r')
        labels = np.load(folder / 'label.npy', mmap_mode='r')
        nodules = np.load(folder / 'nodules.npy', mmap_mode='r')
        probs = sliding_predict(model, image, config['patch_size'], device, config['overlap'], config['sw_batch_size'], config['amp'])
        pred = probs.argmax(0).astype(np.uint8)
        row = {'case_id': case['case_id'], 'dataset': case['dataset'], 'has_nodules': case['has_nodules'],
               **segmentation_metrics(pred, labels, nodules)}
        rows.append(row)
        if visualize and index < config['visualization_cases']:
            fig = overlay_figure(image, labels, pred, nodules, case['case_id'])
            fig.savefig(out / f'epoch_{epoch:04d}_{case["case_id"]}.png', dpi=120)
            if writer:
                writer.add_figure(f'validation/{case["case_id"]}', fig, log_step, close=False)
            plt.close(fig)
        del probs, pred
    if not rows:
        raise ValueError('No evaluation cases')
    with (out / f'epoch_{epoch:04d}_cases.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    metrics = {}
    names = [k for k in rows[0] if k not in ('case_id', 'dataset', 'has_nodules')]
    for group, selected in [('all', rows), *[(d, [r for r in rows if r['dataset'] == d]) for d in sorted({r['dataset'] for r in rows})]]:
        for key in names:
            values = [r[key] for r in selected if r[key] is not None]
            metrics[f'{group}/{key}'] = float(np.mean(values)) if values else None
            metrics[f'{group}/{key}_n'] = len(values)
    (out / f'epoch_{epoch:04d}_summary.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    if writer:
        for key, value in metrics.items():
            if value is not None:
                writer.add_scalar(f'validation/{key}', value, log_step)
    return metrics
