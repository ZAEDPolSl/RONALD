"""Memory-mapped volumes with deterministic per-epoch, per-patch sampling."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, cases, root, config):
        self.cases, self.root, self.config = cases, Path(root), config
        self.epoch = 0

    def __len__(self):
        return len(self.cases) * self.config['patches_per_case']

    def __getitem__(self, index):
        cfg = self.config
        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], self.epoch, index]))
        case = self.cases[index // cfg['patches_per_case']]
        folder = self.root / case['cache']
        image = np.load(folder / 'image.npy', mmap_mode='r')
        label = np.load(folder / 'label.npy', mmap_mode='r')
        nodules = np.load(folder / 'nodules.npy', mmap_mode='r')
        with np.load(folder / 'indices.npz') as pools:
            kind = rng.choice(['uniform', 'airway', 'vessel', 'nodule'], p=cfg['sampling_probabilities'])
            if kind != 'uniform' and len(pools[kind]):
                center = np.array(np.unravel_index(rng.choice(pools[kind]), image.shape))
            else:
                center = np.array([rng.integers(n) for n in image.shape])
        patch = np.array(cfg['patch_size'])
        # Jitter the selected center but keep it inside the patch.
        center += np.array([rng.integers(-max(1, p // 4), max(1, p // 4) + 1) for p in patch])
        start = np.clip(center - patch // 2, 0, np.maximum(0, np.array(image.shape) - patch))
        box = tuple(slice(int(s), int(s + p)) for s, p in zip(start, patch))
        padding = [(0, max(0, int(p) - n)) for p, n in zip(patch, image[box].shape)]
        x = np.pad(np.array(image[box], dtype=np.float32), padding, constant_values=0)
        y = np.pad(np.array(label[box], dtype=np.int64), padding, constant_values=255)
        n = np.pad(np.array(nodules[box], dtype=bool), padding, constant_values=False)
        for axis in range(3):
            if rng.random() < .5:
                x, y, n = [np.flip(a, axis) for a in (x, y, n)]
        if rng.random() < .5:
            x = np.clip(x * rng.uniform(.9, 1.1) + rng.uniform(-.05, .05), 0, 1)
        weight = 1. if case['has_nodules'] else cfg['unannotated_weight']
        return {'image': torch.from_numpy(x[None].copy()), 'label': torch.from_numpy(y.copy()),
                'nodules': torch.from_numpy(n.copy()), 'weight': torch.tensor(weight, dtype=torch.float32)}
