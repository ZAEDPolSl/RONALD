"""Plain convolutional 3D UNet and bounded-GPU-memory sliding-window inference."""
import itertools
import numpy as np
import torch
from torch import nn


def block(in_channels, out_channels):
    groups = min(8, out_channels)
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, out_channels), nn.LeakyReLU(.01, inplace=True),
                         nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, out_channels), nn.LeakyReLU(.01, inplace=True))


class UNet3D(nn.Module):
    def __init__(self, channels=(24, 48, 96, 192, 320)):
        super().__init__()
        self.encoders = nn.ModuleList([block(a, b) for a, b in zip((1, *channels[:-1]), channels)])
        self.downsamplers = nn.ModuleList([nn.Conv3d(c, c, 2, stride=2) for c in channels[:-1]])
        self.upsamplers = nn.ModuleList([nn.ConvTranspose3d(channels[i + 1], channels[i], 2, stride=2)
                                        for i in reversed(range(len(channels) - 1))])
        self.decoders = nn.ModuleList([block(2 * channels[i], channels[i])
                                      for i in reversed(range(len(channels) - 1))])
        self.head = nn.Conv3d(channels[0], 3, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(self.downsamplers[i - 1](x) if i else x)
            skips.append(x)
        for upsample, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips[:-1])):
            x = upsample(x)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)


@torch.inference_mode()
def sliding_predict(model, image, patch_size, device, overlap=.5, batch_size=2, amp=False):
    """Return C,Z,Y,X probabilities on CPU. Image is a Z,Y,X NumPy array.

    CPU accumulation prevents keeping the entire CT on GPU. Gaussian overlap
    blending reduces window seams. CPU RAM still scales with full crop size.
    """
    shape = image.shape
    patch = tuple(patch_size)
    padded = np.pad(image, [(0, max(0, p - n)) for p, n in zip(patch, shape)])
    starts = []
    for n, p in zip(padded.shape, patch):
        step = max(1, int(p * (1 - overlap)))
        starts.append(sorted(set([*range(0, n - p + 1, step), n - p])))
    axes = [np.linspace(-1, 1, p, dtype=np.float32) for p in patch]
    mesh = np.meshgrid(*axes, indexing='ij', sparse=True)
    weight = np.maximum(np.exp(-sum(a ** 2 for a in mesh) / (2 * .25 ** 2)), 1e-4).astype(np.float32)
    scores = np.zeros((3, *padded.shape), dtype=np.float32)
    counts = np.zeros(padded.shape, dtype=np.float32)
    positions = iter(itertools.product(*starts))
    while True:
        batch = list(itertools.islice(positions, batch_size))
        if not batch:
            break
        boxes = [tuple(slice(s, s + p) for s, p in zip(pos, patch)) for pos in batch]
        inputs = torch.from_numpy(np.stack([padded[b] for b in boxes])[:, None].astype(np.float32)).to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
            output = model(inputs)
        probs = output.float().softmax(1).cpu().numpy()
        for b, prob in zip(boxes, probs):
            scores[(slice(None), *b)] += prob * weight
            counts[b] += weight
    scores /= counts[None]
    return scores[:, :shape[0], :shape[1], :shape[2]]
