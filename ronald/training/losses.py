"""Per-scan pseudo-label loss with nodule exclusion and thin-structure support."""
import torch
from torch import nn
from torch.nn import functional as F


def axis_extreme(x, axis, erode=False):
    """Three-voxel extrema without nondeterministic CUDA max-pool backward.

    Inputs are probabilities in [0,1]; padding 1 for erosion and 0 for dilation
    implements the same boundary values as the morphological pooling operators.
    """
    padding = [0] * 6
    offset = 2 * (4 - axis)
    padding[offset:offset + 2] = [1, 1]
    padded = F.pad(x, padding, value=1. if erode else 0.)
    operation = torch.minimum if erode else torch.maximum
    return operation(operation(padded.narrow(axis, 0, x.shape[axis]), x),
                     padded.narrow(axis, 2, x.shape[axis]))


def soft_erode(x):
    return torch.minimum(torch.minimum(axis_extreme(x, 2, True), axis_extreme(x, 3, True)),
                         axis_extreme(x, 4, True))


def soft_dilate(x):
    for axis in (2, 3, 4):
        x = axis_extreme(x, axis)
    return x


def soft_skeleton(x, iterations):
    opened = soft_dilate(soft_erode(x))
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class PseudoLabelLoss(nn.Module):
    def __init__(self, class_weights=(.2, 1., 1.5), ce_weight=1., tversky_weight=1.,
                 cldice_weight=.2, nodule_weight=2., tversky_alpha=.3,
                 tversky_beta=.7, skeleton_iterations=10):
        super().__init__()
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        self.coefficients = dict(ce=ce_weight, tversky=tversky_weight, cldice=cldice_weight, nodule=nodule_weight)
        self.alpha, self.beta, self.iterations = tversky_alpha, tversky_beta, skeleton_iterations

    def forward(self, logits, labels, nodules, sample_weight):
        # Float32 reductions even with autocast; ignore both ambiguous labels and padding.
        logits = logits.float()
        valid = labels != 255
        nodules = nodules.bool() & valid
        target = labels.clone()
        target[nodules] = 0  # enforce precedence even for independently constructed batches
        safe = target.masked_fill(~valid, 0)
        p = logits.softmax(1)
        dims = (2, 3, 4)
        log_prob = F.log_softmax(logits, 1)
        ce_map = -log_prob.gather(1, safe[:, None]).squeeze(1) * self.class_weights[safe]
        ce = (ce_map * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
        truth = F.one_hot(safe, 3).movedim(-1, 1).float()[:, 1:] * valid[:, None]
        foreground = p[:, 1:] * valid[:, None]
        tp = (foreground * truth).sum(dims)
        fp = (foreground * (1 - truth)).sum(dims)
        fn = ((1 - foreground) * truth).sum(dims)
        tversky = (1 - (tp + 1e-5) / (tp + self.alpha * fp + self.beta * fn + 1e-5)).mean(1)
        cldice = torch.zeros_like(ce)
        if self.coefficients['cldice']:
            pred_sk = soft_skeleton(foreground, self.iterations)
            with torch.no_grad():
                true_sk = soft_skeleton(truth, self.iterations)
            precision = ((pred_sk * truth).sum(dims) + 1e-5) / (pred_sk.sum(dims) + 1e-5)
            sensitivity = ((true_sk * foreground).sum(dims) + 1e-5) / (true_sk.sum(dims) + 1e-5)
            present = truth.sum(dims) > 0
            class_loss = 1 - 2 * precision * sensitivity / (precision + sensitivity).clamp_min(1e-5)
            cldice = (class_loss * present).sum(1) / present.sum(1).clamp_min(1)
        nodule_ce = -log_prob[:, 0]
        nodule = (nodule_ce * nodules).flatten(1).sum(1) / nodules.flatten(1).sum(1).clamp_min(1)
        parts = dict(ce=ce, tversky=tversky, cldice=cldice, nodule=nodule)
        # Do NOT divide by sum(weights): homogeneous unannotated batches must still
        # contribute less. The nodule term is zero when no nodule voxel is sampled.
        total = sum(self.coefficients[k] * v for k, v in parts.items())
        loss = (total * sample_weight).mean()
        return loss, {k: float((v.detach() * sample_weight).mean()) for k, v in parts.items()}
