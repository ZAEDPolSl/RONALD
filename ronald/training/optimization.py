"""AMP optimizer updates with bounded overflow recovery on the same batch."""
import logging
import torch

LOG = logging.getLogger(__name__)


def optimizer_update(forward_loss, model, optimizer, scaler, gradient_clip, max_retries=16):
    """Retry AMP overflows without advancing weights or discarding the batch.

    The model uses no stochastic layers or running normalization statistics.
    Persistent overflow and unscaled nonfinite gradients remain fatal.
    """
    for attempt in range(max_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = forward_loss()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite = bool(torch.stack([g.isfinite().all() for g in gradients]).all())
        if finite:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                                  error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            return loss, parts, norm
        if not scaler.is_enabled() or attempt == max_retries:
            raise FloatingPointError(f'Nonfinite gradients after {attempt} AMP retries')
        old_scale = scaler.get_scale()
        # unscale_ recorded the overflow: GradScaler skips the optimizer update.
        scaler.step(optimizer)
        scaler.update()
        LOG.warning('AMP gradient overflow: scale %g -> %g; retrying same batch (%d/%d)',
                    old_scale, scaler.get_scale(), attempt + 1, max_retries)
