"""Overflow must not update weights or drop batches; true errors stay fatal."""
import pytest
import torch
from ronald.training.optimization import optimizer_update


def test_amp_overflow_retries_same_batch_and_updates_once():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=8.)
    calls = []
    def forward():
        calls.append(model.weight.detach().clone())
        loss = model(torch.ones(1, 1)).sum()
        # Simulate backward overflow on the first scaled attempt.
        if len(calls) == 1:
            loss.register_hook(lambda g: g * float('inf'))
        return loss, {}
    _, _, norm = optimizer_update(forward, model, optimizer, scaler, 12)
    assert len(calls) == 2
    torch.testing.assert_close(calls[0], calls[1])
    torch.testing.assert_close(model.weight, torch.tensor([[.9]]))
    assert norm == 1 and scaler.get_scale() == 4


@pytest.mark.parametrize('amp', [True, False])
def test_persistent_bad_gradient_is_fatal_without_weight_update(amp):
    model = torch.nn.Linear(1, 1, bias=False)
    original = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cpu', enabled=amp)
    def forward():
        loss = model(torch.ones(1, 1)).sum()
        loss.register_hook(lambda g: g * float('inf'))
        return loss, {}
    with pytest.raises(FloatingPointError, match='Nonfinite gradients'):
        optimizer_update(forward, model, optimizer, scaler, 12, max_retries=2)
    torch.testing.assert_close(model.weight, original)


def test_nonfinite_loss_is_fatal():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    with pytest.raises(FloatingPointError, match='Nonfinite loss'):
        optimizer_update(lambda: (model(torch.ones(1, 1)).sum()*float('nan'), {}),
                         model, optimizer, torch.amp.GradScaler('cpu'), 12)
