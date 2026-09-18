from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace
import torch
import pytest
from ctc.ctc_loss_imp import ctc_loss_imp
from grid import grid_ranges, masked_step
from loss.loss import grad_distance


@pytest.mark.parametrize("concatenated", [False, True])
def test_ragged_repeated_and_empty_targets_match_pytorch(concatenated):
    torch.manual_seed(7)
    logits = torch.randn(8, 3, 4, dtype=torch.double, requires_grad=True)
    padded = torch.tensor([[1, 1, 2], [2, 3, 0], [0, 0, 0]])
    targets = torch.tensor([1, 1, 2, 2, 3]) if concatenated else padded
    lengths, sizes = torch.tensor([8, 5, 3]), torch.tensor([3, 2, 0])
    for reduction in ("none", "sum", "mean"):
        lp = logits.log_softmax(-1)
        actual = ctc_loss_imp(lp, targets, lengths, sizes, reduction=reduction)
        reference = torch.nn.functional.ctc_loss(lp, targets, lengths, sizes, reduction=reduction)
        torch.testing.assert_close(actual, reference)
        grad_a = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
        grad_b = torch.autograd.grad(reference.sum(), logits, retain_graph=True)[0]
        torch.testing.assert_close(grad_a, grad_b)
        assert torch.all(grad_a[5:, 1] == 0) and torch.all(grad_a[3:, 2] == 0)


def test_second_derivatives_pass_numerical_check():
    torch.manual_seed(4)
    x = torch.randn(4, 1, 3, dtype=torch.double, requires_grad=True)
    loss = lambda z: ctc_loss_imp(z.log_softmax(-1), torch.tensor([[1, 1]]), [4], [2])
    assert torch.autograd.gradcheck(loss, (x,))
    assert torch.autograd.gradgradcheck(loss, (x,))


def test_long_sequences_and_impossible_alignment():
    torch.manual_seed(3)
    x = (torch.randn(150, 1, 5, dtype=torch.double) * 8).requires_grad_()
    loss = ctc_loss_imp(x.log_softmax(-1), torch.tensor([[1, 2, 2, 3]]), [150], [4])
    gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient.square().sum(), x)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all() and torch.isfinite(curvature).all()
    lp = x[:2].log_softmax(-1)
    assert torch.isinf(ctc_loss_imp(lp, torch.tensor([[1, 1]]), [2], [2]))
    zero = ctc_loss_imp(lp, torch.tensor([[1, 1]]), [2], [2], zero_infinity=True)
    assert zero == 0
    assert torch.all(torch.autograd.grad(zero, x)[0] == 0)


def test_grid_coverage_and_no_adam_drift():
    assert grid_ranges(10, 4, 2) == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert grid_ranges(2, 4, 2) == [(0, 2)]
    with pytest.raises(ValueError):
        grid_ranges(10, 4, 4)
    x = torch.nn.Parameter(torch.ones(6))
    opt = torch.optim.AdamW([x], lr=.1, weight_decay=.1)
    x.grad = torch.ones_like(x)
    masked_step(opt, x, 0, 3)
    before = x.detach().clone()
    x.grad = torch.ones_like(x)
    masked_step(opt, x, 3, 6)
    torch.testing.assert_close(x[:3], before[:3], atol=0, rtol=0)
    assert torch.any(x[3:] != before[3:])


def test_tiny_top_gradient_fraction_does_not_select_zero_elements():
    args = SimpleNamespace(top_grad_percentage=.001, distance_metric="l2")
    assert grad_distance(torch.ones(2, 2), torch.zeros(2, 2), args) == 1


def test_actual_grid_loop_update_budget_and_resume_provenance(tmp_path, monkeypatch):
    import optimize
    monkeypatch.setattr(optimize, "plot_four_graphs", lambda *a, **k: None)
    monkeypatch.setattr(optimize, "plot_loss_curves", lambda *a, **k: None)
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 3).double()
    truth = torch.randn(6, 1, 2, dtype=torch.double)
    labels = torch.tensor([[1]])
    loss = ctc_loss_imp(model(truth).log_softmax(-1), labels, [6], [1])
    parameters = tuple(model.parameters())
    observed = torch.autograd.grad(loss, parameters)
    args = SimpleNamespace(max_iterations=5, exp_path=str(tmp_path), regularization="L1", reg_weight=.001,
                           distance_metric="l2", top_grad_percentage=1, patience=100, resume_grids=False)
    class CountAdam(torch.optim.Adam):
        calls = 0
        def step(self, closure=None):
            self.calls += 1
            return super().step(closure)
    estimate = torch.nn.Parameter(torch.zeros_like(truth))
    opt = CountAdam([estimate], lr=.1)
    schedule = torch.optim.lr_scheduler.MultiStepLR(opt, [1], gamma=.5)
    optimize.first_order_optimization_grid_loop(truth, estimate, [6], [1], opt, schedule, model,
        observed, parameters, labels, "fixture", args, grid_size=3, overlap=1)
    assert opt.calls == 5  # Remainder updates are no longer discarded.
    expected = estimate.detach().clone()
    assert len(list(tmp_path.glob("*_checkpoint.pt"))) == 3
    args.resume_grids = True
    with torch.no_grad():
        estimate.zero_()
    opt = CountAdam([estimate], lr=.1)
    schedule = torch.optim.lr_scheduler.MultiStepLR(opt, [1], gamma=.5)
    optimize.first_order_optimization_grid_loop(truth, estimate, [6], [1], opt, schedule, model,
        observed, parameters, labels, "fixture", args, grid_size=3, overlap=1)
    torch.testing.assert_close(estimate, expected, atol=0, rtol=0)
    assert opt.calls == 0
    args.reg_weight = .5
    with pytest.raises(ValueError, match="provenance"):
        optimize.first_order_optimization_grid_loop(truth, estimate, [6], [1], opt, schedule, model,
            observed, parameters, labels, "fixture", args, grid_size=3, overlap=1)
