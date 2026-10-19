import torch

from learned_impact.metrics import binary_fail_f1, mean_metrics


def test_fail_f1_perfect():
    logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])  # pred 1, 0
    target = torch.tensor([1, 0])
    assert abs(binary_fail_f1(logits, target) - 1.0) < 1e-6


def test_mean_metrics_n():
    rows = [{"fail_f1": 1.0, "util_mae": 0.1, "delta_rmse": 0.2}]
    m = mean_metrics(rows)
    assert m["n"] == 1.0
    assert m["fail_f1"] == 1.0
