"""Test grpo_compute_loss with real tensors using post-PR arg order."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from test_helper_extract import get_grpo_compute_loss

grpo_compute_loss = get_grpo_compute_loss()

def test_grpo_loss_basic():
    B, S = 2, 8
    ref = torch.randn(B, S, requires_grad=False)
    new = torch.randn(B, S, requires_grad=True)
    old = torch.randn(B, S, requires_grad=False)
    sampling = torch.randn(B, S)
    ids = torch.randint(0, 1000, (B, S))
    mask = torch.ones(B, S)
    beta = 0.1
    adv = torch.randn(B)

    result = grpo_compute_loss(
        ref, new, old, sampling, ids, mask, beta, adv,
        loss_type="grpo", epsilon_low=0.2, epsilon_high=0.2,
    )
    loss, comp_len, mean_kl, delta, flat_is, coef_1, out_mask = result
    assert loss.isfinite(), f"Loss not finite: {loss}"
    assert comp_len.isfinite()
    assert mean_kl.isfinite()
    print(f"PASS: loss={loss.item():.4f}, comp_len={comp_len.item():.1f}, kl={mean_kl.item():.4f}")

def test_grpo_loss_none_sampling():
    B, S = 2, 8
    ref = torch.randn(B, S)
    new = torch.randn(B, S, requires_grad=True)
    old = torch.randn(B, S)
    ids = torch.randint(0, 1000, (B, S))
    mask = torch.ones(B, S)

    result = grpo_compute_loss(
        ref, new, old, None, ids, mask, 0.1, torch.randn(B),
        loss_type="grpo",
    )
    loss = result[0]
    assert loss.isfinite(), f"Loss not finite with sampling=None: {loss}"
    print(f"PASS: sampling=None works, loss={loss.item():.4f}")

if __name__ == "__main__":
    test_grpo_loss_basic()
    test_grpo_loss_none_sampling()
