"""Test grpo_compute_loss when old_logps is None (on-policy, no importance sampling)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from test_helper_extract import get_grpo_compute_loss

grpo_compute_loss = get_grpo_compute_loss()

def test_old_none_grpo():
    B, S = 2, 8
    new = torch.randn(B, S, requires_grad=True)
    result = grpo_compute_loss(
        torch.randn(B, S), new, None, None,
        torch.randint(0, 100, (B, S)), torch.ones(B, S),
        0.1, torch.randn(B), loss_type="grpo",
    )
    loss = result[0]
    assert loss.isfinite(), f"Loss not finite with old=None: {loss}"
    loss.backward()
    assert new.grad is not None and new.grad.isfinite().all()
    print(f"PASS: old=None works, loss={loss.item():.4f}")

def test_old_none_bnpo():
    B, S = 2, 8
    new = torch.randn(B, S, requires_grad=True)
    result = grpo_compute_loss(
        torch.randn(B, S), new, None, None,
        torch.randint(0, 100, (B, S)), torch.ones(B, S),
        0.0, torch.randn(B), loss_type="bnpo",
    )
    loss = result[0]
    assert loss.isfinite()
    print(f"PASS: old=None bnpo works, loss={loss.item():.4f}")

def test_ref_none_beta_zero():
    B, S = 2, 8
    new = torch.randn(B, S, requires_grad=True)
    result = grpo_compute_loss(
        torch.zeros(B, S), new, None, None,
        torch.randint(0, 100, (B, S)), torch.ones(B, S),
        0.0, torch.randn(B), loss_type="grpo",
    )
    loss = result[0]
    assert loss.isfinite()
    print(f"PASS: beta=0 with zero ref works, loss={loss.item():.4f}")

if __name__ == "__main__":
    test_old_none_grpo()
    test_old_none_bnpo()
    test_ref_none_beta_zero()
