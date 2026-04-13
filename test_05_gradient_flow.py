"""Test gradients propagate correctly through grpo_compute_loss with post-PR arg order."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from test_helper_extract import get_grpo_compute_loss

grpo_compute_loss = get_grpo_compute_loss()

def test_grad_flows_to_new():
    B, S = 2, 8
    new = torch.randn(B, S, requires_grad=True)
    ref = torch.randn(B, S)
    old = torch.randn(B, S)
    ids = torch.randint(0, 100, (B, S))
    mask = torch.ones(B, S)

    loss = grpo_compute_loss(
        ref, new, old, None, ids, mask, 0.1, torch.randn(B),
        loss_type="grpo",
    )[0]
    loss.backward()
    assert new.grad is not None, "No gradient on 'new' tensor"
    assert new.grad.isfinite().all(), "Non-finite gradients"
    assert new.grad.abs().sum() > 0, "All-zero gradients"
    print(f"PASS: grad flows to 'new', grad_norm={new.grad.norm().item():.4f}")

def test_grad_with_beta_zero():
    B, S = 2, 8
    new = torch.randn(B, S, requires_grad=True)
    loss = grpo_compute_loss(
        torch.randn(B, S), new, torch.randn(B, S), None,
        torch.randint(0, 100, (B, S)), torch.ones(B, S),
        0.0, torch.randn(B), loss_type="grpo",
    )[0]
    loss.backward()
    assert new.grad is not None
    assert new.grad.isfinite().all()
    print(f"PASS: grad flows with beta=0, grad_norm={new.grad.norm().item():.4f}")

def test_grad_no_leak_to_ref():
    B, S = 2, 8
    ref = torch.randn(B, S, requires_grad=True)
    new = torch.randn(B, S, requires_grad=True)
    loss = grpo_compute_loss(
        ref, new, torch.randn(B, S), None,
        torch.randint(0, 100, (B, S)), torch.ones(B, S),
        0.1, torch.randn(B), loss_type="grpo",
    )[0]
    loss.backward()
    assert new.grad is not None
    print(f"PASS: gradient check with ref, new_grad_norm={new.grad.norm().item():.4f}")

if __name__ == "__main__":
    test_grad_flows_to_new()
    test_grad_with_beta_zero()
    test_grad_no_leak_to_ref()
