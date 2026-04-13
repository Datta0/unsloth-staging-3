"""Test all loss_type variants work with post-PR arg order."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from test_helper_extract import get_grpo_compute_loss

grpo_compute_loss = get_grpo_compute_loss()
B, S = 4, 16

def make_args():
    return (
        torch.randn(B, S),
        torch.randn(B, S, requires_grad=True),
        torch.randn(B, S),
        None,
        torch.randint(0, 100, (B, S)),
        torch.ones(B, S),
        0.04,
        torch.randn(B),
    )

def test_grpo():
    loss = grpo_compute_loss(*make_args(), loss_type="grpo")[0]
    assert loss.isfinite(), f"grpo loss not finite: {loss}"
    print(f"PASS: grpo loss={loss.item():.4f}")

def test_bnpo():
    loss = grpo_compute_loss(*make_args(), loss_type="bnpo")[0]
    assert loss.isfinite(), f"bnpo loss not finite: {loss}"
    print(f"PASS: bnpo loss={loss.item():.4f}")

def test_dr_grpo():
    loss = grpo_compute_loss(*make_args(), loss_type="dr_grpo")[0]
    assert loss.isfinite(), f"dr_grpo loss not finite: {loss}"
    print(f"PASS: dr_grpo loss={loss.item():.4f}")

def test_dapo():
    loss = grpo_compute_loss(*make_args(), loss_type="dapo", num_items_in_batch=B, num_processes=1)[0]
    assert loss.isfinite(), f"dapo loss not finite: {loss}"
    print(f"PASS: dapo loss={loss.item():.4f}")

def test_cispo():
    loss = grpo_compute_loss(*make_args(), loss_type="cispo", num_items_in_batch=B, num_processes=1)[0]
    assert loss.isfinite(), f"cispo loss not finite: {loss}"
    print(f"PASS: cispo loss={loss.item():.4f}")

def test_unknown_raises():
    try:
        grpo_compute_loss(*make_args(), loss_type="INVALID")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown loss type" in str(e)
    print("PASS: unknown loss_type raises ValueError")

if __name__ == "__main__":
    test_grpo()
    test_bnpo()
    test_dr_grpo()
    test_dapo()
    test_cispo()
    test_unknown_raises()
