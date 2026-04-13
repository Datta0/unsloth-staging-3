"""Test grpo_compute_loss with various batch/sequence sizes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from test_helper_extract import get_grpo_compute_loss

grpo_compute_loss = get_grpo_compute_loss()

shapes = [(1, 1), (1, 4), (2, 4), (4, 8), (8, 32), (16, 64)]

def test_various_shapes():
    for B, S in shapes:
        new = torch.randn(B, S, requires_grad=True)
        result = grpo_compute_loss(
            torch.randn(B, S), new, torch.randn(B, S), None,
            torch.randint(0, 100, (B, S)), torch.ones(B, S),
            0.1, torch.randn(B), loss_type="grpo",
        )
        loss = result[0]
        assert loss.isfinite(), f"Non-finite loss for shape ({B},{S}): {loss}"
        loss.backward()
        assert new.grad is not None and new.grad.isfinite().all(), \
            f"Bad grads for shape ({B},{S})"
        print(f"  ({B},{S}): loss={loss.item():.4f}")
    print("PASS: all batch/seq shapes work")

def test_partial_mask():
    B, S = 4, 16
    mask = torch.ones(B, S)
    mask[:, S//2:] = 0
    new = torch.randn(B, S, requires_grad=True)
    result = grpo_compute_loss(
        torch.randn(B, S), new, torch.randn(B, S), None,
        torch.randint(0, 100, (B, S)), mask,
        0.1, torch.randn(B), loss_type="grpo",
    )
    loss = result[0]
    assert loss.isfinite(), f"Non-finite loss with partial mask: {loss}"
    print(f"PASS: partial mask works, loss={loss.item():.4f}")

if __name__ == "__main__":
    test_various_shapes()
    test_partial_mask()
