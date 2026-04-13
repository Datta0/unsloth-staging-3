"""Test return value shapes and types from grpo_compute_loss."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from test_helper_extract import get_grpo_compute_loss

grpo_compute_loss = get_grpo_compute_loss()

def test_return_7_tuple():
    B, S = 3, 10
    result = grpo_compute_loss(
        torch.randn(B, S), torch.randn(B, S, requires_grad=True),
        torch.randn(B, S), None,
        torch.randint(0, 100, (B, S)), torch.ones(B, S),
        0.1, torch.randn(B), loss_type="grpo",
    )
    assert len(result) == 7, f"Expected 7-tuple, got {len(result)}"
    loss, comp_len, mean_kl, delta, flat_is, coef_1, mask = result
    assert loss.dim() == 0, f"loss should be scalar, got dim={loss.dim()}"
    assert comp_len.dim() == 0
    assert mean_kl.dim() == 0
    assert coef_1.shape == (B, S), f"coef_1 shape wrong: {coef_1.shape}"
    assert mask.shape == (B, S), f"mask shape wrong: {mask.shape}"
    print("PASS: return is 7-tuple with correct shapes")

def test_return_shapes_sequence_level():
    B, S = 3, 10
    result = grpo_compute_loss(
        torch.randn(B, S), torch.randn(B, S, requires_grad=True),
        torch.randn(B, S), None,
        torch.randint(0, 100, (B, S)), torch.ones(B, S),
        0.1, torch.randn(B),
        loss_type="grpo", importance_sampling_level="sequence",
    )
    loss, comp_len, mean_kl, delta, flat_is, coef_1, mask = result
    assert loss.dim() == 0
    assert coef_1.shape == (B, 1), f"coef_1 shape for sequence: {coef_1.shape}"
    print("PASS: sequence-level importance sampling shapes correct")

def test_delta_empty_without_vllm():
    B, S = 2, 8
    result = grpo_compute_loss(
        torch.randn(B, S), torch.randn(B, S, requires_grad=True),
        torch.randn(B, S), torch.randn(B, S),
        torch.randint(0, 100, (B, S)), torch.ones(B, S),
        0.1, torch.randn(B), loss_type="grpo",
    )
    delta = result[3]
    flat_is = result[4]
    assert delta.numel() == 0, f"delta should be empty without vllm, got {delta.shape}"
    assert flat_is.numel() == 0
    print("PASS: delta/flat_is empty when use_vllm=False")

if __name__ == "__main__":
    test_return_7_tuple()
    test_return_shapes_sequence_level()
    test_delta_empty_without_vllm()
