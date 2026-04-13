"""Test that kwargs are correctly forwarded and not confused with positional args."""
import torch

def grpo_compute_loss(ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages, **kwargs):
    return {
        "positionals": (ref, new, old, sampling_per_token_logps, input_ids, mask, beta, advantages),
        "kwargs": kwargs,
    }

def test_kwargs_not_contaminated():
    """Ensure sampling_per_token_logps doesn't end up in kwargs when passed positionally."""
    result = grpo_compute_loss(
        "ref", "new", "old", "sampling",
        "ids", "mask", 0.1, "adv",
        loss_type="grpo", epsilon_low=0.2,
    )
    assert "sampling_per_token_logps" not in result["kwargs"], \
        "sampling_per_token_logps leaked into kwargs!"
    assert result["positionals"][3] == "sampling"
    assert result["kwargs"]["loss_type"] == "grpo"
    assert result["kwargs"]["epsilon_low"] == 0.2
    print("PASS: kwargs clean, no sampling_per_token_logps leak")

def test_all_kwargs_received():
    """Verify all expected kwargs pass through."""
    all_kwargs = dict(
        loss_type="grpo", epsilon_low=0.1, epsilon_high=0.3,
        max_completion_length=4096, delta=None,
        importance_sampling_level="token",
        num_items_in_batch=8, current_gradient_accumulation_steps=2,
        num_processes=4, temperature=0.7,
        pixel_values=None, image_grid_thw=None,
    )
    result = grpo_compute_loss(
        "ref", "new", "old", None, "ids", "mask", 0.1, "adv",
        **all_kwargs,
    )
    for k, v in all_kwargs.items():
        assert k in result["kwargs"], f"Missing kwarg: {k}"
        assert result["kwargs"][k] == v, f"Wrong value for {k}: {result['kwargs'][k]} != {v}"
    print("PASS: all kwargs forwarded correctly")

def test_extra_kwargs_absorbed():
    """Verify unknown kwargs are silently absorbed by **kwargs."""
    result = grpo_compute_loss(
        "ref", "new", "old", None, "ids", "mask", 0.1, "adv",
        totally_unknown_param="hello",
    )
    assert result["kwargs"]["totally_unknown_param"] == "hello"
    print("PASS: extra kwargs absorbed by **kwargs")

if __name__ == "__main__":
    test_kwargs_not_contaminated()
    test_all_kwargs_received()
    test_extra_kwargs_absorbed()
