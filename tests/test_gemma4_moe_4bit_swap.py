"""Unit tests for the Gemma-4 MoE per-expert Linear4bit swap (#5344).

End-to-end correctness on the real 26B-A4B checkpoint requires a GPU + the
checkpoint on disk, so this file restricts itself to fast CPU-only tests
that exercise the swap helper's shape contract, idempotence, and gating
behaviour. The full repro (resident VRAM 46 GB -> 14.27 GB, cosine sim 0.994
vs BF16) is documented in the PR description.
"""

import importlib
import os

import torch
import torch.nn as nn


def _stub_gemma4_module():
    """Construct a stub Gemma4TextExperts-like module without importing
    transformers' Gemma4Config (which would force a fresh transformers
    download in CPU-only CI)."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    except Exception:
        return None

    # The class init requires a config; build a tiny synthetic one and then
    # overwrite the fused weights with shapes small enough for CPU tests.
    class _StubConfig:
        num_experts = 4
        hidden_size = 16
        moe_intermediate_size = 8
        hidden_activation = "gelu_pytorch_tanh"

    module = Gemma4TextExperts.__new__(Gemma4TextExperts)
    nn.Module.__init__(module)
    module.num_experts = _StubConfig.num_experts
    module.hidden_dim = _StubConfig.hidden_size
    module.intermediate_dim = _StubConfig.moe_intermediate_size
    module.gate_up_proj = nn.Parameter(
        torch.randn(
            _StubConfig.num_experts,
            2 * _StubConfig.moe_intermediate_size,
            _StubConfig.hidden_size,
            dtype = torch.bfloat16,
        ),
        requires_grad = False,
    )
    module.down_proj = nn.Parameter(
        torch.randn(
            _StubConfig.num_experts,
            _StubConfig.hidden_size,
            _StubConfig.moe_intermediate_size,
            dtype = torch.bfloat16,
        ),
        requires_grad = False,
    )
    from transformers.activations import ACT2FN

    module.act_fn = ACT2FN[_StubConfig.hidden_activation]
    return module


def test_is_enabled_reads_env_var():
    from unsloth.models import gemma4_moe_4bit

    old = os.environ.pop("UNSLOTH_GEMMA4_MOE_4BIT", None)
    try:
        assert gemma4_moe_4bit.is_gemma4_moe_4bit_enabled() is False
        os.environ["UNSLOTH_GEMMA4_MOE_4BIT"] = "1"
        assert gemma4_moe_4bit.is_gemma4_moe_4bit_enabled() is True
        os.environ["UNSLOTH_GEMMA4_MOE_4BIT"] = "0"
        assert gemma4_moe_4bit.is_gemma4_moe_4bit_enabled() is False
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_GEMMA4_MOE_4BIT", None)
        else:
            os.environ["UNSLOTH_GEMMA4_MOE_4BIT"] = old


def test_swap_skips_models_without_gemma4_experts():
    from unsloth.models.gemma4_moe_4bit import (
        swap_gemma4_experts_to_per_expert_linear4bit,
    )

    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    assert swap_gemma4_experts_to_per_expert_linear4bit(model) == 0


def test_swap_skips_when_transformers_lacks_gemma4():
    """If transformers does not expose Gemma4TextExperts, the helper must
    return 0 without raising. `from X import Y` resolves via
    `builtins.__import__`, so we simulate absence via sys.modules sentinel:
    setting sys.modules[key] = None forces a fresh `from key import ...`
    to raise ImportError."""
    import sys

    import unsloth.models.gemma4_moe_4bit as g4m

    MODKEY = "transformers.models.gemma4.modeling_gemma4"
    _SENTINEL = object()
    original = sys.modules.get(MODKEY, _SENTINEL)
    sys.modules[MODKEY] = None
    try:
        model = nn.Sequential(nn.Linear(8, 8))
        assert g4m.swap_gemma4_experts_to_per_expert_linear4bit(model) == 0
    finally:
        if original is _SENTINEL:
            sys.modules.pop(MODKEY, None)
        else:
            sys.modules[MODKEY] = original


def test_swap_idempotent_on_stub_module_without_cuda():
    """On CPU we cannot exercise bnb (Linear4bit requires CUDA). Verify the
    helper at least returns 0 for the no-bnb-experts case without raising,
    and is idempotent across repeated calls."""
    from unsloth.models.gemma4_moe_4bit import (
        swap_gemma4_experts_to_per_expert_linear4bit,
    )

    if not torch.cuda.is_available():
        # CPU-only: bnb's Linear4bit init would fail. Validate the model-walk
        # path on an empty Sequential to confirm the helper is side-effect-free.
        model = nn.Sequential(nn.Linear(4, 4))
        assert swap_gemma4_experts_to_per_expert_linear4bit(model) == 0
        assert swap_gemma4_experts_to_per_expert_linear4bit(model) == 0
        return

    # GPU path: build the stub and run a real swap.
    module = _stub_gemma4_module()
    if module is None:
        return  # transformers without gemma4 module: nothing to test
    model = nn.Sequential(module.to("cuda"))
    n1 = swap_gemma4_experts_to_per_expert_linear4bit(model)
    n2 = swap_gemma4_experts_to_per_expert_linear4bit(model)
    assert n1 == 1
    assert n2 == 0  # idempotent: already-swapped modules are skipped
    assert hasattr(module, "gate_up_proj_4bit")
    assert hasattr(module, "down_proj_4bit")
    assert len(module.gate_up_proj_4bit) == module.num_experts
    assert len(module.down_proj_4bit) == module.num_experts


def test_per_expert_forward_matches_reference_on_cpu():
    """Exercise the patched _per_expert_forward on CPU by monkey-patching
    the bnb quantizer to return an exact-copy nn.Linear. Verifies the
    routing math, chunk(2) ordering, index_add_ accumulation, and
    top-k weighting match the upstream fused-weight reference forward."""
    import unsloth.models.gemma4_moe_4bit as g4m

    module = _stub_gemma4_module()
    if module is None:
        return  # transformers without gemma4 module: nothing to test

    def _fake_quantize(weight_2d, compute_dtype, quant_type = "nf4"):
        out_features, in_features = weight_2d.shape
        linear = nn.Linear(in_features, out_features, bias = False)
        linear.weight = nn.Parameter(
            weight_2d.detach().clone().to(torch.bfloat16),
            requires_grad = False,
        )
        return linear

    ref_gate_up = module.gate_up_proj.detach().clone()
    ref_down = module.down_proj.detach().clone()
    ref_act_fn = module.act_fn
    num_experts = module.num_experts
    hidden_dim = module.hidden_dim

    real_quant = g4m._quantize_one_expert_to_linear4bit
    g4m._quantize_one_expert_to_linear4bit = _fake_quantize
    try:
        assert g4m.swap_gemma4_experts_to_per_expert_linear4bit(module) == 1
    finally:
        g4m._quantize_one_expert_to_linear4bit = real_quant

    torch.manual_seed(0)
    n_tokens, top_k = 7, 2
    hidden = torch.randn(n_tokens, hidden_dim, dtype = torch.bfloat16)
    top_k_index = torch.tensor(
        [[0, 1], [2, 3], [0, 2], [1, 3], [0, 1], [2, 3], [1, 0]],
        dtype = torch.long,
    )
    top_k_weights = torch.full(
        (n_tokens, top_k), 0.5, dtype = torch.bfloat16,
    )

    got = module(hidden, top_k_index, top_k_weights)

    ref = torch.zeros_like(hidden)
    for e in range(num_experts):
        mask = top_k_index == e
        if not mask.any():
            continue
        tok_idx, kpos = torch.where(mask)
        cs = hidden[tok_idx]
        gate, up = torch.nn.functional.linear(cs, ref_gate_up[e]).chunk(
            2, dim = -1,
        )
        ch = ref_act_fn(gate) * up
        ch = torch.nn.functional.linear(ch, ref_down[e])
        ch = ch * top_k_weights[tok_idx, kpos, None]
        ref.index_add_(0, tok_idx, ch.to(ref.dtype))

    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol = 1e-2, rtol = 1e-2)


def test_partial_swap_marker_count_reflects_failure_mid_loop():
    """If the quantizer fails on the second Gemma4TextExperts module,
    the helper raises, but the first module is already marked swapped.
    A caller inspecting `_unsloth_gemma4_moe_4bit_swapped` can recover an
    accurate partial-swap count -- the basis for the partial-state warning
    in vision.py."""
    import unsloth.models.gemma4_moe_4bit as g4m

    m1 = _stub_gemma4_module()
    m2 = _stub_gemma4_module()
    if m1 is None or m2 is None:
        return

    model = nn.Sequential(m1, m2)

    state = {"count": 0}

    def _quant_fails_on_second_module(weight_2d, compute_dtype, quant_type = "nf4"):
        # Each module quantizes 2 * num_experts tensors (gate_up + down).
        # Fail on the first quantization call for the second module.
        per_module_calls = 2 * 4  # num_experts in stub
        state["count"] += 1
        if state["count"] > per_module_calls:
            raise RuntimeError("simulated quantization failure")
        out_features, in_features = weight_2d.shape
        linear = nn.Linear(in_features, out_features, bias = False)
        linear.weight = nn.Parameter(
            weight_2d.detach().clone().to(torch.bfloat16),
            requires_grad = False,
        )
        return linear

    real_quant = g4m._quantize_one_expert_to_linear4bit
    g4m._quantize_one_expert_to_linear4bit = _quant_fails_on_second_module
    try:
        raised = False
        try:
            g4m.swap_gemma4_experts_to_per_expert_linear4bit(model)
        except RuntimeError:
            raised = True
        assert raised
    finally:
        g4m._quantize_one_expert_to_linear4bit = real_quant

    partial = sum(
        1 for sub in model.modules()
        if getattr(sub, "_unsloth_gemma4_moe_4bit_swapped", False)
    )
    assert partial == 1
    assert getattr(m1, "_unsloth_gemma4_moe_4bit_swapped", False) is True
    assert getattr(m2, "_unsloth_gemma4_moe_4bit_swapped", False) is False


def test_swap_helper_forwards_quant_type_to_each_expert():
    """The swap helper must thread quant_type through to every per-expert
    Linear4bit so callers requesting fp4 (or any non-default) get
    consistent quantization across attention and expert layers.
    """
    import unsloth.models.gemma4_moe_4bit as g4m

    module = _stub_gemma4_module()
    if module is None:
        return

    seen_quant_types = []

    def _record_quant_type(weight_2d, compute_dtype, quant_type = "nf4"):
        seen_quant_types.append(quant_type)
        out_features, in_features = weight_2d.shape
        linear = nn.Linear(in_features, out_features, bias = False)
        linear.weight = nn.Parameter(
            weight_2d.detach().clone().to(torch.bfloat16),
            requires_grad = False,
        )
        return linear

    real_quant = g4m._quantize_one_expert_to_linear4bit
    g4m._quantize_one_expert_to_linear4bit = _record_quant_type
    try:
        n = g4m.swap_gemma4_experts_to_per_expert_linear4bit(
            module, quant_type = "fp4",
        )
    finally:
        g4m._quantize_one_expert_to_linear4bit = real_quant

    assert n == 1
    expected = 2 * module.num_experts
    assert len(seen_quant_types) == expected
    assert all(qt == "fp4" for qt in seen_quant_types)


def test_swap_helper_forwards_torch_dtype_compute_dtype_only():
    """The swap helper must receive a torch.dtype (not a string) so bnb's
    Linear4bit.forward call to `x.to(compute_dtype)` resolves the dtype
    rather than treating it as a device string."""
    import unsloth.models.gemma4_moe_4bit as g4m

    module = _stub_gemma4_module()
    if module is None:
        return

    seen_dtypes = []

    def _record_dtype(weight_2d, compute_dtype, quant_type = "nf4"):
        seen_dtypes.append(compute_dtype)
        out_features, in_features = weight_2d.shape
        linear = nn.Linear(in_features, out_features, bias = False)
        linear.weight = nn.Parameter(
            weight_2d.detach().clone().to(torch.bfloat16),
            requires_grad = False,
        )
        return linear

    real_quant = g4m._quantize_one_expert_to_linear4bit
    g4m._quantize_one_expert_to_linear4bit = _record_dtype
    try:
        g4m.swap_gemma4_experts_to_per_expert_linear4bit(
            module, compute_dtype = torch.float16,
        )
    finally:
        g4m._quantize_one_expert_to_linear4bit = real_quant

    assert all(isinstance(d, torch.dtype) for d in seen_dtypes)
    assert all(d == torch.float16 for d in seen_dtypes)
