# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.mark.parametrize("model_type", ["qwen3_5_loop", "sensenova_loop"])
def test_loop_config_preserves_physical_layers_across_serialization(model_type):
    from vllm.transformers_utils.configs.qwen3_5_loop import (
        Qwen3_5LoopConfig,
        SenseNovaLoopConfig,
    )

    config_cls = (
        SenseNovaLoopConfig if model_type == "sensenova_loop" else Qwen3_5LoopConfig
    )
    config = config_cls(
        text_config={
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2,
        },
        loop_settings={"start": 1, "end": 3, "eval_count": 2},
    )
    restored = config_cls.from_dict(config.to_dict())
    assert restored.text_config.loop_physical_indices == [0, 1, 2, 1, 2, 3]
    assert restored.text_config.layer_types == [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ]
    assert restored.loop_physical_num_layers == 4


def test_loop_config_rejects_mhc_crossing_loop_boundary():
    from vllm.transformers_utils.configs.qwen3_5_loop import Qwen3_5LoopConfig

    with pytest.raises(ValueError, match="crossing a loop boundary"):
        Qwen3_5LoopConfig(
            text_config={"num_hidden_layers": 4},
            loop_settings={"start": 1, "end": 3, "eval_count": 2},
            mhc={
                "start": 0,
                "end": 2,
                "streams": 2,
                "iterations": 20,
                "fixed_identity": False,
            },
        )


def test_loop_mhc_routes_and_merges_residual_streams():
    import torch

    from vllm.model_executor.layers.qwen3_5_mhc import MHCRouter

    router = MHCRouter(2, {"streams": 2, "iterations": 20, "fixed_identity": False})
    with torch.no_grad():
        router.weight.zero_()
        router.bias.zero_()
    residual = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    branch, post, mix = router(residual)
    torch.testing.assert_close(branch, torch.tensor([[4.0, 6.0]]), rtol=0, atol=0)
    torch.testing.assert_close(mix, torch.full((1, 2, 2), 0.5), rtol=0, atol=0)
    result = router.merge(residual, branch, post, mix)
    torch.testing.assert_close(
        result, torch.tensor([[[8.0, 12.0], [8.0, 12.0]]]), rtol=0, atol=0
    )


def test_qwen3_5_models_use_platform_specific_implementation():
    from vllm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MTP
    from vllm.platforms import current_platform

    # Only the NVIDIA variant drops torch.compile; ROCm and other platforms
    # keep the previous torch.compile implementation in
    # vllm/model_executor/models/qwen3_5{,_mtp}.py.
    if current_platform.is_cuda():
        model_module = "vllm.models.qwen3_5.nvidia.model"
        mtp_module = "vllm.models.qwen3_5.nvidia.mtp"
    else:
        model_module = "vllm.model_executor.models.qwen3_5"
        mtp_module = "vllm.model_executor.models.qwen3_5_mtp"
    assert Qwen3_5ForCausalLM.__module__ == model_module
    assert Qwen3_5MTP.__module__ == mtp_module


def test_qwen3_5_nvidia_models_do_not_use_torch_compile_wrapper():
    # Import the NVIDIA modules directly so the assertion is independent of
    # the platform this test runs on.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.models.qwen3_5.nvidia import model as model_module
    from vllm.models.qwen3_5.nvidia import mtp as mtp_module

    assert not issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert not issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def test_qwen3_5_legacy_models_keep_torch_compile_wrapper():
    # The legacy implementation in vllm/model_executor/models/ serves ROCm and
    # other non-NVIDIA platforms; it must keep the torch.compile wrapper.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.model_executor.models import qwen3_5 as model_module
    from vllm.model_executor.models import qwen3_5_mtp as mtp_module

    assert issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert issubclass(
        mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper
    )
    assert issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("hidden_size", [257, 2048])
def test_loop_mhc_cuda_graph_replays_new_residuals(streams, expanded, hidden_size):
    import torch

    from vllm.model_executor.layers.qwen3_5_mhc import MHCRouter

    torch.manual_seed(0)
    router = MHCRouter(
        hidden_size, {"streams": streams, "iterations": 20, "fixed_identity": False}
    ).cuda()
    with torch.no_grad():
        router.weight.normal_(std=0.01)
        router.bias.normal_(std=0.01)
    storage = torch.randn(3, streams, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = storage[:, :1].expand_as(storage) if expanded else storage

    def route_and_merge():
        branch, post, mix = router(residual)
        return router.merge(residual, branch, post, mix)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            route_and_merge()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = route_and_merge()
    for _ in range(2):
        storage.normal_()
        flat = residual.flatten(-2).float()
        normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        logits = torch.nn.functional.linear(normalized, router.weight, router.bias)
        pre = logits[:, :streams].sigmoid().to(residual.dtype)
        reference_branch = (pre.unsqueeze(-2) @ residual).squeeze(-2)
        branch, _, _ = router(residual)
        torch.testing.assert_close(branch, reference_branch, rtol=0, atol=0)
        expected = route_and_merge()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 3, 128, 2048])
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_loop_mhc_log_sinkhorn_matches_reference(streams, tokens, scale):
    import torch

    from vllm.model_executor.kernels.mhc.triton import mhc_log_sinkhorn_triton

    torch.manual_seed(0)
    logits = torch.randn(tokens, 2 * streams + streams**2, device="cuda") * scale
    mix = logits[:, 2 * streams :].unflatten(-1, (streams, streams))
    expected = mix
    for _ in range(20):
        expected = expected - expected.logsumexp(-1, keepdim=True)
        expected = expected - expected.logsumexp(-2, keepdim=True)
    expected = expected.exp()
    actual = mhc_log_sinkhorn_triton(mix, 20)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
