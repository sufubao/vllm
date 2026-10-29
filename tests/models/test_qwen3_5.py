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
