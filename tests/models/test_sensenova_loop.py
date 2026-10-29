# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.mark.parametrize("load_from_disk", [False, True])
def test_loop_config_preserves_physical_layers_across_serialization(
    tmp_path, load_from_disk
):
    from vllm.transformers_utils.configs.sensenova_loop import SenseNovaLoopConfig

    config = SenseNovaLoopConfig(
        architectures=["SenseNovaLoopForConditionalGeneration"],
        text_config={
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2,
        },
        loop_settings={"start": 1, "end": 3, "eval_count": 2},
    )
    if load_from_disk:
        from vllm.transformers_utils.config import get_config

        config.save_pretrained(tmp_path)
        restored = get_config(tmp_path, trust_remote_code=False)
    else:
        restored = SenseNovaLoopConfig.from_dict(config.to_dict())
    assert restored.model_type == "sensenova_loop"
    assert restored.architectures == ["SenseNovaLoopForConditionalGeneration"]
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
    from vllm.transformers_utils.configs.sensenova_loop import SenseNovaLoopConfig

    with pytest.raises(ValueError, match="crossing a loop boundary"):
        SenseNovaLoopConfig(
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

    from vllm.model_executor.layers.sensenova_loop_mhc import MHCRouter

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
