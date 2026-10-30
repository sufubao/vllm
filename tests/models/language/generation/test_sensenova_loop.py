# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.transformers_utils.configs.sensenova_loop import SenseNovaLoopConfig

pytestmark = pytest.mark.hybrid_model


@pytest.fixture
def loop_checkpoint(tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM

    config = SenseNovaLoopConfig(
        architectures=["SenseNovaLoopForConditionalGeneration"],
        text_config={
            "vocab_size": 256,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "max_position_embeddings": 128,
            "eos_token_id": 255,
        },
    )
    torch.manual_seed(0)
    weights = Qwen3_5ForCausalLM(config.text_config).bfloat16().state_dict()
    for layer in (1, 2):
        for router in ("mhc_attn", "mhc_ffn"):
            weights[f"model.layers.{layer}.{router}.weight"] = torch.zeros(8, 512)
            weights[f"model.layers.{layer}.{router}.bias"] = torch.zeros(8)
    save_file(weights, tmp_path / "model.safetensors")
    config.loop_settings = {"start": 1, "end": 3, "eval_count": 2}
    config.mhc = {
        "start": 1,
        "end": 3,
        "streams": 2,
        "iterations": 20,
        "fixed_identity": False,
        "pre_scale": 1.0,
    }
    config.save_pretrained(tmp_path)
    return str(tmp_path)


def test_loop_overrides_reach_loaded_layers(loop_checkpoint, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=loop_checkpoint,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=32,
        max_num_seqs=2,
        gpu_memory_utilization=0.05,
        kv_cache_memory_bytes=16 * 1024**2,
        hf_overrides={
            "loop_settings": {"start": 1, "end": 3, "eval_count": 3},
            "mhc": {
                "start": 1,
                "end": 3,
                "streams": 2,
                "iterations": 20,
                "fixed_identity": False,
                "pre_scale": 0.5,
            },
        },
    )
    try:

        def inspect_layers(model):
            layers = model.model.layers
            return (
                [layer.physical_index for layer in layers],
                [layer.mhc_attn.pre_scale for layer in layers if layer.mhc_active],
            )

        assert llm.apply_model(inspect_layers) == [
            ([0, 1, 2, 1, 2, 1, 2, 3], [0.5] * 6)
        ]
        output = llm.generate(
            {"prompt_token_ids": [1, 2, 3]},
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        )
        assert len(output[0].outputs[0].token_ids) == 2
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.parametrize(
    "offload",
    [
        {"cpu_offload_gb": 1},
        {"offload_group_size": 2, "offload_num_in_group": 1},
    ],
)
def test_loop_rejects_cpu_offload_before_loading(loop_checkpoint, offload):
    with pytest.raises(NotImplementedError, match="SenseNova Loop.*CPU offload"):
        llm = LLM(
            model=loop_checkpoint,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=32,
            gpu_memory_utilization=0.05,
            kv_cache_memory_bytes=16 * 1024**2,
            **offload,
        )
        llm.llm_engine.engine_core.shutdown()
