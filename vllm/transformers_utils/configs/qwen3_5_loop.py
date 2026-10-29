# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

from transformers import Qwen3_5Config


class Qwen3_5LoopConfig(Qwen3_5Config):
    model_type = "qwen3_5_loop"

    def __init__(
        self,
        loop_settings=None,
        mhc=None,
        loop_physical_num_layers=None,
        loop_physical_layer_types=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loop_settings = loop_settings
        self.mhc = mhc
        if loop_settings is None:
            return
        text = self.text_config
        n = loop_physical_num_layers or text.num_hidden_layers
        start, end, count = (loop_settings[k] for k in ("start", "end", "eval_count"))
        if (
            any(type(v) is not int for v in (start, end, count))
            or not 0 <= start < end <= n
            or count < 1
            or loop_settings.get("use_gate", False)
        ):
            raise ValueError(
                "Loop requires valid fixed bounds/count and use_gate=false"
            )
        indices = list(range(start)) + list(range(start, end)) * count
        indices += list(range(end, n))
        types = loop_physical_layer_types or text.layer_types
        if len(types) != n:
            raise ValueError("layer_types must describe the physical layers")
        if mhc is not None:
            a, b, streams, iterations = (
                mhc[k] for k in ("start", "end", "streams", "iterations")
            )
            bounds = (0, start, end, n)
            if (
                any(type(v) is not int for v in (a, b, streams, iterations))
                or streams not in (2, 4)
                or iterations < 1
                or type(mhc["fixed_identity"]) is not bool
                or not any(
                    left <= a < b <= right for left, right in zip(bounds, bounds[1:])
                )
                or not math.isfinite(mhc.get("pre_scale", 1.0))
            ):
                raise ValueError(
                    "Invalid mHC settings or range crossing a loop boundary"
                )
        self.loop_physical_num_layers = n
        self.loop_physical_layer_types = list(types)
        text.loop_physical_indices = indices
        text.mhc = mhc
        text.num_hidden_layers = len(indices)
        text.layer_types = [types[i] for i in indices]


class SenseNovaLoopConfig(Qwen3_5LoopConfig):
    model_type = "sensenova_loop"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
