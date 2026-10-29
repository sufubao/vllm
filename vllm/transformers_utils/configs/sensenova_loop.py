# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any

from transformers import Qwen3_5Config


class SenseNovaLoopConfig(Qwen3_5Config):
    model_type = "sensenova_loop"

    def __init__(
        self,
        loop_settings: dict[str, Any] | None = None,
        mhc: dict[str, Any] | None = None,
        loop_physical_num_layers: int | None = None,
        loop_physical_layer_types: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.loop_settings = loop_settings
        self.mhc = mhc
        if loop_settings is None:
            return
        text = self.text_config
        num_layers = loop_physical_num_layers or text.num_hidden_layers
        start, end, count = (loop_settings[k] for k in ("start", "end", "eval_count"))
        if (
            any(type(v) is not int for v in (start, end, count))
            or not 0 <= start < end <= num_layers
            or count < 1
            or loop_settings.get("use_gate", False)
        ):
            raise ValueError(
                "Loop requires valid fixed bounds/count and use_gate=false"
            )
        indices = list(range(start)) + list(range(start, end)) * count
        indices += list(range(end, num_layers))
        layer_types = loop_physical_layer_types or text.layer_types
        if len(layer_types) != num_layers:
            raise ValueError("layer_types must describe the physical layers")
        if mhc is not None:
            mhc_start, mhc_end, streams, iterations = (
                mhc[k] for k in ("start", "end", "streams", "iterations")
            )
            bounds = (0, start, end, num_layers)
            if (
                any(
                    type(v) is not int
                    for v in (mhc_start, mhc_end, streams, iterations)
                )
                or streams not in (2, 4)
                or iterations < 1
                or type(mhc["fixed_identity"]) is not bool
                or not any(
                    left <= mhc_start < mhc_end <= right
                    for left, right in zip(bounds, bounds[1:])
                )
                or not math.isfinite(mhc.get("pre_scale", 1.0))
            ):
                raise ValueError(
                    "Invalid mHC settings or range crossing a loop boundary"
                )
        self.loop_physical_num_layers = num_layers
        self.loop_physical_layer_types = list(layer_types)
        text.loop_physical_indices = indices
        text.mhc = mhc
        text.num_hidden_layers = len(indices)
        text.layer_types = [layer_types[i] for i in indices]
