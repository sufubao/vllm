# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import nn


class MHCRouter(nn.Module):
    def __init__(self, hidden_size: int, settings: dict):
        super().__init__()
        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        count = 2 * self.streams
        if not self.fixed_identity:
            count += self.streams**2
        self.weight = nn.Parameter(
            torch.empty(count, self.streams * hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(count, dtype=torch.float32), requires_grad=False
        )

    def forward(self, residual: torch.Tensor):
        if residual.is_cuda:
            from vllm.model_executor.kernels.mhc.triton import (
                mhc_project_triton,
                mhc_rmsnorm_triton,
            )

            normalized = mhc_rmsnorm_triton(residual)
            if residual.shape[0] <= 32:
                logits = mhc_project_triton(normalized, self.weight, self.bias)
            else:
                logits = F.linear(normalized, self.weight, self.bias)
        else:
            flat = residual.flatten(-2).float()
            normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
            logits = F.linear(normalized, self.weight, self.bias)
        s = self.streams
        pre = logits[..., :s].sigmoid()
        post = 2 * logits[..., s : 2 * s].sigmoid()
        if self.fixed_identity:
            mix = torch.eye(s, device=residual.device, dtype=torch.float32)
        else:
            mix = logits[..., 2 * s :].unflatten(-1, (s, s))
            if residual.is_cuda:
                from vllm.model_executor.kernels.mhc.triton import (
                    mhc_log_sinkhorn_triton,
                )

                mix = mhc_log_sinkhorn_triton(mix, self.iterations)
            else:
                for _ in range(self.iterations):
                    mix = mix - mix.logsumexp(-1, keepdim=True)
                    mix = mix - mix.logsumexp(-2, keepdim=True)
                mix = mix.exp()
        if residual.is_cuda and residual.dtype == torch.bfloat16:
            from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton

            # The checkpoint rounds coefficients before stream reduction.
            branch = hc_collapse_triton(residual, pre.to(residual.dtype).float())
        else:
            branch = (pre.to(residual.dtype).unsqueeze(-2) @ residual).squeeze(-2)
        return branch * self.pre_scale, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        if residual.is_cuda and residual.dtype == torch.bfloat16:
            from vllm.model_executor.kernels.mhc.triton import mhc_merge_triton

            return mhc_merge_triton(residual, branch, post, mix)
        mixed = mix.to(residual.dtype) @ residual
        update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
        return mixed + update
