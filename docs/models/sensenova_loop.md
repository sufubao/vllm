# SenseNova Loop with mHC

The `SenseNovaLoopForConditionalGeneration` implementation serves text from
`sensenova_loop` checkpoints. It uses a fixed
`loop_settings.eval_count`, shares repeated layer parameters, and keeps separate
attention and recurrent caches for every execution of a layer. Optional mHC
routers retain FP32 weights and use log-space Sinkhorn normalization.

The implementation requires unquantized weights. On NVIDIA GPUs, full decode
CUDA Graphs can run without Torch compilation; prefill executes eagerly.
It supports tensor parallelism, but not pipeline parallelism, vision, audio,
LoRA, speculative decoding, learned loop gates, or layer scaling.

```bash
vllm serve /path/to/checkpoint \
  --served-model-name sensenova_loop \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --host 127.0.0.1 --port 8001
```

Set the following top-level fields in the checkpoint's `config.json`, retaining
its existing text, loop and mHC settings:

```json
{
  "model_type": "sensenova_loop",
  "architectures": ["SenseNovaLoopForConditionalGeneration"]
}
```

Vision weights are ignored for text inference. At load time, repeated
layers temporarily allocate separate weights before tying parameters, so allow
additional memory during initialization. BF16 CUDA inference reuses vLLM's
`hc_collapse_triton` mHC stream reduction. Coefficients are rounded to BF16
before reduction, as in the checkpoint implementation. A fused Triton kernel
performs log-space Sinkhorn normalization. Residual merging preserves the
checkpoint's matrix orientation and BF16 intermediate rounding; the generic
`MHCPreOp` and `MHCPostOp` use different normalization and rounding conventions.

CUDA routing uses a fused FP32 RMS normalization and, for up to 32 tokens, a
split-K FP32 projection. Larger batches retain the cuBLAS projection. Two-stream
BF16 residual merging is fused into one kernel; wider streams retain the
original matrix multiplication to preserve its accumulation order. These paths
support CUDA Graph replay without reducing router precision.

Run `benchmarks/kernels/benchmark_sensenova_loop_mhc.py` with FlashInfer and CUPTI
installed to compare the routing and merging operations. The benchmark checks
projection against FP64 and merging against the original BF16 computation
before measuring CUDA Graph replay with cold L2 cache.

Use `--enforce-eager` instead of the compilation configuration for an eager
baseline. Add `--cudagraph-metrics` to inspect actual graph dispatch statistics.
