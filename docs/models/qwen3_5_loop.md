# Qwen3.5 Loop with mHC

The `Qwen3_5LoopForConditionalGeneration` implementation serves text from
`qwen3_5_loop` and `sensenova_loop` checkpoints. It uses a fixed
`loop_settings.eval_count`, shares repeated layer parameters, and keeps separate
attention and recurrent caches for every execution of a layer. Optional mHC
routers retain FP32 weights and use log-space Sinkhorn normalization.

The initial implementation requires eager execution and unquantized weights.
It supports tensor parallelism, but not pipeline parallelism, vision, audio,
LoRA, speculative decoding, learned loop gates, or layer scaling.

```bash
vllm serve /path/to/checkpoint \
  --served-model-name loop-test \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --host 127.0.0.1 --port 8001
```

Use the original checkpoint configuration; no architecture-name rewrite is
needed. Vision weights are ignored for text inference. At load time, repeated
layers temporarily allocate separate weights before tying parameters, so allow
additional memory during initialization. This correctness-oriented path uses
PyTorch mHC operations; it does not implement a fused mHC kernel.
