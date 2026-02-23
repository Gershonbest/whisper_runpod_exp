# Why Whisper Can't Do Parallel GPU Inference

## The Short Answer

Whisper's decoder is **autoregressive** — each output token depends on the previous one. The CTranslate2 runtime (used by faster-whisper) serializes GPU access internally. Sending multiple inference calls from multiple threads results in serial execution with extra overhead, not parallelism.

## Whisper's Inference Pipeline

```
Audio → Mel Spectrogram → Encoder → Decoder → Text Tokens
         (CPU, fast)       (GPU)     (GPU, slow, sequential)
```

| Stage | Time Share | Batchable? | Why |
|---|---|---|---|
| Mel extraction | ~5% | Yes (CPU) | Pure numpy, no GPU needed |
| Encoder | ~10-15% | Yes (GPU) | Single forward pass, accepts `[batch, n_mels, frames]` |
| Decoder | ~80-85% | No | Autoregressive — token N depends on token N-1 |

The decoder dominates runtime and **cannot be parallelized across different audio inputs** in a single forward pass.

## Why the Decoder Can't Be Batched

### Autoregressive generation

```
Step 1: [<start>]           → "Hello"
Step 2: [<start>, "Hello"]  → ","
Step 3: [<start>, "Hello", ","] → "how"
...
```

Each token is generated one at a time. The model must complete step N before computing step N+1. This is inherent to how Whisper (and all seq2seq models) generate text.

### Different audio = different decode lengths

If you have two audio files:
- Audio A: 5 seconds → 20 tokens
- Audio B: 60 seconds → 400 tokens

You can't batch their decode steps together because:
- They have completely different encoder outputs (different audio content)
- They produce different numbers of tokens
- Audio A would finish 20x earlier, wasting the padded batch slots

### No KV-cache sharing

In LLM serving (e.g., vLLM, TGI), you can batch multiple prompts because they share the same model weights and decode vocabulary. The scheduler interleaves their decode steps.

Whisper is different — each audio input produces a unique encoder output that the decoder cross-attends to. There's no meaningful way to share computation across different audio streams during decoding.

## CTranslate2's Internal Serialization

faster-whisper uses CTranslate2 as its inference backend. CTranslate2:

1. **Holds an internal mutex** on the model during generation
2. If two threads call `model.generate()` simultaneously, the second **blocks** until the first completes
3. You get serial execution + thread synchronization overhead = **slower than single-threaded**

```python
# This does NOT run in parallel — CT2 serializes internally
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(model.transcribe, audio) for audio in batch]
    # Thread 2 blocks while thread 1 runs. Thread 3 blocks while thread 2 runs.
    # Total time ≈ sum of all individual times + thread overhead
```

## What About NVIDIA Triton?

Triton Inference Server achieves parallel Whisper inference through a fundamentally different approach:

| Feature | What it does | Why it helps |
|---|---|---|
| **TensorRT backend** | Compiles Whisper into optimized CUDA kernels | Lower per-token latency |
| **Multiple model instances** | Loads N copies of the model on one GPU | True parallel execution (N × memory cost) |
| **Sequence batcher** | Manages decode state across requests, interleaves CUDA launches | Amortizes GPU scheduling overhead |
| **Dynamic batching** | Pads inputs to same length, runs batched forward passes | Encoder batching + padded decoder batching |

The key difference: Triton manages the decode loop itself and can schedule CUDA kernels from multiple streams. CTranslate2 owns the decode loop and doesn't expose this level of control.

### What it would take to switch

| Requirement | Effort |
|---|---|
| Convert Whisper to TensorRT/ONNX engine | Medium — tooling exists |
| Implement sequence batching for decoder | High — custom scheduling logic |
| Manage per-request KV-caches | High — memory management |
| Replace faster-whisper entirely | High — rewrite inference pipeline |

This is the approach taken by [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [WhisperX with batched pipeline](https://github.com/m-bain/whisperX). It's a significant engineering effort for marginal gains unless you're processing thousands of requests per minute.

## Our Architecture: The Practical Optimum

Given CTranslate2's constraints, the micro-batching strategy we use is the best possible approach:

```
┌─────────────────────────────────────────────────┐
│ Batch of 6 requests from Redis                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Phase 1: CPU Parallel (ThreadPoolExecutor)     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐           │
│  │Down- │ │Down- │ │Down- │ │Down- │  ...       │
│  │load  │ │load  │ │load  │ │load  │            │
│  │+Prep │ │+Prep │ │+Prep │ │+Prep │            │
│  └──────┘ └──────┘ └──────┘ └──────┘            │
│          All run simultaneously                  │
│                                                 │
│  Phase 2: GPU Sequential (single thread)        │
│  ┌──────────────────────────────────────────┐   │
│  │ torch.inference_mode()                    │   │
│  │ item1 → transcribe → done                │   │
│  │ item2 → transcribe → done                │   │
│  │ item3 → transcribe → done                │   │
│  │ ...                                       │   │
│  │ No idle gaps. GPU stays busy.             │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Where the real speedup comes from

The GPU is never waiting for:
- Audio downloads (1–5 seconds each)
- ffmpeg re-encoding (0.5–2 seconds each)
- Volume normalization, resampling

All of that happens in parallel **before** the GPU starts. Without batching, each request pays these costs serially:

```
Without batching (6 requests):
  [download₁ 2s][gpu₁ 5s][download₂ 2s][gpu₂ 5s]...[download₆ 2s][gpu₆ 5s]
  Total: ~42s

With micro-batching (6 requests):
  [download all ≈2s][gpu₁ 5s][gpu₂ 5s]...[gpu₆ 5s]
  Total: ~32s
  Savings: ~10s (24% faster)
```

The savings scale with how long preprocessing takes relative to GPU time.

## Summary

| Parallel inference method | Works with faster-whisper? | Why / why not |
|---|---|---|
| Multiple threads → same model | No | CT2 mutex serializes access |
| Multiple model copies on 1 GPU | Possible but wasteful | 2× VRAM, marginal throughput gain |
| TensorRT + sequence batcher | Yes, but requires full rewrite | Different inference backend entirely |
| **Micro-batching (our approach)** | **Yes** | **Parallel CPU + sequential GPU = zero idle time** |
