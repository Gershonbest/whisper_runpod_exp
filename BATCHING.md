# Micro-Batching Architecture

## Overview

The GPU worker uses a custom micro-batching system to convert individual Redis queue requests into efficient GPU batch workloads. Instead of processing one request at a time, the worker collects multiple requests into a batch, preprocesses them in parallel on CPU, then runs GPU inference sequentially with zero contention.

```
Requests → Redis Queue → Batch Collector → CPU Parallel Prep → GPU Sequential Inference
                              ↓
                    [collect up to N items
                     within timeout window]
```

## How It Works

### 1. Batch Collection (async, event loop)

The batch collector runs in a continuous loop:

```
BRPOP (block-wait for first item)
        ↓
Drain all queued items instantly (step 2)
        ↓
Wait up to BATCH_TIMEOUT for stragglers (step 3)
        ↓
Dispatch batch to processor
```

- **Step 1**: `BRPOP` blocks until at least one job arrives (up to `QUEUE_BRPOP_TIMEOUT` seconds)
- **Step 2**: Immediately drains everything already in the queue — no sleeping, no waiting
- **Step 3**: If the batch isn't full, polls for `BATCH_TIMEOUT` seconds with 10ms sleeps

The batch is capped at `MAX_BATCH_SIZE`. Collection stops at whichever limit is hit first (size or timeout).

### 2. CPU Preprocessing (parallel, ThreadPoolExecutor)

All items in the batch are preprocessed concurrently on CPU:

- Download audio from URL
- Re-encode with ffmpeg (fix corruption)
- Normalize volume
- Convert to waveform tensor

This uses `ThreadPoolExecutor(max_workers=4)` — bounded I/O parallelism that doesn't touch the GPU.

### 3. GPU Inference (sequential, single thread)

After all items are preprocessed, a single thread processes them one-by-one on the GPU inside `torch.inference_mode()`:

- Speaker diarization (if enabled)
- Whisper transcription
- Translation (if requested)

**No ThreadPoolExecutor for GPU work.** One thread owns the GPU for the entire batch. This avoids GPU contention which would actually slow things down.

### 4. GPU Semaphore

An `asyncio.Semaphore(MAX_CONCURRENCY)` gates batch execution. With `MAX_CONCURRENCY=1` (recommended for single GPU), only one batch can use the GPU at a time. The next batch's CPU preprocessing can overlap with the current batch's GPU work.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MAX_BATCH_SIZE` | `6` | Maximum items per batch |
| `BATCH_TIMEOUT` | `0.5` | Seconds to wait for additional items after first arrival |
| `MAX_CONCURRENCY` | `1` | Max batches running on GPU simultaneously |
| `QUEUE_BRPOP_TIMEOUT` | `5` | Seconds to block-wait for first item before retrying |

### Tuning Guide

**High traffic** (queue always has items):
```env
MAX_BATCH_SIZE=8
BATCH_TIMEOUT=0.05
```
Batches fill instantly from the queue. Short timeout avoids unnecessary waiting.

**Moderate traffic** (bursts of 3–6 requests):
```env
MAX_BATCH_SIZE=6
BATCH_TIMEOUT=0.5
```
Drains queued items immediately, waits briefly for stragglers.

**Low traffic** (occasional single requests):
```env
MAX_BATCH_SIZE=4
BATCH_TIMEOUT=0.1
```
Don't wait long — process what you have quickly.

### How the knobs interact

| Scenario | What happens |
|---|---|
| Queue has 10 items, `MAX_BATCH_SIZE=6` | Grabs 6 instantly, timeout never triggers |
| Queue has 2 items, `MAX_BATCH_SIZE=6` | Grabs 2 instantly, waits up to `BATCH_TIMEOUT` for 4 more |
| Queue has 0 items | `BRPOP` blocks until first item, then drains + waits |

**Rule of thumb**: `BATCH_TIMEOUT` only matters when the queue has fewer items than `MAX_BATCH_SIZE`. If your queue is usually full, set it very low.

## Performance Characteristics

### Why batching > concurrency

For Whisper on a single GPU:

| Approach | GPU Utilization | Throughput |
|---|---|---|
| Serial (1 at a time) | Low (idle between jobs) | Baseline |
| Concurrent threads (N threads → GPU) | Contention, thrashing | **Worse** than serial |
| Micro-batching (parallel CPU + sequential GPU) | High (back-to-back inference) | **Best** |

Multiple threads fighting for the GPU causes CUDA synchronization overhead. Sequential inference with zero idle time between items is faster.

### Pipeline overlap

```
Batch 1:  [CPU prep ████] [GPU inference ████████████]
Batch 2:                   [CPU prep ████] [waiting...] [GPU inference ████████████]
```

While batch 1 is on the GPU, batch 2's CPU preprocessing can run concurrently. The GPU semaphore ensures only one batch does inference at a time.

## Monitoring

Watch the logs for batch efficiency:

```
Batch collected: 6/6 items              ← full batch, good
Batch ready: size=6, acquiring GPU semaphore…
Micro-batch start: size=6
Micro-batch CPU prep done: 1.23s for 6 items
Batch item transcribed: language=EN duration=45.20s gpu_time=3.10s
Batch item transcribed: language=EN duration=30.50s gpu_time=2.40s
...
Micro-batch done: 6/6 succeeded in 18.50s (prep=1.23s gpu=17.27s)
```

If you consistently see `Batch collected: 1/6 items`, your `BATCH_TIMEOUT` is too short or traffic is too low to benefit from batching.

### GPU validation

Run while processing requests:
```bash
watch -n 0.5 nvidia-smi
```

- **Good**: GPU util 80–100% during batch processing, drops between batches
- **Bad**: GPU util spikes briefly per item with gaps — means items aren't flowing back-to-back
