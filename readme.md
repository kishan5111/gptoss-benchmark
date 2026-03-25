# GPT-OSS 120B Inference Benchmarks

**H100 SXM vs RTX Pro 6000 Blackwell** — real numbers from two live GPU instances.

Part of the research behind the blog post:

**[Prefill, Decode, and the Memory Wall: A Deep Dive into LLM Inference](https://www.kishanvavdara.ai/blog/prefill-decode-and-the-memory-wall)**

*by Kishan Vavdara — Kaggle Competition Master, ML Engineer*

---

## What this repo contains

```
.
├── gptoss_120b_h100.sh       # Run on H100 SXM instance
├── gptoss_120b_a6000.sh      # Run on RTX Pro 6000 Blackwell instance
├── compare.py                # Merge both results → markdown table
├── logs/
│   ├── h100_sxm/
│   │   ├── system.log        # GPU / CPU / RAM info
│   │   ├── server.log        # vLLM server startup log
│   │   ├── benchmark.log     # timestamped run summary
│   │   ├── bench_*.log       # per-run vLLM output
│   │   ├── agentic_steps.csv # step-by-step agentic loop data
│   │   ├── vram_after_load.txt
│   │   └── vram_4k_ctx.txt
│   └── rtx6000_bw/
│       └── (same structure)
└── results/
    ├── h100_sxm.json         # H100 summary
    ├── rtx6000_bw.json       # RTX Pro 6000 summary
    └── comparison.md         # Side-by-side table
```

---

## Results

| Metric                                     | H100 SXM    | RTX Pro 6000 BW | Ratio (H100/RTX) |
| ------------------------------------------ | ----------- | --------------- | ---------------- |
| Prefill throughput (tok/s, prompt=512)     | 166.0       | 167.4           | 0.99x            |
| Decode tok/s (bs=1)                        | 166.1       | **172.1**       | **0.97x**        |
| Decode tok/s (bs=8)                        | 703.9       | 618.7           | 1.14x            |
| Decode tok/s (bs=32)                       | 1,343.2     | 1,088.2         | 1.23x            |
| TTFT mean (prompt=512)                     | 27.8 ms     | 27.5 ms         | ≈                |
| TTFT mean (prompt=4096)                    | 199.3 ms    | 220.8 ms        | 0.90x ✓          |
| Agentic loop latency (10-step ReAct, bs=1) | 8.04s       | 8.21s           | 0.98x ✓          |
| Peak VRAM used (bs=1, 4K ctx)              | 76.8 GB / 80 GB | 91.7 GB / 96 GB | —                |

### Key Insight

**Surprising result:** RTX Pro 6000 outperforms H100 at batch size 1 (3.6% faster decode), contradicting the simple 1.87× bandwidth prediction. The H100 advantage only materializes at higher batch sizes (14% faster at bs=8, 23% faster at bs=32).

**Why this matters:** For single-user applications, agentic workloads, or low-concurrency serving, the RTX Pro 6000 delivers comparable or better performance at ~1/3 the cost. The H100's bandwidth advantage shows up where it should — at high throughput with large batch sizes.

---

## Setup

```bash
# Both instances need vLLM installed
pip install vllm triton 
```

---

## Running

**On the H100 SXM instance:**

```bash
chmod +x gptoss_120b_h100.sh
./gptoss_120b_h100.sh
```

**On the RTX Pro 6000 Blackwell instance:**

```bash
chmod +x gptoss_120b_a6000.sh
./gptoss_120b_a6000.sh
```

Each script:

1. Logs full system info (GPU name, VRAM, driver, CPU, RAM)
2. Starts a vLLM server with MXFP4 native weights — no `--quantization` flag
3. Runs the full benchmark suite (prefill, decode bs=1/8/32, TTFT, agentic loop)
4. Records VRAM at model load and during 4K context run
5. Writes a summary JSON to `results/`

**After both runs complete** — copy both JSON files to the same location and run:

```bash
python compare.py
```

---

## Model

[`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b)

* MoE architecture, 5.1B active parameters per token, 117B total
* Ships with MXFP4 native weights (~58 GB) — fits on a single card on both GPUs
* vLLM detects MXFP4 automatically from the model config

---

## Hardware

|                  | H100 SXM     | RTX Pro 6000 Blackwell |
| ---------------- | ------------ | ---------------------- |
| Architecture     | Hopper GH100 | Blackwell GB202        |
| VRAM             | 80 GB HBM3   | 96 GB GDDR7            |
| Memory bandwidth | 3.35 TB/s    | 1.792 TB/s             |
| NVLink           | 900 GB/s     | None (PCIe Gen5)       |
| TDP              | 700 W        | 600 W                  |

---

## vLLM config (both instances)

```
--tensor-parallel-size 1
--max-model-len 8192
--gpu-memory-utilization 0.95
--enable-chunked-prefill
--kv-cache-dtype auto
```

**Configuration notes:**
- No `--quantization` flag — MXFP4 is the native checkpoint format
- `gpu-memory-utilization=0.95` for both GPUs (H100 needs higher utilization due to 80GB vs 96GB)
- PyTorch fragmentation fix enabled: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## Why this benchmark matters

Decode is memory-bound at small batch sizes. Arithmetic intensity at BF16 equals batch size:

```
AI = 2 × batch / dtype_bytes = batch   (at BF16)
```

H100 SXM ridge point ≈ 295. At bs=1, decode sits at AI=1 — deep in the memory-bound regime.

The bandwidth gap (3.35 vs 1.792 TB/s) **theoretically predicts ~1.87× H100 advantage** on decode.

**But the actual results show RTX Pro 6000 winning at bs=1**, proving that bandwidth theory alone is insufficient. Software maturity, memory controller efficiency, and dispatch overhead all matter at real-world batch sizes. This benchmark reveals the gap between theoretical predictions and measured performance.


Full explanation: **[kishanvavdara.ai/blog/prefill-decode-memory-wall](https://www.kishanvavdara.ai/blog/prefill-decode-and-the-memory-wall)**

---

## Author

**Kishan Vavdara** — Kaggle Competition Master (top ~500 globally), ML Engineer

[kishanvavdara.ai](https://www.kishanvavdara.ai/) · [LinkedIn](https://linkedin.com/in/kishanvavdara) · [Kaggle](https://kaggle.com/kishanvavdara)
