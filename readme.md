# GPT-OSS 120B Inference Benchmarks

**H100 SXM vs RTX Pro 6000 Blackwell** — real numbers from two live GPU instances.

Part of the research behind the blog post:

**[Prefill, Decode, and the Memory Wall: A Deep Dive into LLM Inference](https://www.kishanvavdara.ai/blog/prefill-decode-memory-wall)**

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

| Metric                                     | H100 SXM | RTX Pro 6000 BW | Ratio |
| ------------------------------------------ | -------- | --------------- | ----- |
| Prefill throughput (tok/s, prompt=512)     | —       | —              | —    |
| Decode tok/s (bs=1)                        | —       | —              | —    |
| Decode tok/s (bs=8)                        | —       | —              | —    |
| Decode tok/s (bs=32)                       | —       | —              | —    |
| TTFT mean (prompt=512)                     | —       | —              | —    |
| TTFT mean (prompt=4096)                    | —       | —              | —    |
| Agentic loop latency (10-step ReAct, bs=1) | —       | —              | —    |
| Peak VRAM used (bs=1, 4K ctx)              | —       | —              | —    |

*Results will be updated once both benchmark runs complete.*

*See [`results/comparison.md`](https://claude.ai/chat/results/comparison.md) for the filled table.*

---

## Setup

```bash
# Both instances need vLLM installed
pip install vllm

# HuggingFace login required (gated model)
huggingface-cli login
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
--gpu-memory-utilization 0.90
--enable-chunked-prefill
--kv-cache-dtype auto
--disable-log-requests
```

No `--quantization` flag — MXFP4 is the native checkpoint format.

---

## Why this benchmark matters

Decode is memory-bound at small batch sizes. Arithmetic intensity at BF16 equals batch size:

```
AI = 2 × batch / dtype_bytes = batch   (at BF16)
```

H100 SXM ridge point ≈ 295. At bs=1, decode sits at AI=1 — deep in the memory-bound regime.

The bandwidth gap (3.35 vs 1.792 TB/s) predicts ~1.87× H100 advantage on decode.

The actual numbers test that prediction against real MoE routing overhead.


Full explanation: **[kishanvavdara.ai/blog/prefill-decode-memory-wall](https://www.kishanvavdara.ai/blog/prefill-decode-memory-wall)**

---

## Author

**Kishan Vavdara** — Kaggle Competition Master (top ~500 globally), ML Engineer

[kishanvavdara.ai](https://www.kishanvavdara.ai/) · [LinkedIn](https://linkedin.com/in/kishanvavdara) · [Kaggle](https://kaggle.com/kishanvavdara)
