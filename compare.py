#!/usr/bin/env python3
"""
compare.py
──────────
After running both benchmark scripts, copy results/h100_sxm.json
and results/rtx6000_bw.json to the same directory, then run:

    python compare.py

Prints a markdown table ready to paste into the blog.
Also writes results/comparison.md
"""

import json
import pathlib
import sys

RESULTS = pathlib.Path("./results")

def load(gpu):
    p = RESULTS / f"{gpu}.json"
    if not p.exists():
        print(f"ERROR: {p} not found. Run the benchmark script on that instance first.")
        sys.exit(1)
    return json.load(open(p))

h = load("h100_sxm")
r = load("rtx6000_bw")

def ratio(hv, rv, lower_is_better=False):
    try:
        hf = float(str(hv).replace(" ms","").replace(" GB",""))
        rf = float(str(rv).replace(" ms","").replace(" GB",""))
        if hf == 0 or rf == 0:
            return "—"
        if lower_is_better:
            winner = "H100" if hf < rf else "RTX"
        else:
            winner = "H100" if hf > rf else "RTX"
        factor = max(hf, rf) / min(hf, rf)
        return f"{factor:.2f}x {winner}"
    except:
        return "—"

def fmt(v, suffix=""):
    return f"{v}{suffix}" if v is not None else "—"

rows = [
    ("Prefill throughput (tok/s, prompt=512)",
        fmt(h.get("prefill_tok_s")),
        fmt(r.get("prefill_tok_s")),
        False),
    ("Decode tok/s (bs=1)",
        fmt(h.get("decode_bs1_tok_s")),
        fmt(r.get("decode_bs1_tok_s")),
        False),
    ("Decode tok/s (bs=8)",
        fmt(h.get("decode_bs8_tok_s")),
        fmt(r.get("decode_bs8_tok_s")),
        False),
    ("Decode tok/s (bs=32)",
        fmt(h.get("decode_bs32_tok_s")),
        fmt(r.get("decode_bs32_tok_s")),
        False),
    ("TTFT mean (prompt=512)",
        fmt(h.get("ttft_p512_ms"), " ms"),
        fmt(r.get("ttft_p512_ms"), " ms"),
        True),
    ("TTFT mean (prompt=4096)",
        fmt(h.get("ttft_p4096_ms"), " ms"),
        fmt(r.get("ttft_p4096_ms"), " ms"),
        True),
    ("Agentic loop latency (10-step ReAct, bs=1)",
        fmt(h.get("agentic_10step_ms"), " ms"),
        fmt(r.get("agentic_10step_ms"), " ms"),
        True),
    ("Peak VRAM used (bs=1, 4K ctx)",
        fmt(h.get("vram_4k_ctx_bs1_gb"), " GB"),
        fmt(r.get("vram_4k_ctx_bs1_gb"), " GB"),
        True),
]

lines = []
lines.append("| Metric | H100 SXM | RTX Pro 6000 BW | Ratio |")
lines.append("|---|---|---|---|")
for label, hv, rv, lib in rows:
    rat = ratio(hv, rv, lib)
    lines.append(f"| {label} | {hv} | {rv} | {rat} |")

table = "\n".join(lines)
print(table)

# Write comparison.md
out = RESULTS / "comparison.md"
out.write_text(
    f"# GPT-OSS 120B — H100 SXM vs RTX Pro 6000 Blackwell\n\n"
    f"Model: `{h.get('model')}`  \n"
    f"Precision: MXFP4 native (no quantization flag)  \n"
    f"TP: {h.get('tp')}  \n\n"
    f"{table}\n\n"
    f"## Notes\n\n"
    f"- MXFP4 weight footprint ~58 GB — fits single card on both GPUs\n"
    f"- H100 SXM: 80 GB HBM3, 3.35 TB/s bandwidth, NVLink 900 GB/s\n"
    f"- RTX Pro 6000 BW: 96 GB GDDR7, 1.792 TB/s bandwidth, PCIe Gen5 only\n"
    f"- Agentic loop: 10 sequential calls, context grows from 1012 to 5512 tokens\n"
    f"- All runs: vLLM, --enable-chunked-prefill, --kv-cache-dtype auto\n"
)
print(f"\nSaved: {out}")