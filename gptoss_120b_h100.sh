#!/bin/bash
# =============================================================================
# gptoss_120b_h100.sh
# Benchmark: openai/gpt-oss-120b (MXFP4) on H100 SXM
#
# Run on an H100 SXM instance:
#   chmod +x gptoss_120b_h100.sh
#   ./gptoss_120b_h100.sh
#
# Logs everything to ./logs/h100_sxm/
# Summary JSON written to ./results/h100_sxm.json
# =============================================================================

set -euo pipefail

GPU_NAME="h100_sxm"
MODEL="openai/gpt-oss-120b"
TP=1
HOST="127.0.0.1"
PORT=8000
MAX_LEN=8192
GPU_UTIL=0.92

LOGS_DIR="./logs/${GPU_NAME}"
RESULTS_DIR="./results"
SERVER_LOG="${LOGS_DIR}/server.log"
BENCH_LOG="${LOGS_DIR}/benchmark.log"
SYSTEM_LOG="${LOGS_DIR}/system.log"

mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

# Timestamp every log line
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$BENCH_LOG"; }

log "========================================================"
log " GPT-OSS 120B Benchmark — H100 SXM"
log " Model: $MODEL  |  Precision: MXFP4 native"
log "========================================================"

# ── SYSTEM INFO ───────────────────────────────────────────────────────────────
log "--- System info ---"
{
    echo "=== Date ==="
    date

    echo ""
    echo "=== Python / vLLM ==="
    python -c "import vllm; print('vLLM:', vllm.__version__)"
    python -c "import torch; print('PyTorch:', torch.__version__)"
    python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

    echo ""
    echo "=== nvidia-smi ==="
    nvidia-smi

    echo ""
    echo "=== GPU properties ==="
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap \
        --format=csv

    echo ""
    echo "=== CPU / RAM ==="
    lscpu | grep -E "Model name|CPU\(s\)|Thread|Socket"
    free -h

    echo ""
    echo "=== Disk ==="
    df -h /
} | tee "$SYSTEM_LOG"

log "System info saved: $SYSTEM_LOG"

# ── START vLLM SERVER ─────────────────────────────────────────────────────────
log "--- Starting vLLM server ---"
log "  No --quantization flag: MXFP4 detected automatically from model config"
log "  H100: gpu-util=0.95 for 80GB VRAM, PyTorch fragmentation fix enabled"

# Fix PyTorch memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-chunked-prefill \
    --kv-cache-dtype auto \
    --port "$PORT" \
    --host "$HOST" \
    >> "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
log "Server PID: $SERVER_PID  |  Log: $SERVER_LOG"

# Wait for ready
log "Waiting for server (first run downloads ~60 GB — may take 20-30 min)..."
for i in $(seq 1 720); do
    if curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
        log "Server ready after $((i * 5))s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: Server died. Check $SERVER_LOG"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    if (( i % 12 == 0 )); then
        log "Still waiting ($((i*5))s)... $(tail -1 "$SERVER_LOG" 2>/dev/null || echo '')"
    fi
    sleep 5
done

# VRAM after model load
log "--- VRAM after model load ---"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
    --format=csv,noheader | tee "${LOGS_DIR}/vram_after_load.txt"

# ── BENCHMARK HELPER ──────────────────────────────────────────────────────────
run_bench() {
    local INPUT=$1 OUTPUT=$2 PROMPTS=$3 CONCURRENCY=$4 TAG=$5
    local OUT="${RESULTS_DIR}/${GPU_NAME}_${TAG}.json"
    local RUN_LOG="${LOGS_DIR}/bench_${TAG}.log"

    log "  Running: $TAG (input=$INPUT out=$OUTPUT prompts=$PROMPTS conc=$CONCURRENCY)"

    vllm bench serve \
        --backend openai \
        --base-url "http://$HOST:$PORT" \
        --model "$MODEL" \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len "$INPUT" \
        --random-output-len "$OUTPUT" \
        --num-prompts "$PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-filename "$OUT" \
        2>&1 | tee "$RUN_LOG" | tail -10

    # Quick summary to bench log
    python3 -c "
import json
with open('$OUT') as f: d = json.load(f)
print(f'  tok/s={round(d.get(\"output_throughput\",d.get(\"tokens_per_second\",0)),1)}'
      f'  ttft={round(d.get(\"mean_ttft_ms\",0),1)}ms'
      f'  tpot={round(d.get(\"mean_tpot_ms\",0),1)}ms'
      f'  e2e={round(d.get(\"mean_e2el_ms\",d.get(\"mean_e2e_latency_ms\",0)),1)}ms')
" 2>/dev/null | tee -a "$BENCH_LOG" || true
}

# ── BENCHMARK SUITE ───────────────────────────────────────────────────────────
log "========================================================"
log " Running benchmark suite"
log "========================================================"

run_bench 512  256 100  1  "prefill_bs1_p512"
run_bench 512  256 100  1  "decode_bs1"
run_bench 512  256 100  8  "decode_bs8"
run_bench 512  256 50   32 "decode_bs32"
run_bench 512  32  100  1  "ttft_p512"
run_bench 4096 32  30   1  "ttft_p4096"
run_bench 4096 256 20   1  "vram_4k_bs1"

# VRAM during 4K run
log "--- VRAM during 4K context run ---"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
    --format=csv,noheader | tee "${LOGS_DIR}/vram_4k_ctx.txt"

# ── AGENTIC LOOP ──────────────────────────────────────────────────────────────
log "--- 10-step agentic loop (growing context, bs=1) ---"

AGENTIC_TOTAL=0
echo "step,input_len,e2e_ms,ttft_ms,tpot_ms" > "${LOGS_DIR}/agentic_steps.csv"

for step in $(seq 1 10); do
    INPUT=$((512 + step * 500))
    TAG="agentic_s${step}_ctx${INPUT}"
    run_bench "$INPUT" 128 10 1 "$TAG"

    E2E=$(python3 -c "
import json
with open('${RESULTS_DIR}/${GPU_NAME}_${TAG}.json') as f: d=json.load(f)
print(round(d.get('mean_e2el_ms',d.get('mean_e2e_latency_ms',0)),1))
" 2>/dev/null || echo 0)

    TTFT_V=$(python3 -c "
import json
with open('${RESULTS_DIR}/${GPU_NAME}_${TAG}.json') as f: d=json.load(f)
print(round(d.get('mean_ttft_ms',0),1))
" 2>/dev/null || echo 0)

    TPOT_V=$(python3 -c "
import json
with open('${RESULTS_DIR}/${GPU_NAME}_${TAG}.json') as f: d=json.load(f)
print(round(d.get('mean_tpot_ms',0),1))
" 2>/dev/null || echo 0)

    echo "${step},${INPUT},${E2E},${TTFT_V},${TPOT_V}" \
        >> "${LOGS_DIR}/agentic_steps.csv"

    AGENTIC_TOTAL=$(python3 -c "print(round($AGENTIC_TOTAL + $E2E, 1))")
    log "  step=$step ctx=$INPUT  e2e=${E2E}ms  running_total=${AGENTIC_TOTAL}ms"
done

log "10-step total: ${AGENTIC_TOTAL}ms = $(python3 -c "print(round($AGENTIC_TOTAL/1000,2))")s"
echo "$AGENTIC_TOTAL" > "${LOGS_DIR}/agentic_total_ms.txt"

# ── BUILD SUMMARY JSON ────────────────────────────────────────────────────────
log "--- Building summary JSON ---"

python3 - <<PYEOF
import json, os, pathlib

gpu = "$GPU_NAME"
rd  = pathlib.Path("$RESULTS_DIR")
ld  = pathlib.Path("$LOGS_DIR")

def load(tag):
    p = rd / f"{gpu}_{tag}.json"
    if not p.exists(): return {}
    return json.load(open(p))

def tok_s(tag):
    d = load(tag)
    v = d.get("output_throughput", d.get("tokens_per_second", 0))
    return round(v, 1) if v else None

def ttft_ms(tag):
    v = load(tag).get("mean_ttft_ms", 0)
    return round(v, 1) if v else None

def vram_gb(fname):
    p = ld / fname
    if not p.exists(): return None
    try:
        mib = float(p.read_text().strip().split("\n")[0].split(",")[0].split()[0])
        return round(mib / 1024, 1)
    except: return None

ag = float((ld / "agentic_total_ms.txt").read_text().strip())

summary = {
    "gpu":                   gpu,
    "model":                 "$MODEL",
    "precision":             "mxfp4_native",
    "tp":                    $TP,
    "max_model_len":         $MAX_LEN,
    "prefill_tok_s":         tok_s("prefill_bs1_p512"),
    "decode_bs1_tok_s":      tok_s("decode_bs1"),
    "decode_bs8_tok_s":      tok_s("decode_bs8"),
    "decode_bs32_tok_s":     tok_s("decode_bs32"),
    "ttft_p512_ms":          ttft_ms("ttft_p512"),
    "ttft_p4096_ms":         ttft_ms("ttft_p4096"),
    "agentic_10step_ms":     round(ag, 1),
    "vram_model_loaded_gb":  vram_gb("vram_after_load.txt"),
    "vram_4k_ctx_bs1_gb":    vram_gb("vram_4k_ctx.txt"),
}

out = f"$RESULTS_DIR/{gpu}.json"
json.dump(summary, open(out, "w"), indent=2)
print(json.dumps(summary, indent=2))
print(f"\nSaved: {out}")
PYEOF

# ── STOP SERVER ───────────────────────────────────────────────────────────────
log "--- Stopping server ---"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

log "========================================================"
log " Done. Outputs:"
log "   Logs:    $LOGS_DIR/"
log "   Results: $RESULTS_DIR/${GPU_NAME}.json"
log "========================================================"