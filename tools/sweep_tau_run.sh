#!/bin/bash
#
# One-shot tau-screening launcher: trains for 5K steps at a given (plan_type, tau)
# under the screening yaml, then computes FID/IS/KID against CIFAR-10 train.
#
# Usage:
#   bash tools/sweep_tau_run.sh --tau 0.05 --plan sinkhorn --gpus 4 --port 29600
#
# Optional flags:
#   --bf16 / --fp32     mixed precision toggle (default: bf16)
#   --batch <N>         override num_real_samples=num_fake_samples=N (default: keeps yaml)
#   --T <N>             sinkhorn_iters (default: 20)
#   --runs-root <DIR>   parent dir for runs/ (default: ./runs)
#   --no-fid            skip FID computation at the end
#
# Output:
#   runs/<runs-root>/screen_<plan>_tau<TAU>_b<B>_<precision>/
#       config.yaml  output.log  ckpt/  samples/  fid_curve.json (1 row)
#   plus an aggregated row appended to runs/sweep_results.csv

set -euo pipefail

# ----- defaults -----
TAU=""
PLAN=""
GPUS=4
PORT=29600
PRECISION="bf16"
BATCH=""
T=20
RUNS_ROOT="$PWD/runs"
DO_FID=1

# ----- parse args -----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tau)        TAU="$2"; shift 2 ;;
        --plan)       PLAN="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --bf16)       PRECISION="bf16"; shift ;;
        --fp32)       PRECISION="fp32"; shift ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --T)          T="$2"; shift 2 ;;
        --runs-root)  RUNS_ROOT="$2"; shift 2 ;;
        --no-fid)     DO_FID=0; shift ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$TAU" || -z "$PLAN" ]]; then
    echo "ERROR: --tau and --plan are required"; sed -n '2,18p' "$0"; exit 1
fi
if [[ "$PLAN" != "sinkhorn" && "$PLAN" != "two-sided" && "$PLAN" != "one-sided" ]]; then
    echo "ERROR: --plan must be sinkhorn / two-sided / one-sided (got $PLAN)"; exit 1
fi

# ----- compose run name + dirs -----
TAU_TAG=$(echo "$TAU" | sed 's/[][.,]/p/g')           # 0.05 -> 0p05
B_TAG=$([[ -n "$BATCH" ]] && echo "$BATCH" || echo "2048")
RUN_NAME="screen_${PLAN}_tau${TAU_TAG}_b${B_TAG}_${PRECISION}"
EXP_DIR="$RUNS_ROOT/$RUN_NAME"

mkdir -p "$RUNS_ROOT"
if [[ -d "$EXP_DIR" ]]; then
    echo "WARN: $EXP_DIR already exists; appending .$(date +%s)"
    EXP_DIR="${EXP_DIR}.$(date +%s)"
fi

echo "=== sweep_tau_run.sh ==="
echo "  plan        : $PLAN"
echo "  tau (eps)   : $TAU"
echo "  T (sinkhorn): $T"
echo "  GPUs        : $GPUS  (master_port $PORT)"
echo "  precision   : $PRECISION"
echo "  batch       : $B_TAG"
echo "  exp dir     : $EXP_DIR"
echo

# ----- compose the torchrun command -----
SET_FLAGS=(
    --set "drifting.plan_type=$PLAN"
    --set "drifting.eps=$TAU"
    --set "drifting.sinkhorn_iters=$T"
)
if [[ -n "$BATCH" ]]; then
    SET_FLAGS+=( --set "train.num_real_samples=$BATCH" --set "train.num_fake_samples=$BATCH" )
fi

BF16_FLAG=""
[[ "$PRECISION" == "bf16" ]] && BF16_FLAG="--bf16"

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc-per-node "$GPUS" --master_port "$PORT" train_unc.py \
    -c configs/cifar10-unc-split-screening.yaml \
    -e "$EXP_DIR" \
    $BF16_FLAG \
    "${SET_FLAGS[@]}"

echo
echo "=== training done; computing FID/IS/KID ==="

# ----- post-train FID -----
if [[ "$DO_FID" -eq 1 ]]; then
    python tools/fid_watch.py \
        --exp-dir "$EXP_DIR" \
        --config configs/cifar10-unc-split-screening.yaml \
        --num-samples 10000 --bspp 64 --once
else
    echo "(skipped, --no-fid)"
fi

# ----- aggregate row -----
SWEEP_CSV="$RUNS_ROOT/sweep_results.csv"
if [[ ! -f "$SWEEP_CSV" ]]; then
    echo "run_name,plan,tau,T,batch,precision,steps,fid,is_mean,kid_mean,exp_dir" > "$SWEEP_CSV"
fi
python - <<EOF
import json, os, sys
exp_dir = "$EXP_DIR"
run_name = "$RUN_NAME"
plan = "$PLAN"
tau = "$TAU"
T = "$T"
batch = "$B_TAG"
precision = "$PRECISION"
steps = 5000
fid_json = os.path.join(exp_dir, "fid_curve.json")
fid = is_mean = kid_mean = ""
if os.path.exists(fid_json):
    rows = json.load(open(fid_json))
    if rows:
        r = sorted(rows, key=lambda x: x["step"])[-1]
        fid = f"{r.get('fid', '') :.4f}" if r.get("fid") is not None else ""
        is_mean = f"{r.get('is_mean', '') :.4f}" if r.get("is_mean") is not None else ""
        kid_mean = f"{r.get('kid_mean', '') :.6f}" if r.get("kid_mean") is not None else ""
with open("$SWEEP_CSV", "a") as f:
    f.write(f"{run_name},{plan},{tau},{T},{batch},{precision},{steps},{fid},{is_mean},{kid_mean},{exp_dir}\n")
print(f"row appended -> $SWEEP_CSV")
print(f"  fid={fid}  is={is_mean}  kid={kid_mean}")
EOF

echo
echo "=== sweep_tau_run.sh DONE ==="
echo "  exp dir       : $EXP_DIR"
echo "  sweep summary : $SWEEP_CSV"
