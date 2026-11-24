#!/usr/bin/env bash
# Generate responses of a trained adapter AND the original model on all four benchmarks.
#
# Usage (from the repository root):
#   bash scripts/run_inference.sh <adapter_dir_or_hf_id> <run_name> [base_model]
# e.g.
#   bash scripts/run_inference.sh models/llava1.5-7b_text_unlearned text_unlearn_llava1.5
#
# Approximate runtimes (single GPU, from our runs) are given below.
set -e
ADAPTER=$(realpath -m "$1")
[ -d "$1" ] || ADAPTER="$1"          # not a local dir -> treat as a HF Hub id
RUN=$2
BASE=${3:-llava-hf/llava-1.5-7b-hf}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

run () {  # $1 = benchmark directory
    cd "$ROOT/inference/$1"
    mkdir -p logs
    python run_inference.py --peft_model_id "$ADAPTER" --base_model "$BASE" \
        --output_name "${RUN}_$1" --log_file "logs/${RUN}_$1.log"
    echo "$1 FINISHED!"
}

run JBPieces   # ~15 min  (needs restricted images, see data/README.md)
run PKU_RLHF   # ~30 min
run Figstep    # ~4 h
run VQA        # 520 VQA-v2 validation examples
