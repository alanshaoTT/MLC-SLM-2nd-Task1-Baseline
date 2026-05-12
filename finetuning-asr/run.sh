#!/usr/bin/env bash

STAGE=0
STOP_STAGE=4

RAW_DIR="./raw_data"
TRAIN_DIR="./train"

MODEL_PATH="./model/VibeVoice-ASR"
OUTPUT_DIR="./output"

GPUS="1,2,3"
NPROC=3

LANG="eng_american"
SCP_FILE="../dev/scp/${LANG}.scp"
LORA_PATH="./output/checkpoint-XXX"
INFER_OUTPUT="./infer_results/${LANG}.jsonl"

# Meeteval inputs: should be STM or SegLST format
REF_FILE="./meeteval_refs/${LANG}.json"
HYP_FILE="./meeteval_hyps/${LANG}.json"
EVAL_DIR="./eval_results/${LANG}"
COLLAR=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --stop-stage) STOP_STAGE="$2"; shift 2 ;;
    --raw-dir) RAW_DIR="$2"; shift 2 ;;
    --train-dir) TRAIN_DIR="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --lang) LANG="$2"; shift 2 ;;
    --scp-file) SCP_FILE="$2"; shift 2 ;;
    --lora-path) LORA_PATH="$2"; shift 2 ;;
    --infer-output) INFER_OUTPUT="$2"; shift 2 ;;
    --ref-file) REF_FILE="$2"; shift 2 ;;
    --hyp-file) HYP_FILE="$2"; shift 2 ;;
    --eval-dir) EVAL_DIR="$2"; shift 2 ;;
    --collar) COLLAR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p ./infer_results ./eval_results

run_stage() {
  local s=$1
  [[ "$STAGE" -le "$s" && "$STOP_STAGE" -ge "$s" ]]
}

echo "Stage range: ${STAGE} -> ${STOP_STAGE}"

# Stage 0: Extract raw data
if run_stage 0; then
  echo "========== Stage 0: Extract raw data =========="
  mkdir -p "${RAW_DIR}"

  for f in MLC-SLM_Workshop-Training_Set_*.zip; do
    unzip "$f" -d "${RAW_DIR}"
  done
fi

# Stage 1: Convert training data
if run_stage 1; then
  echo "========== Stage 1: Prepare training data =========="
  python data_prepare.py \
    --raw-dir "${RAW_DIR}" \
    --out-dir "${TRAIN_DIR}"
fi

# Stage 2: LoRA fine-tuning
if run_stage 2; then
  echo "========== Stage 2: LoRA fine-tuning =========="
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NPROC}" lora_finetune.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${TRAIN_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --bf16
fi

# Stage 3: Batch inference
if run_stage 3; then
  echo "========== Stage 3: Inference =========="
  python infer_lora_batch.py \
    --base_model "${MODEL_PATH}" \
    --lora_path "${LORA_PATH}" \
    --scp_file "${SCP_FILE}" \
    --output_file "${INFER_OUTPUT}" \
    --device cuda \
    --resume
fi

# Stage 4: Meeteval tcpWER / tcpCER
if run_stage 4; then
  echo "========== Stage 4: Meeteval tcpWER / tcpCER =========="
  mkdir -p "${EVAL_DIR}"
  
  # NOTE:
  # Before evaluation, Japanese, Korean, and Thai transcriptions
  # MUST be converted into character-level tokenization to compute tcpCER.
  # Other languages should keep standard word-level tokenization for tcpWER.
  
  meeteval-wer tcpwer \
    -r "${REF_FILE}" \
    -h "${HYP_FILE}" \
    --collar "${COLLAR}" \
    --average-out "${EVAL_DIR}/average.json" \
    --per-reco-out "${EVAL_DIR}/per_reco.json"

  echo "Evaluation results saved to: ${EVAL_DIR}"
fi

echo "Done."
