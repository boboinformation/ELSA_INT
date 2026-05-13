#!/usr/bin/env bash
# FP_Base — PyTorch backend, fp32 precision (no Triton kernel).
#
# ---- 資料與 checkpoint 路徑 ----------------------------------
#   ImageNet  : /raid/ilsvrc  (train/ + val/)
#
#   sic_int checkpoints (model_best):
#     /raid/jess/SICNet/out_sic_small_int/20250527-172928-sic_small_patch4_256_int_win-256/model_best.pth.tar
#
#   elsa_swin 初始化 (SwinV2 pretrained):
#     /raid/jess/SICNet/swinv2_small_patch4_window8_256.pth   (elsa_small_*)
#     /raid/jess/SICNet/swinv2_tiny_patch4_window8_256.pth    (elsa_tiny_*)
#
# ---- 常用範例 ------------------------------------------------
#   ※ 路徑寫法依目前所在目錄而異：
#     從 ELSA_INT/ 根目錄：bash script/FP_Base/run_fp_base.sh
#     從 script/FP_Base/ 目錄：bash run_fp_base.sh
#
#   bench (預設):
#     bash run_fp_base.sh
#
#   validate sic_int:
#     RUN_TASK=validate RUN_FAMILY=sic_int \
#     RUN_CHECKPOINT=/raid/jess/SICNet/out_sic_small_int/20250527-172928-sic_small_patch4_256_int_win-256/model_best.pth.tar \
#     bash run_fp_base.sh
#
#   validate elsa_swin (無 checkpoint，從預訓練 HF 抓):
#     RUN_TASK=validate bash run_fp_base.sh
#
#   train elsa_swin from scratch:
#     RUN_TASK=train bash run_fp_base.sh
#
#   train elsa_vit from scratch:
#     RUN_TASK=train RUN_FAMILY=elsa_vit bash run_fp_base.sh
#
#   bench elsa_vit (img 224):
#     RUN_FAMILY=elsa_vit bash run_fp_base.sh
#
#   train sic_int from scratch:
#     RUN_TASK=train RUN_FAMILY=sic_int bash run_fp_base.sh
#
#   dry run:
#     RUN_DRY_RUN=1 bash run_fp_base.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="/home/pojen/miniconda3/etc/profile.d/conda.sh"

# ---- FP_Base 固定設定 ----------------------------------------
RUN_BACKEND="pytorch"
RUN_DTYPE="fp32"

# ---- 可覆寫的預設值 ------------------------------------------
RUN_TASK="${RUN_TASK:-bench}"
RUN_FAMILY="${RUN_FAMILY:-elsa_swin}"      # elsa_swin | elsa_vit | sic_int
RUN_MODEL="${RUN_MODEL:-}"
RUN_CHECKPOINT="${RUN_CHECKPOINT:-}"
RUN_INITIAL_CHECKPOINT="${RUN_INITIAL_CHECKPOINT:-}"
RUN_DEVICE="${RUN_DEVICE:-cuda:0}"
if [[ "${RUN_FAMILY}" == "elsa_vit" ]]; then
  RUN_IMG_SIZE="${RUN_IMG_SIZE:-224}"
else
  RUN_IMG_SIZE="${RUN_IMG_SIZE:-256}"
fi
RUN_BATCH="${RUN_BATCH:-1}"
RUN_VAL_BATCH="${RUN_VAL_BATCH:-64}"
RUN_TRAIN_BATCH="${RUN_TRAIN_BATCH:-64}"
RUN_WARMUP="${RUN_WARMUP:-5}"
RUN_TRIALS="${RUN_TRIALS:-20}"
RUN_WORKERS="${RUN_WORKERS:-4}"
RUN_DATA_DIR="${RUN_DATA_DIR:-/home/pojen/project/ELSA/ViT/ViT-pytorch/data/ILSVRC2012}"
RUN_SPLIT="${RUN_SPLIT:-val}"
RUN_EPOCHS="${RUN_EPOCHS:-300}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-/raid/pojen/ELSA/ELSA_INT/output/train}"
RUN_DRY_RUN="${RUN_DRY_RUN:-0}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT}/logs/FP_Base}"
RUN_OUT="${RUN_OUT:-${ROOT}/results/FP_Base/fp_base_${RUN_FAMILY}_${RUN_DTYPE}.csv}"

# ---- 設定預設 model -----------------------------------------
if [[ -z "${RUN_MODEL}" ]]; then
  if [[ "${RUN_FAMILY}" == "sic_int" ]]; then
    RUN_MODEL="sic_small_patch4_256_int_win"
  elif [[ "${RUN_FAMILY}" == "elsa_vit" ]]; then
    RUN_MODEL="elsa_small_patch16_224"
  else
    RUN_MODEL="elsa_small_window8_256"
  fi
fi

# ---- conda 環境 ---------------------------------------------
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERR] conda init script not found: ${CONDA_SH}" >&2; exit 1
fi
set +u; source "${CONDA_SH}"; conda activate sicnet; set -u
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# ---- 防止重複執行 (train 才鎖) --------------------------------
LOCKFILE="/tmp/fp_base_${RUN_FAMILY}_${RUN_MODEL}.lock"
if [[ "${RUN_TASK}" == "train" && "${RUN_DRY_RUN}" != "1" ]]; then
  if [[ -f "${LOCKFILE}" ]]; then
    pid="$(cat "${LOCKFILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[ERR] 訓練已在執行中 (PID=${pid})，請勿重複啟動。" >&2
      echo "[ERR] 若要強制重跑，先執行：rm ${LOCKFILE}" >&2
      exit 1
    fi
  fi
  echo "$$" > "${LOCKFILE}"
  trap "rm -f '${LOCKFILE}'" EXIT
fi

# ---- checkpoint helper --------------------------------------
_checkpoint_args() {
  local var="$1"
  if [[ -n "${var}" && "${var}" != "none" ]]; then
    if [[ -f "${var}" ]]; then
      echo "--checkpoint ${var}"
    else
      echo "[WARN] checkpoint not found, skipping: ${var}" >&2
    fi
  fi
}

# ---- 組合指令 -----------------------------------------------
_model_kwargs_for_family() {
  [[ "${RUN_FAMILY}" == "elsa_swin" ]] && echo "--model-kwargs elsa_backend=${RUN_BACKEND}"
}

if [[ "${RUN_TASK}" == "bench" ]]; then
  bench_model_kwargs=()
  if [[ "${RUN_FAMILY}" == "elsa_swin" ]]; then
    bench_model_kwargs=(--model-kwargs "elsa_backend=${RUN_BACKEND}")
  fi
  cmd=(
    python "${ROOT}/models/benchmark.py"
    --model    "${RUN_MODEL}"
    --bench    inference
    --amp-dtype "${RUN_DTYPE}"
    --batch-size "${RUN_BATCH}"
    --img-size "${RUN_IMG_SIZE}"
    --num-warm-iter "${RUN_WARMUP}"
    --num-bench-iter "${RUN_TRIALS}"
    --device   "${RUN_DEVICE}"
    --results-file "${RUN_OUT}"
    "${bench_model_kwargs[@]}"
  )

elif [[ "${RUN_TASK}" == "validate" ]]; then
  checkpoint_args=()
  if [[ -n "${RUN_CHECKPOINT}" && "${RUN_CHECKPOINT}" != "none" ]]; then
    [[ -f "${RUN_CHECKPOINT}" ]] && checkpoint_args=(--checkpoint "${RUN_CHECKPOINT}") \
      || echo "[WARN] checkpoint not found, skipping: ${RUN_CHECKPOINT}" >&2
  fi
  _val_out="${ROOT}/results/FP_Base/fp_base_val_${RUN_FAMILY}_${RUN_MODEL}_${RUN_DTYPE}.csv"
  model_kwargs_args=()
  [[ "${RUN_FAMILY}" == "elsa_swin" ]] && model_kwargs_args=(--model-kwargs "elsa_backend=${RUN_BACKEND}")
  cmd=(
    python "${ROOT}/models/validate.py"
    --data-dir   "${RUN_DATA_DIR}"
    --split      "${RUN_SPLIT}"
    --model      "${RUN_MODEL}"
    --batch-size "${RUN_VAL_BATCH}"
    --img-size   "${RUN_IMG_SIZE}"
    --workers    "${RUN_WORKERS}"
    --device     "${RUN_DEVICE}"
    --model-dtype "float32"
    --results-file "${_val_out}"
    "${model_kwargs_args[@]}"
    "${checkpoint_args[@]}"
  )

elif [[ "${RUN_TASK}" == "train" ]]; then
  # 每次 train 加時間戳，獨立目錄，不互相覆蓋
  _stamp="$(date +%Y%m%d_%H%M%S)"
  _experiment="${RUN_FAMILY}_${RUN_MODEL}_fp_base_${_stamp}"
  initial_ckpt_args=()
  if [[ -n "${RUN_INITIAL_CHECKPOINT}" ]]; then
    [[ -f "${RUN_INITIAL_CHECKPOINT}" ]] && initial_ckpt_args=(--initial-checkpoint "${RUN_INITIAL_CHECKPOINT}") \
      || echo "[WARN] initial checkpoint not found: ${RUN_INITIAL_CHECKPOINT}" >&2
  fi
  model_kwargs_args=()
  [[ "${RUN_FAMILY}" == "elsa_swin" ]] && model_kwargs_args=(--model-kwargs "elsa_backend=${RUN_BACKEND}")
  cmd=(
    python "${ROOT}/models/train.py"
    --data-dir              "${RUN_DATA_DIR}"
    --train-split           train
    --val-split             "${RUN_SPLIT}"
    --model                 "${RUN_MODEL}"
    --batch-size            "${RUN_TRAIN_BATCH}"
    --validation-batch-size "${RUN_VAL_BATCH}"
    --img-size              "${RUN_IMG_SIZE}"
    --workers               "${RUN_WORKERS}"
    --device                "${RUN_DEVICE}"
    --epochs                "${RUN_EPOCHS}"
    --output                "${RUN_OUTPUT_DIR}"
    --experiment            "${_experiment}"
    --opt adamw --lr-base 0.001 --weight-decay 0.05
    --warmup-epochs 20 --sched cosine --min-lr 1e-5
    --mixup 0.8 --cutmix 1.0 --smoothing 0.1
    "${model_kwargs_args[@]}"
    "${initial_ckpt_args[@]}"
  )
else
  echo "[ERR] RUN_TASK must be bench | validate | train, got: ${RUN_TASK}" >&2; exit 2
fi

# ---- 執行 ---------------------------------------------------
echo "[FP_Base] task=${RUN_TASK} family=${RUN_FAMILY} model=${RUN_MODEL} backend=${RUN_BACKEND} dtype=${RUN_DTYPE} device=${RUN_DEVICE}" >&2
echo "[FP_Base] data=${RUN_DATA_DIR}" >&2
echo "[FP_Base] command: ${cmd[*]}" >&2

if [[ "${RUN_DRY_RUN}" == "1" ]]; then exit 0; fi

mkdir -p "${RUN_LOG_DIR}" "${ROOT}/results/FP_Base"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${RUN_LOG_DIR}/${stamp}_${RUN_TASK}_${RUN_FAMILY}_${RUN_MODEL}.log"
echo "[FP_Base] log=${log}" >&2

{ echo "[FP_Base] task=${RUN_TASK} family=${RUN_FAMILY} model=${RUN_MODEL} backend=${RUN_BACKEND} dtype=${RUN_DTYPE}"
  echo "[FP_Base] data=${RUN_DATA_DIR}"
  echo "[FP_Base] command: ${cmd[*]}"
  "${cmd[@]}"
} 2>&1 | grep -v "^🚨" | tee "${log}"
exit "${PIPESTATUS[0]}"
