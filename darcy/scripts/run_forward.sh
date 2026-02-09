#!/bin/bash
# ============================================================
# Benchmark Forward / Inverse / Reconstruction Task Errors
# Script for eval_task_error.py
# ============================================================

set -e  # exit on error

# ------------------------------------------------------------
# Common settings
# ------------------------------------------------------------
DEVICE="cuda"
BATCH_SIZE=4
TOTAL_SAMPLES=256
SEED=123
SAVE_PATH="./eval_results"
SAVE_NAME="task_error.csv"
FIGURE_BASE="./figures"

mkdir -p "${SAVE_PATH}"
mkdir -p "${FIGURE_BASE}"

# ------------------------------------------------------------
# Problem settings
#   forward | inverse | reconstruction
# ------------------------------------------------------------
PROBLEM_TYPE="forward"

# Reconstruction-only options (ignored otherwise)
RECON_MASK_PERCENT=0.75
MASK_MSE_EVAL=""

# PDE residual option
USE_PDE_MASK=""

# ------------------------------------------------------------
# Sampling step sweeps
# ------------------------------------------------------------
CM_STEPS_LIST=(16 32 64)
GUIDED_STEPS_LIST=(16 32 64)
# DIFFUSION_STEPS_LIST=(16 32 64)

RHO=7.0
ZETA_OBS_A=5.0
ZETA_OBS_U=0.0
ZETA_PDE=100.0

# ------------------------------------------------------------
# Model configurations
# Each model is one entry in the MODELS array
#
# Format:
#   "MODEL_NAME|MODEL_TYPE|NETWORK_TYPE|MODEL_PATH"
#
# MODEL_TYPE   : cm | diffusion
# NETWORK_TYPE : unet | sep_unet
# ------------------------------------------------------------
MODELS=(
    # "consistency-fdm-sep-nomask-uniform|cm|sep_unet|./darcy_redo_output/sCM/consistency-fdm-sep-nomask-uniform/model_epoch/model_epoch_1.pth"
    # "cm|cm|unet|./darcy_redo_output/sCM/consistency/model_epoch/model_epoch_8.pth"
    "diffpde|diffpde|unet|../DiffusionPDE_data/pretrained-models/pretrained-darcy.pkl"    
)

# ============================================================
# Main loop
# ============================================================
for MODEL_CFG in "${MODELS[@]}"; do
    IFS="|" read -r MODEL_NAME MODEL_TYPE NETWORK_TYPE MODEL_PATH <<< "${MODEL_CFG}"

    echo "============================================================"
    echo "Model: ${MODEL_NAME}"
    echo "  Type   : ${MODEL_TYPE}"
    echo "  Network: ${NETWORK_TYPE}"
    echo "============================================================"

    if [[ "${MODEL_TYPE}" == "cm" ]]; then
        # ----------------------------
        # Consistency Models
        # ----------------------------
        for STEPS in "${CM_STEPS_LIST[@]}"; do
            FIG_DIR="${FIGURE_BASE}/${MODEL_NAME}/${PROBLEM_TYPE}/cm_steps_${STEPS}"
            mkdir -p "${FIG_DIR}"

            python eval_task_error.py \
                --device "${DEVICE}" \
                --batch_size "${BATCH_SIZE}" \
                --total_samples "${TOTAL_SAMPLES}" \
                --seed "${SEED}" \
                --model_type "cm" \
                --network_type "${NETWORK_TYPE}" \
                --model_path "${MODEL_PATH}" \
                --cm_steps "${STEPS}" \
                --problem_type "${PROBLEM_TYPE}" \
                --recon_mask_percent "${RECON_MASK_PERCENT}" \
                --figure_dir "${FIG_DIR}" \
                --figure_name "sample.png" \
                --save_path "${SAVE_PATH}" \
                --save_name "${SAVE_NAME}" \
                --network_name "${MODEL_NAME}" \
                ${USE_PDE_MASK} \
                ${MASK_MSE_EVAL}

        done

    elif [[ "${MODEL_TYPE}" == "diffpde" ]]; then
        # ----------------------------
        # DiffusionPDE model
        # ----------------------------
        for STEPS in "${GUIDED_STEPS_LIST[@]}"; do
            FIG_DIR="${FIGURE_BASE}/${MODEL_NAME}/${PROBLEM_TYPE}/diffpde_steps_${STEPS}"
            mkdir -p "${FIG_DIR}"

            python eval_task_error.py \
                --device "${DEVICE}" \
                --batch_size "${BATCH_SIZE}" \
                --total_samples "${TOTAL_SAMPLES}" \
                --seed "${SEED}" \
                --model_type "diffpde" \
                --network_type "${NETWORK_TYPE}" \
                --model_path "${MODEL_PATH}" \
                --num_steps "${STEPS}" \
                --rho "${RHO}" \
                --zeta_obs_a "${ZETA_OBS_A}" \
                --zeta_obs_u "${ZETA_OBS_U}" \
                --zeta_pde "${ZETA_PDE}" \
                --problem_type "${PROBLEM_TYPE}" \
                --recon_mask_percent "${RECON_MASK_PERCENT}" \
                --figure_dir "${FIG_DIR}" \
                --figure_name "sample.png" \
                --save_path "${SAVE_PATH}" \
                --save_name "${SAVE_NAME}" \
                --network_name "${MODEL_NAME}" \
                ${USE_PDE_MASK} \
                ${MASK_MSE_EVAL}
        done

    else
        # ----------------------------
        # Diffusion Models
        # ----------------------------
        for STEPS in "${DIFFUSION_STEPS_LIST[@]}"; do
            FIG_DIR="${FIGURE_BASE}/${MODEL_NAME}/${PROBLEM_TYPE}/diff_steps_${STEPS}"
            mkdir -p "${FIG_DIR}"

            python eval_task_error.py \
                --device "${DEVICE}" \
                --batch_size "${BATCH_SIZE}" \
                --total_samples "${TOTAL_SAMPLES}" \
                --seed "${SEED}" \
                --model_type "diffusion" \
                --network_type "${NETWORK_TYPE}" \
                --model_path "${MODEL_PATH}" \
                --num_steps "${STEPS}" \
                --problem_type "${PROBLEM_TYPE}" \
                --recon_mask_percent "${RECON_MASK_PERCENT}" \
                --figure_dir "${FIG_DIR}" \
                --figure_name "sample.png" \
                --save_path "${SAVE_PATH}" \
                --save_name "${SAVE_NAME}" \
                --network_name "${MODEL_NAME}" \
                ${USE_PDE_MASK} \
                ${MASK_MSE_EVAL}

        done
    fi
done

echo "All evaluations completed."
