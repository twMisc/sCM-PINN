#!/bin/bash
# ============================================================
# Benchmark Unconditional Sampling Quality (PDE Residuals)
# Script for eval_unconditional.py
# ============================================================

set -e  # exit on error

# ------------------------------------------------------------
# Common settings
# ------------------------------------------------------------
DEVICE="cuda"
BATCH_SIZE=1
TOTAL_SAMPLES=256
SEED=123
SAVE_PATH="./eval_results_uncond"
SAVE_NAME="uncond_error.csv"
FIGURE_BASE="./figures_uncond"

mkdir -p "${SAVE_PATH}"
mkdir -p "${FIGURE_BASE}"

# ------------------------------------------------------------
# Sampling step sweeps
# ------------------------------------------------------------
CM_STEPS_LIST=(2) # Single step or few step for CM
GUIDED_STEPS_LIST=(32) # Standard for DiffPDE
# DIFFUSION_STEPS_LIST=(32 64)

RHO=7.0

# ------------------------------------------------------------
# Model configurations
# Each model is one entry in the MODELS array
#
# Format:
#   "MODEL_NAME|MODEL_TYPE|NETWORK_TYPE|MODEL_PATH"
#
# MODEL_TYPE   : cm | diffusion | diffpde
# NETWORK_TYPE : unet | sep_unet
# ------------------------------------------------------------
MODELS=(
    "sCM-PINN|cm|sep_unet|./darcy_output/sCM/consistency-fdm-sep-nomask-uniform/model_epoch/model_epoch_1.pth"
    "sCM-Base|cm|unet|./darcy_output/sCM/consistency/model_epoch/model_epoch_8.pth"
    "DiffusionPDE|diffpde|unet|../DiffusionPDE_data/pretrained-models/pretrained-darcy.pkl"    
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

    # 1. Consistency Models
    if [[ "${MODEL_TYPE}" == "cm" ]]; then
        for STEPS in "${CM_STEPS_LIST[@]}"; do
            FIG_DIR="${FIGURE_BASE}/${MODEL_NAME}/steps_${STEPS}"
            mkdir -p "${FIG_DIR}"

            python eval_unconditional.py \
                --device "${DEVICE}" \
                --batch_size "${BATCH_SIZE}" \
                --total_samples "${TOTAL_SAMPLES}" \
                --seed "${SEED}" \
                --model_type "cm" \
                --network_type "${NETWORK_TYPE}" \
                --model_path "${MODEL_PATH}" \
                --cm_steps "${STEPS}" \
                --figure_dir "${FIG_DIR}" \
                --save_path "${SAVE_PATH}" \
                --save_name "${SAVE_NAME}" \
                --network_name "${MODEL_NAME}"

        done

    # 2. DiffusionPDE (Pickle model)
    elif [[ "${MODEL_TYPE}" == "diffpde" ]]; then
        for STEPS in "${GUIDED_STEPS_LIST[@]}"; do
            FIG_DIR="${FIGURE_BASE}/${MODEL_NAME}/steps_${STEPS}"
            mkdir -p "${FIG_DIR}"

            # Note: We don't pass zetas here because the python script 
            # hardcodes them to 0.0 for unconditional sampling.
            python eval_unconditional.py \
                --device "${DEVICE}" \
                --batch_size "${BATCH_SIZE}" \
                --total_samples "${TOTAL_SAMPLES}" \
                --seed "${SEED}" \
                --model_type "diffpde" \
                --network_type "${NETWORK_TYPE}" \
                --model_path "${MODEL_PATH}" \
                --num_steps "${STEPS}" \
                --rho "${RHO}" \
                --figure_dir "${FIG_DIR}" \
                --save_path "${SAVE_PATH}" \
                --save_name "${SAVE_NAME}" \
                --network_name "${MODEL_NAME}"
        done

    # 3. Standard Diffusion (DPM-Solver)
    else
        for STEPS in "${DIFFUSION_STEPS_LIST[@]}"; do
            FIG_DIR="${FIGURE_BASE}/${MODEL_NAME}/steps_${STEPS}"
            mkdir -p "${FIG_DIR}"

            python eval_unconditional.py \
                --device "${DEVICE}" \
                --batch_size "${BATCH_SIZE}" \
                --total_samples "${TOTAL_SAMPLES}" \
                --seed "${SEED}" \
                --model_type "diffusion" \
                --network_type "${NETWORK_TYPE}" \
                --model_path "${MODEL_PATH}" \
                --num_steps "${STEPS}" \
                --figure_dir "${FIG_DIR}" \
                --save_path "${SAVE_PATH}" \
                --save_name "${SAVE_NAME}" \
                --network_name "${MODEL_NAME}"

        done
    fi
done

echo "All unconditional evaluations completed."