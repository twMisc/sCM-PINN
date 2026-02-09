set -e  # exit on error

python train_diffusion_model.py
python train_consistency.py
python train_stage_2_unified.py --config="config/stage2_fdm.yaml"