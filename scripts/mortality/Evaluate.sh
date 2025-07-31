#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-08:59:59
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err

    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate medpatch

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--mode eval \
--epochs 100 --batch_size 16 --lr 0.001 \
--num_classes 1 \
--modalities EHR-CXR-RR \
--fusion_type chooseModel \
--ehr_encoder lstm \
--cxr_encoder vit_small_patch16_384 \
--text_encoder biobert \
--load_ehr Your/Checkpoint \
--load_cxr Your/Checkpoint \
--load_rr Your/Checkpoint \
--load_state Your/Checkpoint \
--classifier mlp \
--loss bce \
--save_dir 'checkpoints/calibrated' \
--cxr_data_dir Your/Directory/data/physionet.org/files/mimic-cxr-jpg/2.0.0 \
--ehr_data_dir Your/Directory/data/mimic-iv-extracted \
--task in-hospital-mortality \
--labels_set mortality \
--output_dim 512 \
--output_dim_cxr 512 \
--output_dim_rr 512 \
--data_pairs partial \
--cxr_confidence_threshold 0.75 \
--ehr_confidence_threshold 0.75 \
--rr_confidence_threshold 0.75