#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-23:59:59
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate medfuse

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--mode train \
--epochs 50 --batch_size 1 --lr 0.00001 \
--num_classes 1 \
--modalities EHR-CXR \
--order EHR-CXR \
--H_mode relevancy-based-hierarchical \
--load_ehr checkpoints/in-hospital-mortality/unimodal/best_checkpoint_unimodal_1e-05_in-hospital-mortality_EHR.pth.tar \
--load_cxr checkpoints/in-hospital-mortality/unimodal/best_checkpoint_unimodal_1e-06_in-hospital-mortality_CXR.pth.tar \
--save_dir checkpoints \
--task in-hospital-mortality \
--labels_set mortality

