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
--epochs 50 --batch_size 16 --lr 0.0001 \
--num_classes 1 \
--modalities EHR-CXR \
--data_pairs paired \
--order EHR-CXR-RR-ensemble \
--H_mode ensemble \
--load_early checkpoints/in-hospital-mortality/early/best_checkpoint_0.0001_in-hospital-mortality_early_trimodal.pth.tar \
--load_joint checkpoints/in-hospital-mortality/joint/best_checkpoint_1e-05_in-hospital-mortality_joint_trimodal.pth.tar \
--load_late checkpoints/in-hospital-mortality/late/best_checkpoint_1e-05_in-hospital-mortality_late_trimodal.pth.tar \
--save_dir checkpoints \
--task in-hospital-mortality \
--labels_set mortality

