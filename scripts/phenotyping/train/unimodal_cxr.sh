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

python fusion_main.py \
--mode train \
--epochs 50 --batch_size 16 --lr 0.000001 \
--num_classes 25 \
--modalities EHR-CXR \
--pretraining CXR \
--H_mode unimodal \
--save_dir checkpoints/ \
--task phenotyping \
--labels_set phenotyping

