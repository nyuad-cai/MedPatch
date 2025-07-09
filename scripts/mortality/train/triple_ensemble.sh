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
--epochs 50 --batch_size 4 --lr 0.00000077494775956886 \
--num_classes 1 \
--modalities EHR-CXR-RR \
--order triple-ensemble \
--H_mode triple-late \
--load_model_1 sweep-checkpoints/in-hospital-mortality/late/best_checkpoint_2.370810604992005e-05_in-hospital-mortality_late_trimodal_paired_late.pth.tar \
--load_model_2 sweep-checkpoints/in-hospital-mortality/late/best_checkpoint_2.2370872476611134e-05_in-hospital-mortality_late_trimodal_paired_late.pth.tar \
--load_model_3 sweep-checkpoints/in-hospital-mortality/late/best_checkpoint_2.252566057419246e-05_in-hospital-mortality_late_trimodal_paired_late.pth.tar \
--save_dir checkpoints \
--task in-hospital-mortality \
--labels_set mortality

