#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-08:00:00
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate medfuse

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--mode train \
--epochs 50 --batch_size 4 --lr 0.00000077494775956886 \
--num_classes 1 \
--modalities EHR-CXR-RR \
--data_pairs paired \
--order EHR-CXR-RR-ensemble-late \
--H_mode ensemble-late \
--save_dir checkpoints \
--task in-hospital-mortality \
--labels_set mortality
--load_early sweep-checkpoints/in-hospital-mortality/early/best_checkpoint_0.0004380324317987968_in-hospital-mortality_early_trimodal_paired_early.pth.tar \
--load_joint sweep-checkpoints/in-hospital-mortality/joint/best_checkpoint_0.0001102949949768536_in-hospital-mortality_joint_trimodal_paired_joint.pth.tar \
--load_late sweep-checkpoints/in-hospital-mortality/late/best_checkpoint_2.370810604992005e-05_in-hospital-mortality_late_trimodal_paired_late.pth.tar \

