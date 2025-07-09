#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#SBATCH --cpus-per-task=40
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate medfuse

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--vision-backbone resnet34 \
--dim 256 --layers 2 \
--mode train \
--epochs 3 --batch_size 16 --lr 0.00011115452661553602 \
--vision_num_classes 14 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--fusion_type uni_ehr \
--task decompensation \
--labels_set decompensation \
--save_dir checkpoints/decompensation/uni_ehr_all 
