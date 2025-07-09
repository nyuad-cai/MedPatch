#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:59:59
#SBATCH --cpus-per-task=16
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate medfuse

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py --dim 256 \
--dropout 0.3 --mode train \
--epochs 50 --pretrained \
--vision-backbone resnet34 --data_pairs paired_ehr_cxr \
--batch_size 16 --align 0.0 --labels_set mortality  --save_dir checkpoints/medfuse_Results \
--fusion_type uni_cxr --layers 2 --vision_num_classes 1 \
--task in-hospital-mortality \

