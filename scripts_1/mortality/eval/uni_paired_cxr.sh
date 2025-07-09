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
--vision-backbone resnet34 --lr 5.095704924187496e-05\
--mode eval \
--epochs 50 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--save_dir checkpoints/mortality/uni_cxr_all \
--fusion_type uni_cxr --task in-hospital-mortality 

