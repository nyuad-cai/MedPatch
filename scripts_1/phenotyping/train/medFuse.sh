#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate medfuse

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 50 --batch_size 16 --lr 0.0003052106285255343 \
--vision_num_classes 14 --num_classes 25 \
--data_pairs paired_ehr_cxr \
--fusion_type lstm \
--labels_set pheno \
--save_dir checkpoints/phenotyping/besttt \
--load_state_cxr checkpoints/medfuse_Results/best_uni_cxr_0.0001_checkpoint.pth.tar \
--load_state_ehr checkpoints/phenotyping/bestt/best_uni_ehr_phenotyping_0.0004388215792647098_checkpoint.pth.tar