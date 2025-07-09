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
conda activate medfuse

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--mode train \
--epochs 100 --batch_size 16 --lr 0.0000143757823218211 \
--num_classes 25 \
--modalities EHR-RR \
--fusion_type unimodal_rr \
--text_encoder biobert \
--classifier mlp \
--loss bce \
--save_dir checkpoints \
--task phenotyping \
--labels_set pheno \
--output_dim 512 \
--output_dim_cxr 512 \
--output_dim_rr 512 \
--output_dim_dn 512 \
--data_pairs paired \
--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_rr/best_checkpoint_0.0001150062585745688_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar'
