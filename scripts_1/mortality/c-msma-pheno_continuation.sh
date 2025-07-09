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
--epochs 155 --batch_size 16 --lr 0.00022145277520170303 \
--num_classes 25 \
--modalities EHR-CXR-RR-DN \
--ehr_confidence_threshold 0.7 \
--ehr_encoder lstm \
--cxr_confidence_threshold 0.7 \
--cxr_encoder vit_small_patch16_384 \
--fusion_type c-msma \
--dn_confidence_threshold 0.7 \
--rr_confidence_threshold 0.7 \
--text_encoder biobert \
--classifier mlp \
--loss bce \
--save_dir 'checkpoints/continued' \
--task phenotyping \
--labels_set pheno \
--output_dim 512 \
--output_dim_cxr 512 \
--output_dim_rr 512 \
--output_dim_dn 512 \
--data_pairs partial \
--weight_high 1 \
--weight_low 0.5 \
--weight_late 2 \
--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/c-msma/best_checkpoint_0.00022145277520170303_phenotyping_c-msma_EHR-CXR-RR-DN_partial.pth.tar'