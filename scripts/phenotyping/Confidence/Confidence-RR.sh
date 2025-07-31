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
conda activate medpatch

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--mode train \
--epochs 100 --batch_size 16 --lr 0.001 \
--num_classes 25 \
--modalities EHR-RR \
--fusion_type c-unimodal_rr \
--text_encoder biobert \
--load_rr Your/Checkpoint \
--classifier mlp \
--loss bce \
--save_dir 'checkpoints/' \
--cxr_data_dir Your/Directory/data/physionet.org/files/mimic-cxr-jpg/2.0.0 \
--ehr_data_dir Your/Directory/data/mimic-iv-extracted \
--task phenotyping \
--labels_set pheno \
--output_dim_rr 512 \
--data_pairs paired 
