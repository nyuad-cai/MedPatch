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
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode eval \
--epochs 50 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type late_avg \
--save_dir checkpoints/mortality/medFuse \
--load_state_ehr checkpoints/mortality/uni_ehr_all/best_uni_ehr_in-hospital-mortality_0.0008123508746111482_checkpoint.pth.tar \
--load_state_cxr checkpoints/mortality/uni_cxr_all/best_uni_cxr_in-hospital-mortality_5.095704924187496e-05_checkpoint.pth.tar