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

CUDA_VISIBLE_DEVICES=0  python fusion_main.py \
--dim 256 --layers 2 \
--vision-backbone resnet34 \
--mode eval \
--epochs 50 --batch_size 16 --lr 0.00011115452661553602 \
--vision_num_classes 10 --num_classes 10 \
--data_pairs paired_ehr_cxr \
--fusion_type uni_ehr --task length-of-stay \
--save_dir checkpoints/length-of-stay/uni_ehr_all \
--labels_set length-of-stay