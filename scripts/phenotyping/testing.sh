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
--mode eval \
--epochs 1 --batch_size 16 --lr 0.0001 \
--num_classes 25 \
--modalities EHR-CXR-RR-DN \
--fusion_type ensemble \
--load_early mixed \
--ehr_encoder lstm \
--cxr_encoder vit_small_patch16_384 \
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
--data_pairs partial \
--load_model_1 '/scratch/baj321/MSMA/checkpoints/phenotyping/early/best_checkpoint_4.838001055284273e-05_phenotyping_early_EHR-CXR-RR-DN_partial.pth.tar' \
--load_model_2 '/scratch/baj321/MSMA/checkpoints/phenotyping/joint/best_checkpoint_3.764920670212206e-05_phenotyping_joint_EHR-CXR-RR-DN_partial.pth.tar' \
--load_ehr_3 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_ehr/best_checkpoint_0.0003041926769942766_phenotyping_unimodal_ehr_EHR_paired.pth.tar' \
--load_cxr_3 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar' \
--load_rr_3 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_rr/best_checkpoint_1.43757823218211e-05_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar' \
--load_dn_3 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_dn/best_checkpoint_0.00021528834007325203_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar'
# --load_ehr '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_ehr/best_checkpoint_0.0003041926769942766_phenotyping_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_rr '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_rr/best_checkpoint_1.43757823218211e-05_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar' \
# --load_dn '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_dn/best_checkpoint_0.00021528834007325203_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar'
# --load_ehr_1 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_ehr/best_checkpoint_0.0001882776644215001_phenotyping_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr_1 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_2.3921183199885533e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_rr_1 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_rr/best_checkpoint_0.0001150062585745688_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar' \
# --load_dn_1 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_dn/best_checkpoint_0.0008611533602930081_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar' \
# --load_ehr_2 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_ehr/best_checkpoint_0.00039947828865124184_phenotyping_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr_2 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_8.995380802416628e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_rr_2 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_rr/best_checkpoint_6.979492280790797e-05_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar' \
# --load_dn_2 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_dn/best_checkpoint_0.0008574577609317103_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar' \
# --load_ehr_3 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_ehr/best_checkpoint_0.0003041926769942766_phenotyping_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr_3 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_rr_3 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_rr/best_checkpoint_1.43757823218211e-05_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar' \
# --load_dn_3 '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_dn/best_checkpoint_0.00021528834007325203_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar'

