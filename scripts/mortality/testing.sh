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
--epochs 2 --batch_size 16 --lr 0.001 \
--num_classes 25 \
--modalities EHR-RR \
--fusion_type temp_c-unimodal_rr \
--calibration calibrate \
--text_encoder biobert \
--classifier mlp \
--loss bce \
--save_dir 'checkpoints/calibrated' \
--task phenotyping \
--labels_set pheno \
--output_dim 512 \
--output_dim_cxr 512 \
--output_dim_rr 512 \
--data_pairs paired \
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/c-unimodal_rr/best_checkpoint_0.0001_in-hospital-mortality_c-unimodal_rr_EHR-RR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/c-unimodal_cxr/best_checkpoint_0.0001_in-hospital-mortality_c-unimodal_cxr_EHR-CXR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/c-unimodal_ehr/best_checkpoint_0.0001_in-hospital-mortality_c-unimodal_ehr_EHR_paired.pth.tar'

# --pretrained_model '/scratch/baj321/MSMA/checkpoints/vit/cls/phenotyping/c-unimodal_cxr/best_checkpoint_0.0007339519834304888_phenotyping_c-unimodal_cxr_EHR-CXR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/c-unimodal_ehr/best_checkpoint_4.3662561397696474e-05_phenotyping_c-unimodal_ehr_EHR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/joint/best_checkpoint_1.7290509449737742e-05_phenotyping_joint_EHR-CXR_partial.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/joint/best_checkpoint_1.7290509449737742e-05_phenotyping_joint_EHR-CXR_partial.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/joint/best_checkpoint_8.446410203268946e-05_phenotyping_joint_EHR-CXR-RR-DN_partial.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_ehr/best_checkpoint_0.0003041926769942766_phenotyping_unimodal_ehr_EHR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/c-unimodal_ehr/best_checkpoint_4.3662561397696474e-05_phenotyping_c-unimodal_ehr_EHR_paired.pth.tar'
#c
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/c-unimodal_rr/best_checkpoint_0.0001_in-hospital-mortality_c-unimodal_rr_EHR-RR_paired.pth.tar'
#--load_rr '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/c-unimodal_rr/best_checkpoint_0.0008954749935045941_in-hospital-mortality_c-unimodal_rr_EHR-RR_paired.pth.tar'
# --load_ehr '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_ehr/best_checkpoint_8.995380802416628e-05_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr '/scratch/baj321/MSMA/checkpoints/vit/cls/in-hospital-mortality/unimodal_cxr/best_checkpoint_1.240393186295578e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_rr '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_rr/best_checkpoint_2.181972195740935e-05_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar'
# --load_ehr '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_ehr/best_checkpoint_8.995380802416628e-05_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_cxr/best_checkpoint_1.9620812058430023e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/phenotyping/unimodal_dn/best_checkpoint_0.00014681285318933384_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar'
# --ehr_encoder lstm \
# --cxr_encoder vit_small_patch16_384 \
# --use_cls_token cls \
# --ablation reverse_w \

# --load_model_1 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/early/best_checkpoint_0.000253314032136948_in-hospital-mortality_early_EHR-CXR-RR_partial.pth.tar' \
# --load_model_2 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/joint/best_checkpoint_1.0052767507013117e-05_in-hospital-mortality_joint_EHR-CXR-RR_partial.pth.tar' \
# --load_model_3 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/joint/best_checkpoint_0.00018620863641332256_in-hospital-mortality_joint_EHR-CXR-RR_partial.pth.tar' \
# --load_ehr_3 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_ehr/best_checkpoint_8.995380802416628e-05_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr_3 '/scratch/baj321/MSMA/checkpoints/vit/cls/in-hospital-mortality/unimodal_cxr/best_checkpoint_1.240393186295578e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_rr_3 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_rr/best_checkpoint_2.181972195740935e-05_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar'
# --load_model_1 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/joint/best_checkpoint_1.0052767507013117e-05_in-hospital-mortality_joint_EHR-CXR-RR_partial.pth.tar' \
# --load_model_2 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/joint/best_checkpoint_1.0893028880992206e-05_in-hospital-mortality_joint_EHR-CXR-RR_partial.pth.tar' \
# --load_model_3 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/joint/best_checkpoint_0.00018620863641332256_in-hospital-mortality_joint_EHR-CXR-RR_partial.pth.tar'
# --load_model_1 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/early/best_checkpoint_0.000253314032136948_in-hospital-mortality_early_EHR-CXR-RR_partial.pth.tar' \
# --load_model_2 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/early/best_checkpoint_2.504445474742038e-05_in-hospital-mortality_early_EHR-CXR-RR_partial.pth.tar' \
# --load_model_3 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/early/best_checkpoint_0.0003041926769942766_in-hospital-mortality_early_EHR-CXR-RR_partial.pth.tar'
# --load_ehr_1 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_ehr/best_checkpoint_1.9620812058430023e-05_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar' \
# --load_ehr_2 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_ehr/best_checkpoint_2.3921183199885533e-05_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar' \
# --load_ehr_3 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_ehr/best_checkpoint_8.995380802416628e-05_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr_1 '/scratch/baj321/MSMA/checkpoints/vit/cls/in-hospital-mortality/unimodal_cxr/best_checkpoint_1.4484104086398404e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_cxr_2 '/scratch/baj321/MSMA/checkpoints/vit/cls/in-hospital-mortality/unimodal_cxr/best_checkpoint_2.7647670350210293e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_cxr_3 '/scratch/baj321/MSMA/checkpoints/vit/cls/in-hospital-mortality/unimodal_cxr/best_checkpoint_1.240393186295578e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_rr_1 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_rr/best_checkpoint_1.589333510528785e-05_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar' \
# --load_rr_2 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_rr/best_checkpoint_1.0589431311904655e-05_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar' \
# --load_rr_3 '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_rr/best_checkpoint_2.181972195740935e-05_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar'
#--pretrained_model '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/metra/best_checkpoint_4.3662561397696474e-05_in-hospital-mortality_metra_EHR_partial.pth.tar'
# --freeze all \
# --load_ehr '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_ehr/best_checkpoint_8.995380802416628e-05_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar' \
# --load_cxr '/scratch/baj321/MSMA/checkpoints/vit/cls/in-hospital-mortality/unimodal_cxr/best_checkpoint_1.240393186295578e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar' \
# --load_rr '/scratch/baj321/MSMA/checkpoints/in-hospital-mortality/unimodal_rr/best_checkpoint_2.181972195740935e-05_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar'

