import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.ehr_encoder import EHRTransformer
from models.cxr_encoder import CXRTransformer
from models.rr_encoder import RadiologyNotesEncoder
from models.dn_encoder import DischargeNotesEncoder
from models.classifier import MLPClassifier
from models.customtransformer import CustomTransformerLayer
from .trainer import Trainer
import pandas as pd
import os

import numpy as np
from sklearn import metrics
import wandb

class EnsembleFusionTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl
        ):

        super(EnsembleFusionTrainer, self).__init__(args)
        run = wandb.init(project=f'Fusion_{self.args.H_mode}_{self.args.task}', config=args)
        self.epoch = 0 
        self.start_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.token_dim = 384
        
        self.seed = 1002
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
        # Initialize encoders for each modality and type (early, joint, late)
        self.early_ehr_encoder = EHRTransformer(self.args,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            dim_head=128
        ).to(self.device)
        self.joint_ehr_encoder = EHRTransformer(self.args,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            dim_head=128
        ).to(self.device)
        self.late_ehr_encoder = EHRTransformer(self.args,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            dim_head=128
        ).to(self.device)
        
        self.early_cxr_encoder = CXRTransformer(
            model_name='vit_small_patch16_384',
            image_size=384,
            patch_size=16,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            emb_dropout=0.0,
            dim_head=128
        ).to(self.device)
        
        self.joint_cxr_encoder = CXRTransformer(
            model_name='vit_small_patch16_384',
            image_size=384,
            patch_size=16,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            emb_dropout=0.0,
            dim_head=128
        ).to(self.device)
        
        self.late_cxr_encoder = CXRTransformer(
            model_name='vit_small_patch16_384',
            image_size=384,
            patch_size=16,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            emb_dropout=0.0,
            dim_head=128
        ).to(self.device)
        
        self.early_dn_encoder = DischargeNotesEncoder(device= self.device,
            pretrained_model_name='allenai/longformer-base-4096',
            output_dim=384
        ).to(self.device)
        
        self.joint_dn_encoder = DischargeNotesEncoder(device= self.device,
            pretrained_model_name='allenai/longformer-base-4096',
            output_dim=384
        ).to(self.device)
        
        self.late_dn_encoder = DischargeNotesEncoder(device= self.device,
            pretrained_model_name='allenai/longformer-base-4096',
            output_dim=384
        ).to(self.device)
        
        
        self.early_rr_encoder = RadiologyNotesEncoder(device= self.device, 
            pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT',
            output_dim=384
        ).to(self.device)
        
        self.joint_rr_encoder = RadiologyNotesEncoder(device= self.device, 
            pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT',
            output_dim=384
        ).to(self.device)
        
        self.late_rr_encoder = RadiologyNotesEncoder(device= self.device, 
            pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT',
            output_dim=384
        ).to(self.device)
        
        self.final_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        
        # Initialize transformer layers
        self.early_transformer_layer = CustomTransformerLayer(input_dim=384 * len(self.args.modalities.split('-')), model_dim=384, nhead=4, num_layers=1).to(self.device)
        self.joint_transformer_layer = CustomTransformerLayer(input_dim=384 * len(self.args.modalities.split('-')), model_dim=384, nhead=4, num_layers=1).to(self.device)
        self.late_transformer_layer = CustomTransformerLayer(input_dim=384 * len(self.args.modalities.split('-')), model_dim=384, nhead=4, num_layers=1).to(self.device)
        self.final_transformer_layer = CustomTransformerLayer(input_dim=384 * 3, model_dim=384, nhead=4, num_layers=1).to(self.device)
        #self.final_transformer_layer = CustomTransformerLayer(input_dim=384 * len(self.args.modalities.split('-')) * 3, model_dim=384, nhead=4, num_layers=1).to(self.device)
        
        if self.args.load_early:
            checkpoint = torch.load(self.args.load_early)
            self.early_ehr_encoder.load_state_dict(checkpoint['ehr_encoder_state_dict'])
            self.early_cxr_encoder.load_state_dict(checkpoint['cxr_encoder_state_dict'])
            self.early_dn_encoder.load_state_dict(checkpoint['dn_encoder_state_dict'])
            self.early_rr_encoder.load_state_dict(checkpoint['rr_encoder_state_dict'])
            print("early loaded")
            
        if self.args.load_joint:
            checkpoint = torch.load(self.args.load_joint)
            self.joint_ehr_encoder.load_state_dict(checkpoint['ehr_encoder_state_dict'])
            self.joint_cxr_encoder.load_state_dict(checkpoint['cxr_encoder_state_dict'])
            self.joint_dn_encoder.load_state_dict(checkpoint['dn_encoder_state_dict'])
            self.joint_rr_encoder.load_state_dict(checkpoint['rr_encoder_state_dict'])
            print("joint loaded")
        
        if self.args.load_late:
            checkpoint = torch.load(self.args.load_late)
            self.late_ehr_encoder.load_state_dict(checkpoint['ehr_encoder_state_dict'])
            self.late_cxr_encoder.load_state_dict(checkpoint['cxr_encoder_state_dict'])
            self.late_dn_encoder.load_state_dict(checkpoint['dn_encoder_state_dict'])
            self.late_rr_encoder.load_state_dict(checkpoint['rr_encoder_state_dict'])
            print("late loaded")
        
        for param in self.early_ehr_encoder.parameters():
            param.requires_grad = False
        for param in self.early_cxr_encoder.parameters():
            param.requires_grad = False
        for param in self.early_dn_encoder.parameters():
            param.requires_grad = False
        for param in self.early_rr_encoder.parameters():
            param.requires_grad = False
        for param in self.joint_ehr_encoder.parameters():
            param.requires_grad = False
        for param in self.joint_cxr_encoder.parameters():
            param.requires_grad = False
        for param in self.joint_dn_encoder.parameters():
            param.requires_grad = False
        for param in self.joint_rr_encoder.parameters():
            param.requires_grad = False
        for param in self.late_ehr_encoder.parameters():
            param.requires_grad = False
        for param in self.late_cxr_encoder.parameters():
            param.requires_grad = False
        for param in self.late_dn_encoder.parameters():
            param.requires_grad = False
        for param in self.late_rr_encoder.parameters():
            param.requires_grad = False
            
        all_params = (
                list(self.joint_ehr_encoder.parameters()) +
                list(self.joint_cxr_encoder.parameters()) +
                list(self.joint_dn_encoder.parameters()) +
                list(self.joint_rr_encoder.parameters()) +
                list(self.late_ehr_encoder.parameters()) +
                list(self.late_cxr_encoder.parameters()) +
                list(self.late_dn_encoder.parameters()) +
                list(self.late_rr_encoder.parameters()) +
                list(self.early_transformer_layer.parameters()) +
                list(self.joint_transformer_layer.parameters()) +
                list(self.late_transformer_layer.parameters()) +
                list(self.final_transformer_layer.parameters()) +
                list(self.final_classifier.parameters())
            )

        self.count_params(all_params)
        self.optimizer = optim.Adam(all_params, lr=args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
    
    def count_params(self, param_list):
        total_params = sum(p.numel() for p in param_list)
        trainable_params = sum(p.numel() for p in param_list if p.requires_grad)
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {total_params - trainable_params}")
        
    def save_fusion_checkpoint(self):
        # Define the checkpoint directory path
        checkpoint_dir = f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}'
        
        # Create the directory and all intermediate-level directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'early_ehr_encoder_state_dict': self.early_ehr_encoder.state_dict(),
            'joint_ehr_encoder_state_dict': self.joint_ehr_encoder.state_dict(),
            'late_ehr_encoder_state_dict': self.late_ehr_encoder.state_dict(),
            'early_cxr_encoder_state_dict': self.early_cxr_encoder.state_dict(),
            'joint_cxr_encoder_state_dict': self.joint_cxr_encoder.state_dict(),
            'late_cxr_encoder_state_dict': self.late_cxr_encoder.state_dict(),
            'early_dn_encoder_state_dict': self.early_dn_encoder.state_dict(),
            'joint_dn_encoder_state_dict': self.joint_dn_encoder.state_dict(),
            'late_dn_encoder_state_dict': self.late_dn_encoder.state_dict(),
            'early_rr_encoder_state_dict': self.early_rr_encoder.state_dict(),
            'joint_rr_encoder_state_dict': self.joint_rr_encoder.state_dict(),
            'late_rr_encoder_state_dict': self.late_rr_encoder.state_dict(),
            'early_transformer_layer_state_dict': self.early_transformer_layer.state_dict(),
            'joint_transformer_layer_state_dict': self.joint_transformer_layer.state_dict(),
            'late_transformer_layer_state_dict': self.late_transformer_layer.state_dict(),
            'final_transformer_layer_state_dict': self.final_transformer_layer.state_dict(),
            'final_classifier_state_dict': self.final_classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'{checkpoint_dir}/best_checkpoint_{self.args.H_mode}_{self.args.order}_{self.args.lr}_{self.args.task}_{self.args.data_pairs}.pth.tar')

    def load_fusion_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        self.early_ehr_encoder.load_state_dict(checkpoint['early_ehr_encoder_state_dict'])
        self.joint_ehr_encoder.load_state_dict(checkpoint['joint_ehr_encoder_state_dict'])
        self.late_ehr_encoder.load_state_dict(checkpoint['late_ehr_encoder_state_dict'])
        self.early_cxr_encoder.load_state_dict(checkpoint['early_cxr_encoder_state_dict'])
        self.joint_cxr_encoder.load_state_dict(checkpoint['joint_cxr_encoder_state_dict'])
        self.late_cxr_encoder.load_state_dict(checkpoint['late_cxr_encoder_state_dict'])
        self.early_dn_encoder.load_state_dict(checkpoint['early_dn_encoder_state_dict'])
        self.joint_dn_encoder.load_state_dict(checkpoint['joint_dn_encoder_state_dict'])
        self.late_dn_encoder.load_state_dict(checkpoint['late_dn_encoder_state_dict'])
        self.early_rr_encoder.load_state_dict(checkpoint['early_rr_encoder_state_dict'])
        self.joint_rr_encoder.load_state_dict(checkpoint['joint_rr_encoder_state_dict'])
        self.late_rr_encoder.load_state_dict(checkpoint['late_rr_encoder_state_dict'])
        self.early_transformer_layer.load_state_dict(checkpoint['early_transformer_layer_state_dict'])
        self.joint_transformer_layer.load_state_dict(checkpoint['joint_transformer_layer_state_dict'])
        self.late_transformer_layer.load_state_dict(checkpoint['late_transformer_layer_state_dict'])
        self.final_transformer_layer.load_state_dict(checkpoint['final_transformer_layer_state_dict'])
        self.final_classifier.load_state_dict(checkpoint['final_classifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def set_train_mode(self):
        """Set all neural network components to training mode."""
        self.early_ehr_encoder.train()
        self.early_cxr_encoder.train()
        self.early_dn_encoder.train()
        self.early_rr_encoder.train()
        self.joint_ehr_encoder.train()
        self.joint_cxr_encoder.train()
        self.joint_dn_encoder.train()
        self.joint_rr_encoder.train()
        self.late_ehr_encoder.train()
        self.late_cxr_encoder.train()
        self.late_dn_encoder.train()
        self.late_rr_encoder.train()
        self.early_transformer_layer.train()
        self.joint_transformer_layer.train()
        self.late_transformer_layer.train()
        self.final_transformer_layer.train()
        self.final_classifier.train()

    def set_eval_mode(self):
        """Set all neural network components to evaluation mode."""
        self.early_ehr_encoder.eval()
        self.early_cxr_encoder.eval()
        self.early_dn_encoder.eval()
        self.early_rr_encoder.eval()
        self.joint_ehr_encoder.eval()
        self.joint_cxr_encoder.eval()
        self.joint_dn_encoder.eval()
        self.joint_rr_encoder.eval()
        self.late_ehr_encoder.eval()
        self.late_cxr_encoder.eval()
        self.late_dn_encoder.eval()
        self.late_rr_encoder.eval()
        self.early_transformer_layer.eval()
        self.joint_transformer_layer.eval()
        self.late_transformer_layer.eval()
        self.final_transformer_layer.eval()
        self.final_classifier.eval()

    
    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            if self.args.task == 'in-hospital-mortality':
                y = y.unsqueeze(1)
            img = img.to(self.device)
            
            early_vectors = []
            joint_vectors = []
            late_vectors = []

            if 'EHR' in self.args.modalities:
                early_v_ehr, _ = self.early_ehr_encoder(x)
                joint_v_ehr, _ = self.joint_ehr_encoder(x)
                late_v_ehr, _ = self.late_ehr_encoder(x)
                early_vectors.append(early_v_ehr)
                joint_vectors.append(joint_v_ehr)
                late_vectors.append(late_v_ehr)
                
            if 'CXR' in self.args.modalities:
                early_v_cxr, _ = self.early_cxr_encoder(img)
                joint_v_cxr, _ = self.joint_cxr_encoder(img)
                late_v_cxr, _ = self.late_cxr_encoder(img)
                early_vectors.append(early_v_cxr)
                joint_vectors.append(joint_v_cxr)
                late_vectors.append(late_v_cxr)

            if 'DN' in self.args.modalities:
                early_v_dn, _ = self.early_dn_encoder(dn)
                joint_v_dn, _ = self.joint_dn_encoder(dn)
                late_v_dn, _ = self.late_dn_encoder(dn)
                early_vectors.append(early_v_dn)
                joint_vectors.append(joint_v_dn)
                late_vectors.append(late_v_dn)

            if 'RR' in self.args.modalities:
                early_v_rr, _ = self.early_rr_encoder(rr)
                joint_v_rr, _ = self.joint_rr_encoder(rr)
                late_v_rr, _ = self.late_rr_encoder(rr)
                early_vectors.append(early_v_rr)
                joint_vectors.append(joint_v_rr)
                late_vectors.append(late_v_rr)

            early_fused_vector = torch.cat(early_vectors, dim=1)
            early_fused_vector = self.early_transformer_layer(early_fused_vector)

            joint_fused_vector = torch.cat(joint_vectors, dim=1)
            joint_fused_vector = self.joint_transformer_layer(joint_fused_vector)

            late_fused_vector = torch.cat(late_vectors, dim=1)
            late_fused_vector = self.late_transformer_layer(late_fused_vector)

            final_fused_vector = torch.cat([early_fused_vector, joint_fused_vector, late_fused_vector], dim=1)
            final_fused_vector = self.final_transformer_layer(final_fused_vector)

            y_fused_pred = self.final_classifier(final_fused_vector[:, 0, :])
            
            loss = nn.BCEWithLogitsLoss()(y_fused_pred, y)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            outPRED = torch.cat((outPRED, y_fused_pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f}")
        
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
        wandb.log({
                'train_Loss': epoch_loss/i, 
                'train_AUC': ret['auroc_mean']
            })
        return ret
    
    def validate(self, dl):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
    
        with torch.no_grad():
            for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (dl):
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = x.to(self.device)
                y = y.to(self.device)
                if self.args.task == 'in-hospital-mortality':
                    y = y.unsqueeze(1)
                img = img.to(self.device)
                
                early_vectors = []
                joint_vectors = []
                late_vectors = []

                if 'EHR' in self.args.modalities:
                    early_v_ehr, _ = self.early_ehr_encoder(x)
                    joint_v_ehr, _ = self.joint_ehr_encoder(x)
                    late_v_ehr, _ = self.late_ehr_encoder(x)
                    early_vectors.append(early_v_ehr)
                    joint_vectors.append(joint_v_ehr)
                    late_vectors.append(late_v_ehr)
                    
                if 'CXR' in self.args.modalities:
                    early_v_cxr, _ = self.early_cxr_encoder(img)
                    joint_v_cxr, _ = self.joint_cxr_encoder(img)
                    late_v_cxr, _ = self.late_cxr_encoder(img)
                    early_vectors.append(early_v_cxr)
                    joint_vectors.append(joint_v_cxr)
                    late_vectors.append(late_v_cxr)

                if 'DN' in self.args.modalities:
                    early_v_dn, _ = self.early_dn_encoder(dn)
                    joint_v_dn, _ = self.joint_dn_encoder(dn)
                    late_v_dn, _ = self.late_dn_encoder(dn)
                    early_vectors.append(early_v_dn)
                    joint_vectors.append(joint_v_dn)
                    late_vectors.append(late_v_dn)

                if 'RR' in self.args.modalities:
                    early_v_rr, _ = self.early_rr_encoder(rr)
                    joint_v_rr, _ = self.joint_rr_encoder(rr)
                    late_v_rr, _ = self.late_rr_encoder(rr)
                    early_vectors.append(early_v_rr)
                    joint_vectors.append(joint_v_rr)
                    late_vectors.append(late_v_rr)

                early_fused_vector = torch.cat(early_vectors, dim=1)
                early_fused_vector = self.early_transformer_layer(early_fused_vector)

                joint_fused_vector = torch.cat(joint_vectors, dim=1)
                joint_fused_vector = self.joint_transformer_layer(joint_fused_vector)

                late_fused_vector = torch.cat(late_vectors, dim=1)
                late_fused_vector = self.late_transformer_layer(late_fused_vector)

                final_fused_vector = torch.cat([early_fused_vector, joint_fused_vector, late_fused_vector], dim=1)
                final_fused_vector = self.final_transformer_layer(final_fused_vector)

                y_fused_pred = self.final_classifier(final_fused_vector[:, 0, :])
                
                loss = nn.BCEWithLogitsLoss()(y_fused_pred, y)
                epoch_loss += loss.item()
                outPRED = torch.cat((outPRED, y_fused_pred), 0)
                outGT = torch.cat((outGT, y), 0)
    
            print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")
    
            ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy())
            np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy())
            wandb.log({
                'val_Loss': epoch_loss / i,
                'val_AUC': ret['auroc_mean']
            })
    
        return ret

    def eval(self):
        checkpoint_dir = f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}'
        self.load_fusion_checkpoint(f'{checkpoint_dir}/best_checkpoint_{self.args.H_mode}_{self.args.order}_{self.args.lr}_{self.args.task}_{self.args.data_pairs}.pth.tar')
        
        self.epoch = 0
        self.set_eval_mode() 

        ret = self.validate(self.test_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
        wandb.log({
                'test_auprc': ret['auprc_mean'], 
                'test_AUC': ret['auroc_mean']
            })
        return
    
    def train(self):
        print(f'running for fusion_type {self.args.task}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            print(self.epoch)
            self.set_eval_mode() 
            ret = self.validate(self.val_dl)
    
            if self.best_auroc < ret['auroc_mean']:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_fusion_checkpoint()
                print("checkpoint")
                self.patience = 0
            else:
                self.patience += 1
            
            if self.patience >= self.args.patience:
                break
            
            self.set_train_mode() 
            self.train_epoch()
