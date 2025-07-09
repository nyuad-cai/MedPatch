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

class TripleFusionTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl
        ):

        super(TripleFusionTrainer, self).__init__(args)
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
        
        self.ehr_encoders = [
            EHRTransformer(self.args, dim=384, depth=4, heads=4, mlp_dim=768, dropout=0.0, dim_head=128).to(self.device)
            for _ in range(3)
        ]
        self.cxr_encoders = [
            CXRTransformer(model_name='vit_small_patch16_384', image_size=384, patch_size=16, dim=384, depth=4, heads=4, mlp_dim=768, dropout=0.0, emb_dropout=0.0, dim_head=128).to(self.device)
            for _ in range(3)
        ]
        self.dn_encoders = [
            DischargeNotesEncoder(device=self.device, pretrained_model_name='allenai/longformer-base-4096', output_dim=384).to(self.device)
            for _ in range(3)
        ]
        self.rr_encoders = [
            RadiologyNotesEncoder(device=self.device, pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT', output_dim=384).to(self.device)
            for _ in range(3)
        ]
        
        if self.args.H_mode != 'triple-late':
            self.final_classifiers = [ 
                MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
                for _ in range(3)
            ]
            self.transformer_layers = [
                CustomTransformerLayer(input_dim=384 * len(self.args.modalities.split('-')), model_dim=384, nhead=4, num_layers=1).to(self.device)
                for _ in range(3)
            ]
        else:
            self.ehr_classifiers = [MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device) for _ in range(3)]
            self.cxr_classifiers = [MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device) for _ in range(3)]
            self.dn_classifiers = [MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device) for _ in range(3)]
            self.rr_classifiers = [MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device) for _ in range(3)]
        
        
        self.load_architectures()
        

        if self.args.H_mode == 'triple-early':
            # Freeze parameters of all encoders in the ensemble
            for encoders in [self.ehr_encoders, self.cxr_encoders, self.dn_encoders, self.rr_encoders]:
                for encoder in encoders:
                    for param in encoder.parameters():
                        param.requires_grad = False
            all_params = []
            for classifier in self.final_classifiers:
                all_params += list(classifier.parameters())
                
            for transformer in self.transformer_layers:
                all_params += list(transformer.parameters())
            
        elif self.args.H_mode == 'triple-joint':
            all_params = []
            for encoders in [self.ehr_encoders, self.cxr_encoders, self.dn_encoders, self.rr_encoders]:
                for encoder in encoders:
                    all_params += list(encoder.parameters())
            for classifier in self.final_classifiers:
                all_params += list(classifier.parameters())
                
            for transformer in self.transformer_layers:
                all_params += list(transformer.parameters())
        else:  # late fusion
            all_params = []
            for encoders, classifiers in zip(
                [self.ehr_encoders, self.cxr_encoders, self.dn_encoders, self.rr_encoders],
                [self.ehr_classifiers, self.cxr_classifiers, self.dn_classifiers, self.rr_classifiers]
            ):
                for encoder in encoders:
                    all_params += list(encoder.parameters())
                for classifier in classifiers:
                    all_params += list(classifier.parameters())
        
        self.optimizer = optim.Adam(all_params, lr=args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
        
        
    def load_architectures(self):
        # Define the paths to the three model checkpoints
        checkpoint_paths = [self.args.load_model_1, self.args.load_model_2, self.args.load_model_3]
    
        for arch_idx, checkpoint_path in enumerate(checkpoint_paths):
            if checkpoint_path is not None:
                checkpoint = torch.load(checkpoint_path)
                print(f"Loading architecture {arch_idx + 1} from {checkpoint_path}")
    
                # Load encoders for the current architecture
                self.ehr_encoders[arch_idx].load_state_dict(checkpoint[f'ehr_encoder_state_dict'])
                self.cxr_encoders[arch_idx].load_state_dict(checkpoint[f'cxr_encoder_state_dict'])
                self.dn_encoders[arch_idx].load_state_dict(checkpoint[f'dn_encoder_state_dict'])
                self.rr_encoders[arch_idx].load_state_dict(checkpoint[f'rr_encoder_state_dict'])
    
                # Load transformer layers and final classifier for triple-early and triple-joint modes
                if self.args.H_mode in ['triple-early', 'triple-joint']:
                    self.transformer_layers[arch_idx].load_state_dict(checkpoint[f'transformer_layer1_state_dict'])
                    self.final_classifiers[arch_idx].load_state_dict(checkpoint[f'final_classifier_state_dict'])
    
                # Load classifiers for triple-late mode
                if self.args.H_mode == 'triple-late':
                    self.ehr_classifiers[arch_idx].load_state_dict(checkpoint[f'ehr_classifier_state_dict'])
                    self.cxr_classifiers[arch_idx].load_state_dict(checkpoint[f'cxr_classifier_state_dict'])
                    self.dn_classifiers[arch_idx].load_state_dict(checkpoint[f'dn_classifier_state_dict'])
                    self.rr_classifiers[arch_idx].load_state_dict(checkpoint[f'rr_classifier_state_dict'])
    
                # # Optionally load optimizer state if needed (depending on training setup)
                # if 'optimizer_state_dict' in checkpoint:
                #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        
    def save_fusion_checkpoint(self):
        # Define the checkpoint directory path
        checkpoint_dir = f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}'
        
        # Create the directory and all intermediate-level directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
        # Initialize the checkpoint dictionary
        checkpoint = {
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
    
        # Save encoder states for all fusion modes
        for i, encoders in enumerate([self.ehr_encoders, self.cxr_encoders, self.dn_encoders, self.rr_encoders]):
            for j, encoder in enumerate(encoders):
                checkpoint[f'encoder_{i}_{j}_state_dict'] = encoder.state_dict()
    
        # Save final classifier and transformer layers only for triple-early and triple-joint modes
        if self.args.H_mode in ['triple-early', 'triple-joint']:
            for k, classifier in enumerate(self.final_classifiers):
                checkpoint[f'final_classifier_{k}_state_dict'] = classifier.state_dict()
    
            for t, transformer in enumerate(self.transformer_layers):
                checkpoint[f'transformer_layer_{t}_state_dict'] = transformer.state_dict()
    
        # Save separate classifiers for late fusion
        if self.args.H_mode == 'triple-late':
            for i, classifiers in enumerate([self.ehr_classifiers, self.cxr_classifiers, self.dn_classifiers, self.rr_classifiers]):
                for j, classifier in enumerate(classifiers):
                    checkpoint[f'classifier_{i}_{j}_state_dict'] = classifier.state_dict()
    
        # Save checkpoint to file
        torch.save(checkpoint, f'{checkpoint_dir}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.H_mode}_{self.args.order}.pth.tar')


    def load_fusion_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    
        # Load encoder states for all fusion modes
        for i, encoders in enumerate([self.ehr_encoders, self.cxr_encoders, self.dn_encoders, self.rr_encoders]):
            for j, encoder in enumerate(encoders):
                encoder.load_state_dict(checkpoint[f'encoder_{i}_{j}_state_dict'])
    
        # Load final classifier and transformer layers only for triple-early and triple-joint modes
        if self.args.H_mode in ['triple-early', 'triple-joint']:
            for k, classifier in enumerate(self.final_classifiers):
                classifier.load_state_dict(checkpoint[f'final_classifier_{k}_state_dict'])
    
            for t, transformer in enumerate(self.transformer_layers):
                transformer.load_state_dict(checkpoint[f'transformer_layer_{t}_state_dict'])
    
        # Load separate classifiers for late fusion
        if self.args.H_mode == 'triple-late':
            for i, classifiers in enumerate([self.ehr_classifiers, self.cxr_classifiers, self.dn_classifiers, self.rr_classifiers]):
                for j, classifier in enumerate(classifiers):
                    classifier.load_state_dict(checkpoint[f'classifier_{i}_{j}_state_dict'])
    
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        
    def set_train_mode(self):
        """Set all neural network components to training mode."""
        
        # Set all encoders to training mode for all fusion types
        for encoders in [self.ehr_encoders, self.cxr_encoders, self.dn_encoders, self.rr_encoders]:
            for encoder in encoders:
                encoder.train()
    
        # Set final classifiers and transformers to training mode for triple-early and triple-joint
        if self.args.H_mode in ['triple-early', 'triple-joint']:
            for classifier in self.final_classifiers:
                classifier.train()
            for transformer in self.transformer_layers:
                transformer.train()
    
        # Set separate classifiers to training mode for triple-late
        if self.args.H_mode == 'triple-late':
            for classifiers in [self.ehr_classifiers, self.cxr_classifiers, self.dn_classifiers, self.rr_classifiers]:
                for classifier in classifiers:
                    classifier.train()


    def set_eval_mode(self):
        """Set all neural network components to evaluation mode."""
        
        # Set all encoders to evaluation mode for all fusion types
        for encoders in [self.ehr_encoders, self.cxr_encoders, self.dn_encoders, self.rr_encoders]:
            for encoder in encoders:
                encoder.eval()
    
        # Set final classifiers and transformers to evaluation mode for triple-early and triple-joint
        if self.args.H_mode in ['triple-early', 'triple-joint']:
            for classifier in self.final_classifiers:
                classifier.eval()
            for transformer in self.transformer_layers:
                transformer.eval()
    
        # Set separate classifiers to evaluation mode for triple-late
        if self.args.H_mode == 'triple-late':
            for classifiers in [self.ehr_classifiers, self.cxr_classifiers, self.dn_classifiers, self.rr_classifiers]:
                for classifier in classifiers:
                    classifier.eval()
    
    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
    
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float().to(self.device)
            y = y.to(self.device)
            if self.args.task == 'in-hospital-mortality':
                y = y.unsqueeze(1)
            img = img.to(self.device)
    
            # Prepare lists to collect outputs from all three models
            y_preds_all_architectures = []
    
            # Go through each of the 3 architectures independently
            for arch_idx in range(3):
                vectors = []
                y_preds = []
    
                # Process the encoders for each modality in the ensemble
                if 'EHR' in self.args.modalities:
                    v_ehr, cls_ehr = self.ehr_encoders[arch_idx](x)
                    vectors.append(v_ehr)
                    if self.args.H_mode == 'triple-late':
                        y_ehr_pred = self.ehr_classifiers[arch_idx](cls_ehr)
                        y_preds.append(y_ehr_pred)
    
                if 'CXR' in self.args.modalities:
                    v_cxr, cls_cxr = self.cxr_encoders[arch_idx](img)
                    vectors.append(v_cxr)
                    if self.args.H_mode == 'triple-late':
                        y_cxr_pred = self.cxr_classifiers[arch_idx](cls_cxr)
                        y_preds.append(y_cxr_pred)
    
                if 'DN' in self.args.modalities:
                    v_dn, cls_dn = self.dn_encoders[arch_idx](dn)
                    vectors.append(v_dn)
                    if self.args.H_mode == 'triple-late':
                        y_dn_pred = self.dn_classifiers[arch_idx](cls_dn)
                        y_preds.append(y_dn_pred)
    
                if 'RR' in self.args.modalities:
                    v_rr, cls_rr = self.rr_encoders[arch_idx](rr)
                    vectors.append(v_rr)
                    if self.args.H_mode == 'triple-late':
                        y_rr_pred = self.rr_classifiers[arch_idx](cls_rr)
                        y_preds.append(y_rr_pred)
    
                # Handle the different fusion modes
                if self.args.H_mode == 'triple-late':
                    # Late fusion: average predictions from this architecture's classifiers
                    y_fused_pred = torch.mean(torch.stack(y_preds, dim=0), dim=0)
                else:
                    # Early and Joint fusion: concatenate vectors and pass through transformers
                    fused_vector = torch.cat(vectors, dim=1)
                    fused_vector = self.transformer_layers[arch_idx](fused_vector)
                    y_fused_pred = self.final_classifiers[arch_idx](fused_vector[:, 0, :])
    
                # Collect the prediction for this architecture
                y_preds_all_architectures.append(y_fused_pred)
    
            # Average predictions across the 3 architectures
            y_fused_pred = torch.mean(torch.stack(y_preds_all_architectures, dim=0), dim=0)
    
            # Compute the loss and backpropagate
            loss = nn.BCEWithLogitsLoss()(y_fused_pred, y)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
            # Append predictions and ground truth for performance evaluation
            outPRED = torch.cat((outPRED, y_fused_pred), 0)
            outGT = torch.cat((outGT, y), 0)
    
            # Log training progress
            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f}")
    
        # Compute performance metrics
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
        wandb.log({
            'train_Loss': epoch_loss / i, 
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
                
                # Prepare lists to collect outputs from all three models
                y_preds_all_architectures = []
        
                # Go through each of the 3 architectures independently
                for arch_idx in range(3):
                    vectors = []
                    y_preds = []
        
                    # Process the encoders for each modality in the ensemble
                    if 'EHR' in self.args.modalities:
                        v_ehr, cls_ehr = self.ehr_encoders[arch_idx](x)
                        vectors.append(v_ehr)
                        if self.args.H_mode == 'triple-late':
                            y_ehr_pred = self.ehr_classifiers[arch_idx](cls_ehr)
                            y_preds.append(y_ehr_pred)
        
                    if 'CXR' in self.args.modalities:
                        v_cxr, cls_cxr = self.cxr_encoders[arch_idx](img)
                        vectors.append(v_cxr)
                        if self.args.H_mode == 'triple-late':
                            y_cxr_pred = self.cxr_classifiers[arch_idx](cls_cxr)
                            y_preds.append(y_cxr_pred)
        
                    if 'DN' in self.args.modalities:
                        v_dn, cls_dn = self.dn_encoders[arch_idx](dn)
                        vectors.append(v_dn)
                        if self.args.H_mode == 'triple-late':
                            y_dn_pred = self.dn_classifiers[arch_idx](cls_dn)
                            y_preds.append(y_dn_pred)
        
                    if 'RR' in self.args.modalities:
                        v_rr, cls_rr = self.rr_encoders[arch_idx](rr)
                        vectors.append(v_rr)
                        if self.args.H_mode == 'triple-late':
                            y_rr_pred = self.rr_classifiers[arch_idx](cls_rr)
                            y_preds.append(y_rr_pred)
        
                    # Handle the different fusion modes
                    if self.args.H_mode == 'triple-late':
                        # Late fusion: average predictions from this architecture's classifiers
                        y_fused_pred = torch.mean(torch.stack(y_preds, dim=0), dim=0)
                    else:
                        # Early and Joint fusion: concatenate vectors and pass through transformers
                        fused_vector = torch.cat(vectors, dim=1)
                        fused_vector = self.transformer_layers[arch_idx](fused_vector)
                        y_fused_pred = self.final_classifiers[arch_idx](fused_vector[:, 0, :])
        
                    # Collect the prediction for this architecture
                    y_preds_all_architectures.append(y_fused_pred)
        
                # Average predictions across the 3 architectures
                y_fused_pred = torch.mean(torch.stack(y_preds_all_architectures, dim=0), dim=0)
                
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
        self.load_fusion_checkpoint(f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.H_mode}_{self.args.order}.pth.tar')
        
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
        print(f'running for fusion_type {self.args.H_mode}')
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
