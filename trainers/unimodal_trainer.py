from __future__ import absolute_import
from __future__ import print_function

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

class UnimodalTrainer(Trainer):
    def __init__(self, 
                 train_dl, 
                 val_dl, 
                 args,
                 test_dl):
        super(UnimodalTrainer, self).__init__(args)
        run = wandb.init(project=f'Unimodal_{self.args.pretraining}_{self.args.task}', config=args)
        self.epoch = 0 
        self.start_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
        print("train_dl:", len(self.train_dl))
        print("val_dl:", len(self.val_dl))
        print("test_dl:", len(self.test_dl))
        
        
        self.modality = args.pretraining
        self.token_dim = 384
        
        self.seed = 1002
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        if self.modality == 'EHR':
            self.encoder = EHRTransformer(self.args,
                dim=384,
                depth=4,
                heads=4,
                mlp_dim=768,
                dropout=0.0,
                dim_head=128
            ).to(self.device)
            self.classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        elif self.modality == 'CXR':
            self.encoder = CXRTransformer(
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
            self.classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        elif self.modality == 'DN':
            self.encoder = DischargeNotesEncoder(device=self.device,
                pretrained_model_name='allenai/longformer-base-4096',
                output_dim=384
            ).to(self.device)
            self.classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        elif self.modality == 'RR':
            self.encoder = RadiologyNotesEncoder(device=self.device, 
                pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT',
                output_dim=384
            ).to(self.device)
            self.classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()), 
            lr=args.lr, 
            betas=(0.9, self.args.beta_1)
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None

    def save_unimodal_checkpoint(self):
        # Define the checkpoint directory path
        checkpoint_dir = f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}'
        
        # Create the directory and all intermediate-level directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}/best_checkpoint_unimodal_{self.args.lr}_{self.args.task}_{self.args.pretraining}_{self.args.order}.pth.tar')

    def load_unimodal_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def set_train_mode(self):
        """Set all neural network components to training mode."""
        self.encoder.train()
        self.classifier.train()

    def set_eval_mode(self):
        """Set all neural network components to evaluation mode."""
        self.encoder.eval()
        self.classifier.eval()

    def train_epoch(self):
        print(f'Starting train epoch {self.epoch}')
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        age_list = []
        gender_list = []
        ethnicity_list = []
        steps = len(self.train_dl)
            
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, age, gender, ethnicity) in enumerate (self.train_dl):
            #print("seq_lengths:",seq_lengths)
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            if self.args.task == 'in-hospital-mortality':
                y = y.unsqueeze(1)
            img = img.to(self.device)
            
            
            if self.modality == 'EHR':
                mod=x
            elif self.modality == 'CXR':
                mod=img
            elif self.modality == 'DN':
                mod=dn
            elif self.modality == 'RR':
                mod=rr
            
            v, cls = self.encoder(mod)
            y_pred = self.classifier(cls)
            loss = self.loss(y_pred, y)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            outPRED = torch.cat((outPRED, y_pred), 0)
            outGT = torch.cat((outGT, y), 0)
            age_list.extend(age.cpu().numpy())
            gender_list.extend(gender.cpu().numpy())
            ethnicity_list.extend(ethnicity.cpu().numpy())

            if i % 100 == 99:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f}")
        
        if self.args.task != 'in-hospital-mortality':
            ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
            wandb.log({
            'train_Loss': epoch_loss / i,
            'train_AUC': ret['auroc_mean'],
            'train_AUPRC': ret['auprc_mean']
        })
        else:
            ret = self.compute_unimodal_AUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), age_list, gender_list, ethnicity_list, prefix='train_')
            wandb.log({
            'train_Loss': epoch_loss / i,
            'train_AUC': ret['auroc_mean'],
            'train_AUPRC': ret['auprc_mean'],
            **ret['age_aucs'],
            **ret['gender_aucs'],
            **ret['ethnicity_aucs'],
            **ret['age_auprcs'],
            **ret['gender_auprcs'],
            **ret['ethnicity_auprcs']
        })
        np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy())
        np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy())
        
        return ret

    def validate(self, dl):
        print(f'Starting val epoch {self.epoch}')
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        age_list = []
        gender_list = []
        ethnicity_list = []
    
        with torch.no_grad():
            for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, age, gender, ethnicity) in enumerate (dl):
                #print("seq_lengths:",seq_lengths)
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = x.to(self.device)
                y = y.to(self.device)
                if self.args.task == 'in-hospital-mortality':
                    y = y.unsqueeze(1)
                img = img.to(self.device)
                
                #print(len(img))
                
                if self.modality == 'EHR':
                    mod=x
                elif self.modality == 'CXR':
                    mod=img
                elif self.modality == 'DN':
                    mod=dn
                elif self.modality == 'RR':
                    mod=rr
                
                v, cls = self.encoder(mod)
                y_pred = self.classifier(cls)
                loss = self.loss(y_pred, y)
                epoch_loss += loss.item()
                outPRED = torch.cat((outPRED, y_pred), 0)
                outGT = torch.cat((outGT, y), 0)
                age_list.extend(age.cpu().numpy())
                gender_list.extend(gender.cpu().numpy())
                ethnicity_list.extend(ethnicity.cpu().numpy())
    
            print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")
    
            
            if self.args.task != 'in-hospital-mortality':
                ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
                wandb.log({
                'val_Loss': epoch_loss / i,
                'val_AUC': ret['auroc_mean'],
                'val_AUPRC': ret['auprc_mean']
            })
            else:
                ret = self.compute_unimodal_AUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), age_list, gender_list, ethnicity_list, prefix='val_')
                wandb.log({
                'val_Loss': epoch_loss / i,
                'val_AUC': ret['auroc_mean'],
                'val_AUPRC': ret['auprc_mean'],
                **ret['age_aucs'],
                **ret['gender_aucs'],
                **ret['ethnicity_aucs'],
                **ret['age_auprcs'],
                **ret['gender_auprcs'],
                **ret['ethnicity_auprcs']
            })
            np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy())
            np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy())
            
    
        return ret

    def eval(self):
        self.load_unimodal_checkpoint(f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}/best_checkpoint_unimodal_{self.args.lr}_{self.args.task}_{self.args.pretraining}_{self.args.order}.pth.tar')
        
        self.epoch = 0
        self.set_eval_mode() 
    
        ret = self.validate(self.test_dl)
        self.print_and_write(ret, isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
        
        log_data = {
            'test_auprc': ret['auprc_mean'], 
            'test_AUC': ret['auroc_mean'],
        }
    
        # Add individual AUROC and AUPRC values for age, gender, and ethnicity
        if self.args.task == 'in-hospital-mortality':
            for key, value in ret['age_aucs'].items():
                log_data[f'test_{key}'] = value
            for key, value in ret['gender_aucs'].items():
                log_data[f'test_{key}'] = value
            for key, value in ret['ethnicity_aucs'].items():
                log_data[f'test_{key}'] = value
            for key, value in ret['age_auprcs'].items():
                log_data[f'test_{key}'] = value
            for key, value in ret['gender_auprcs'].items():
                log_data[f'test_{key}'] = value
            for key, value in ret['ethnicity_auprcs'].items():
                log_data[f'test_{key}'] = value
    
        wandb.log(log_data)
        return

    
    def train(self):
        print(f'Running unimodal pretraining for modality {self.args.pretraining}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            print(self.epoch)
            self.set_eval_mode() 
            ret = self.validate(self.val_dl)
    
            if self.best_auroc < ret['auroc_mean']:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_unimodal_checkpoint()
                print("checkpoint")
                self.patience = 0
            else:
                self.patience += 1
            
            if self.patience >= self.args.patience:
                break
            
            self.set_train_mode() 
            self.train_epoch()
