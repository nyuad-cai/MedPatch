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

class StagedFusionTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl
        ):

        super(StagedFusionTrainer, self).__init__(args)
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
        
        self.ehr_encoder = EHRTransformer(self.args,
            dim=384,
            depth=4,
            heads=4,
            mlp_dim=768,
            dropout=0.0,
            dim_head=128
        ).to(self.device)
        self.cxr_encoder = CXRTransformer(
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
        self.dn_encoder = DischargeNotesEncoder(device= self.device,
            pretrained_model_name='allenai/longformer-base-4096',
            output_dim=384
        ).to(self.device)
        self.rr_encoder = RadiologyNotesEncoder(device= self.device, 
            pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT',
            output_dim=384
        ).to(self.device)
        
        self.final_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        
        # Initialize transformer layers
        self.transformer_layer1 = CustomTransformerLayer(input_dim=384 * len(self.args.modalities.split('-')), model_dim=384, nhead=4, num_layers=4).to(self.device)
        self.transformer_layer2 = CustomTransformerLayer(input_dim=384, model_dim=384, nhead=4, num_layers=1).to(self.device)
        self.transformer_layer3 = CustomTransformerLayer(input_dim=384, model_dim=384, nhead=4, num_layers=1).to(self.device)
        self.transformer_layer4 = CustomTransformerLayer(input_dim=384, model_dim=384, nhead=4, num_layers=1).to(self.device)
        
        if self.args.load_ehr:
            checkpoint = torch.load(self.args.load_ehr)
            self.ehr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("ehr loaded")
            
        if self.args.load_cxr:
            checkpoint = torch.load(self.args.load_cxr)
            self.cxr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("cxr loaded")
        
        if self.args.load_dn:
            checkpoint = torch.load(self.args.load_dn)
            self.dn_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("dn loaded")
        
        if self.args.load_rr:
            checkpoint = torch.load(self.args.load_rr)
            self.rr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("rr loaded")

        if self.args.H_mode == 'early':
            for param in self.ehr_encoder.parameters():
                param.requires_grad = False
            for param in self.cxr_encoder.parameters():
                param.requires_grad = False
            for param in self.dn_encoder.parameters():
                param.requires_grad = False
            for param in self.rr_encoder.parameters():
                param.requires_grad = False

        if self.args.H_mode == 'early':
            all_params = (
                list(self.transformer_layer1.parameters()) +
                list(self.transformer_layer2.parameters()) +
                list(self.transformer_layer3.parameters()) +
                list(self.transformer_layer4.parameters()) +
                list(self.final_classifier.parameters())
            )
        elif self.args.H_mode == 'joint':
            all_params = (
                list(self.ehr_encoder.parameters()) +
                list(self.cxr_encoder.parameters()) +
                list(self.dn_encoder.parameters()) +
                list(self.rr_encoder.parameters()) +
                list(self.transformer_layer1.parameters()) +
                list(self.transformer_layer2.parameters()) +
                list(self.transformer_layer3.parameters()) +
                list(self.transformer_layer4.parameters()) +
                list(self.final_classifier.parameters())
            )
        else:  # late fusion
            self.ehr_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
            self.cxr_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
            self.dn_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
            self.rr_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
            all_params = (
                list(self.ehr_encoder.parameters()) +
                list(self.cxr_encoder.parameters()) +
                list(self.dn_encoder.parameters()) +
                list(self.rr_encoder.parameters()) +
                list(self.ehr_classifier.parameters()) +
                list(self.cxr_classifier.parameters()) +
                list(self.dn_classifier.parameters()) +
                list(self.rr_classifier.parameters()) +
                list(self.final_classifier.parameters())
            )
        
        self.optimizer = optim.Adam(all_params, lr=args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
        
        self.print_model_parameters()
        
    def print_model_parameters(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        print(f"Number of parameters in EHR Transformer: {count_parameters(self.ehr_encoder)}")
        print(f"Number of parameters in CXR Transformer: {count_parameters(self.cxr_encoder)}")
        print(f"Number of parameters in DN Encoder: {count_parameters(self.dn_encoder)}")
        print(f"Number of parameters in RR Encoder: {count_parameters(self.rr_encoder)}")
        print(f"Number of parameters in 4-layer transformer Encoder: {count_parameters(self.transformer_layer1)}")
        print(f"Number of parameters in 1-layer transformer Encoder: {count_parameters(self.transformer_layer2)}")
        
        print(f"Number of parameters in Final Classifier: {count_parameters(self.final_classifier)}")

        # total_params = sum([
        #     count_parameters(self.ehr_encoder),
        #     count_parameters(self.cxr_encoder),
        #     count_parameters(self.dn_encoder),
        #     count_parameters(self.rr_encoder),
        #     total_transformer_layers_params,
        #     count_parameters(self.final_classifier),
        # ])

        # print(f"Total number of parameters: {total_params}")
        
        

        
    def save_fusion_checkpoint(self):
        # Define the checkpoint directory path
        checkpoint_dir = f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}'
        
        # Create the directory and all intermediate-level directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'ehr_encoder_state_dict': self.ehr_encoder.state_dict(),
            'cxr_encoder_state_dict': self.cxr_encoder.state_dict(),
            'dn_encoder_state_dict': self.dn_encoder.state_dict(),
            'rr_encoder_state_dict': self.rr_encoder.state_dict(),
            'transformer_layer1_state_dict': self.transformer_layer1.state_dict(),
            'transformer_layer2_state_dict': self.transformer_layer2.state_dict(),
            'transformer_layer3_state_dict': self.transformer_layer3.state_dict(),
            'transformer_layer4_state_dict': self.transformer_layer4.state_dict(),
            'final_classifier_state_dict': self.final_classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.args.H_mode == 'late':
            checkpoint.update({
                'ehr_classifier_state_dict': self.ehr_classifier.state_dict(),
                'cxr_classifier_state_dict': self.cxr_classifier.state_dict(),
                'dn_classifier_state_dict': self.dn_classifier.state_dict(),
                'rr_classifier_state_dict': self.rr_classifier.state_dict(),
            })
        torch.save(checkpoint, f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.H_mode}_{self.args.order}.pth.tar')

    def load_fusion_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        self.ehr_encoder.load_state_dict(checkpoint['ehr_encoder_state_dict'])
        self.cxr_encoder.load_state_dict(checkpoint['cxr_encoder_state_dict'])
        self.dn_encoder.load_state_dict(checkpoint['dn_encoder_state_dict'])
        self.rr_encoder.load_state_dict(checkpoint['rr_encoder_state_dict'])
        self.transformer_layer1.load_state_dict(checkpoint['transformer_layer1_state_dict'])
        self.transformer_layer2.load_state_dict(checkpoint['transformer_layer2_state_dict'])
        self.transformer_layer3.load_state_dict(checkpoint['transformer_layer3_state_dict'])
        self.transformer_layer4.load_state_dict(checkpoint['transformer_layer4_state_dict'])
        self.final_classifier.load_state_dict(checkpoint['final_classifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.args.H_mode == 'late':
            self.ehr_classifier.load_state_dict(checkpoint['ehr_classifier_state_dict'])
            self.cxr_classifier.load_state_dict(checkpoint['cxr_classifier_state_dict'])
            self.dn_classifier.load_state_dict(checkpoint['dn_classifier_state_dict'])
            self.rr_classifier.load_state_dict(checkpoint['rr_classifier_state_dict'])
        
    def set_train_mode(self):
        """Set all neural network components to training mode."""
        self.ehr_encoder.train()
        self.cxr_encoder.train()
        self.dn_encoder.train()
        self.rr_encoder.train()
        self.transformer_layer1.train()
        self.transformer_layer2.train()
        self.transformer_layer3.train()
        self.transformer_layer4.train()
        self.final_classifier.train()
        if self.args.H_mode == 'late':
            self.ehr_classifier.train()
            self.cxr_classifier.train()
            self.dn_classifier.train()
            self.rr_classifier.train()

    def set_eval_mode(self):
        """Set all neural network components to evaluation mode."""
        self.ehr_encoder.eval()
        self.cxr_encoder.eval()
        self.dn_encoder.eval()
        self.rr_encoder.eval()
        self.transformer_layer1.eval()
        self.transformer_layer2.eval()
        self.transformer_layer3.eval()
        self.transformer_layer4.eval()
        self.final_classifier.eval()
        if self.args.H_mode == 'late':
            self.ehr_classifier.eval()
            self.cxr_classifier.eval()
            self.dn_classifier.eval()
            self.rr_classifier.eval()
    
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
            
            vectors = []
            y_preds = []

            if 'EHR' in self.args.modalities:
                v_ehr, cls_ehr = self.ehr_encoder(x)
                vectors.append(v_ehr)
                if self.args.H_mode == 'late':
                    y_ehr_pred = self.ehr_classifier(cls_ehr)
                    y_preds.append(y_ehr_pred)
            if 'CXR' in self.args.modalities:
                v_cxr, cls_cxr = self.cxr_encoder(img)
                vectors.append(v_cxr)
                if self.args.H_mode == 'late':
                    y_cxr_pred = self.cxr_classifier(cls_cxr)
                    y_preds.append(y_cxr_pred)
            if 'DN' in self.args.modalities:
                v_dn, cls_dn = self.dn_encoder(dn)
                vectors.append(v_dn)
                if self.args.H_mode == 'late':
                    y_dn_pred = self.dn_classifier(cls_dn)
                    y_preds.append(y_dn_pred)
            if 'RR' in self.args.modalities:
                v_rr, cls_rr = self.rr_encoder(rr)
                vectors.append(v_rr)
                if self.args.H_mode == 'late':
                    y_rr_pred = self.rr_classifier(cls_rr)
                    y_preds.append(y_rr_pred)

            if self.args.H_mode == 'late':
                y_fused_pred = torch.mean(torch.stack(y_preds, dim=0), dim=0)
            else:
                fused_vector = torch.cat(vectors, dim=1)
                fused_vector = self.transformer_layer1(fused_vector)
                # fused_vector = self.transformer_layer2(fused_vector)
                # fused_vector = self.transformer_layer3(fused_vector)
                # fused_vector = self.transformer_layer4(fused_vector)

                y_fused_pred = self.final_classifier(fused_vector[:, 0, :])
            
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
                
                vectors = []
                y_preds = []

                if 'EHR' in self.args.modalities:
                    v_ehr, cls_ehr = self.ehr_encoder(x)
                    vectors.append(v_ehr)
                    if self.args.H_mode == 'late':
                        y_ehr_pred = self.ehr_classifier(cls_ehr)
                        y_preds.append(y_ehr_pred)
                if 'CXR' in self.args.modalities:
                    v_cxr, cls_cxr = self.cxr_encoder(img)
                    vectors.append(v_cxr)
                    if self.args.H_mode == 'late':
                        y_cxr_pred = self.cxr_classifier(cls_cxr)
                        y_preds.append(y_cxr_pred)
                if 'DN' in self.args.modalities:
                    v_dn, cls_dn = self.dn_encoder(dn)
                    vectors.append(v_dn)
                    if self.args.H_mode == 'late':
                        y_dn_pred = self.dn_classifier(cls_dn)
                        y_preds.append(y_dn_pred)
                if 'RR' in self.args.modalities:
                    v_rr, cls_rr = self.rr_encoder(rr)
                    vectors.append(v_rr)
                    if self.args.H_mode == 'late':
                        y_rr_pred = self.rr_classifier(cls_rr)
                        y_preds.append(y_rr_pred)

                if self.args.H_mode == 'late':
                    y_fused_pred = torch.mean(torch.stack(y_preds, dim=0), dim=0)
                else:
                    fused_vector = torch.cat(vectors, dim=1)
                    fused_vector = self.transformer_layer1(fused_vector)
                    # fused_vector = self.transformer_layer2(fused_vector)
                    # fused_vector = self.transformer_layer3(fused_vector)
                    # fused_vector = self.transformer_layer4(fused_vector)

                    y_fused_pred = self.final_classifier(fused_vector[:, 0, :])
                
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
