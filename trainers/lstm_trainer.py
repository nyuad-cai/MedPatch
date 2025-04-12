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
import torch.nn.utils.rnn as rnn_utils

class LSTMFusionTrainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl):
        super(LSTMFusionTrainer, self).__init__(args)
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

        # Define encoders for each modality
        self.ehr_encoder = EHRTransformer(self.args, dim=384, depth=4, heads=4, mlp_dim=768, dropout=0.0, dim_head=128).to(self.device)
        self.cxr_encoder = CXRTransformer(model_name='vit_small_patch16_384', image_size=384, patch_size=16, dim=384, depth=4, heads=4, mlp_dim=768, dropout=0.0, emb_dropout=0.0, dim_head=128).to(self.device)
        self.dn_encoder = DischargeNotesEncoder(device=self.device, pretrained_model_name='allenai/longformer-base-4096', output_dim=384).to(self.device)
        self.rr_encoder = RadiologyNotesEncoder(device=self.device, pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT', output_dim=384).to(self.device)

        # Define LSTM layer for fusion
        lstm_in_dim = 384  # Assuming each modality has an output dim of 384
        self.lstm_fusion_layer = nn.LSTM(input_size=lstm_in_dim, hidden_size=512, num_layers=1, batch_first=True, dropout=0.0).to(self.device)

        # Final classifier that takes the output of the LSTM and produces the prediction
        self.final_classifier = MLPClassifier(input_dim=512, output_dim=self.args.num_classes).to(self.device)

        if self.args.load_ehr:
            checkpoint = torch.load(self.args.load_ehr)
            self.ehr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("EHR encoder loaded")
            
        if self.args.load_cxr:
            checkpoint = torch.load(self.args.load_cxr)
            self.cxr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("CXR encoder loaded")

        if self.args.load_dn:
            checkpoint = torch.load(self.args.load_dn)
            self.dn_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("Discharge Notes encoder loaded")

        if self.args.load_rr:
            checkpoint = torch.load(self.args.load_rr)
            self.rr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("Radiology Reports encoder loaded")

        # Set optimizer and scheduler
        all_params = (
            list(self.ehr_encoder.parameters()) +
            list(self.cxr_encoder.parameters()) +
            list(self.dn_encoder.parameters()) +
            list(self.rr_encoder.parameters()) +
            list(self.lstm_fusion_layer.parameters()) +
            list(self.final_classifier.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
        
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
            'lstm_fusion_layer_dict': self.lstm_fusion_layer.state_dict(),
            'final_classifier_state_dict': self.final_classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.H_mode}_{self.args.order}.pth.tar')

    def load_fusion_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        self.ehr_encoder.load_state_dict(checkpoint['ehr_encoder_state_dict'])
        self.cxr_encoder.load_state_dict(checkpoint['cxr_encoder_state_dict'])
        self.dn_encoder.load_state_dict(checkpoint['dn_encoder_state_dict'])
        self.rr_encoder.load_state_dict(checkpoint['rr_encoder_state_dict'])
        self.lstm_fusion_layer.load_state_dict(checkpoint['lstm_fusion_layer_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def set_train_mode(self):
        """Set all neural network components to training mode."""
        self.ehr_encoder.train()
        self.cxr_encoder.train()
        self.dn_encoder.train()
        self.rr_encoder.train()
        self.lstm_fusion_layer.train()
        self.final_classifier.train()

    def set_eval_mode(self):
        """Set all neural network components to evaluation mode."""
        self.ehr_encoder.eval()
        self.cxr_encoder.eval()
        self.dn_encoder.eval()
        self.rr_encoder.eval()
        self.lstm_fusion_layer.eval()
        self.final_classifier.eval()
        

    def fuse_with_lstm(self, vectors):
        """
        Fuse modality vectors using LSTM. Handles variable sequence lengths by padding.
        """
        # Get the batch size and feature dimension
        batch_size = vectors[0].size(0)
        feature_dim = vectors[0].size(2)
    
        # Find the max sequence length across all modalities
        max_seq_len = max(v.size(1) for v in vectors)
    
        # Pad each modality vector to the max sequence length
        padded_vectors = []
        for v in vectors:
            seq_len = v.size(1)
            if seq_len < max_seq_len:
                # Pad along the sequence length dimension (dim=1)
                padding = (0, 0, 0, max_seq_len - seq_len)  # Only pad the sequence length dimension
                padded_v = F.pad(v, padding)
            else:
                padded_v = v
            padded_vectors.append(padded_v)
    
        # Stack the padded vectors along the modality dimension (dim=1)
        feats = torch.stack(padded_vectors, dim=1)  # Shape: [batch_size, num_modalities, max_seq_len, feature_dim]
    
        # Reshape or permute the tensor for the LSTM input if necessary
        # For example, you may need to combine modality and sequence dimensions for LSTM
        feats = feats.view(batch_size, -1, feature_dim)  # Shape: [batch_size, num_modalities * max_seq_len, feature_dim]
    
        # Pass the sequence through LSTM
        lstm_out, (ht, _) = self.lstm_fusion_layer(feats)  # ht is the final hidden state of the LSTM
    
        return ht.squeeze(0)  # Shape: [batch_size, hidden_size]


    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)

        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            if self.args.task == 'in-hospital-mortality':
                y = y.unsqueeze(1)
            img = img.to(self.device)

            # Vectors to fuse
            vectors = []

            if 'EHR' in self.args.modalities:
                v_ehr, _ = self.ehr_encoder(x)
                vectors.append(v_ehr)
            if 'CXR' in self.args.modalities:
                v_cxr, _ = self.cxr_encoder(img)
                vectors.append(v_cxr)
            if 'DN' in self.args.modalities:
                v_dn, _ = self.dn_encoder(dn)
                vectors.append(v_dn)
            if 'RR' in self.args.modalities:
                v_rr, _ = self.rr_encoder(rr)
                vectors.append(v_rr)

            # Fuse vectors using LSTM
            fused_vector = self.fuse_with_lstm(vectors)

            # Final classification
            y_fused_pred = self.final_classifier(fused_vector)

            # Calculate loss
            loss = nn.BCEWithLogitsLoss()(y_fused_pred, y)
            epoch_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            outPRED = torch.cat((outPRED, y_fused_pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f"epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20} lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f}")

        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
        wandb.log({'train_Loss': epoch_loss/i, 'train_AUC': ret['auroc_mean']})
        return ret

    def validate(self, dl):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(dl):
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = x.to(self.device)
                y = y.to(self.device)
                if self.args.task == 'in-hospital-mortality':
                    y = y.unsqueeze(1)
                img = img.to(self.device)

                # Vectors to fuse
                vectors = []

                if 'EHR' in self.args.modalities:
                    v_ehr, _ = self.ehr_encoder(x)
                    vectors.append(v_ehr)
                if 'CXR' in self.args.modalities:
                    v_cxr, _ = self.cxr_encoder(img)
                    vectors.append(v_cxr)
                if 'DN' in self.args.modalities:
                    v_dn, _ = self.dn_encoder(dn)
                    vectors.append(v_dn)
                if 'RR' in self.args.modalities:
                    v_rr, _ = self.rr_encoder(rr)
                    vectors.append(v_rr)

                # Fuse vectors using LSTM
                fused_vector = self.fuse_with_lstm(vectors)

                # Final classification
                y_fused_pred = self.final_classifier(fused_vector)

                # Calculate loss
                loss = nn.BCEWithLogitsLoss()(y_fused_pred, y)
                epoch_loss += loss.item()

                outPRED = torch.cat((outPRED, y_fused_pred), 0)
                outGT = torch.cat((outGT, y), 0)

            print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")

            ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            wandb.log({'val_Loss': epoch_loss / i, 'val_AUC': ret['auroc_mean']})

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
