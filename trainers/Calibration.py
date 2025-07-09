from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.fusion import Fusion
from models.ehr_models import EHR_encoder
from models.CXR_models import CXR_encoder
from models.text_models import Text_encoder
from models.loss_set import Loss

from .trainer import Trainer
import pandas as pd
import os

import numpy as np
from sklearn import metrics
import wandb
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

class calibration(Trainer):
    """
    Minimal trainer code specialized for the 'c-unimodal' case.
    Removes all logic and branches associated with other fusion types (e.g., c-msma).
    """

    def __init__(self, train_dl, val_dl, args, test_dl):
        super(calibration, self).__init__(args)
        run = wandb.init(project=f'MSMA_Calibrate_{self.args.fusion_type}_{self.args.task}', config=args)
        
        self.epoch = 0 
        self.start_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.seed = 379647
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
        # Encoders: Keep these since c-unimodal may still need one or more encoders
        self.ehr_encoder = None
        self.cxr_encoder = None
        self.text_encoder = None
        
        if 'EHR' in args.modalities and args.ehr_encoder is not None:
            self.ehr_encoder = EHR_encoder(args)
        if 'CXR' in args.modalities and args.cxr_encoder is not None:
            self.cxr_encoder = CXR_encoder(args)
        if ('RR' in args.modalities or 'DN' in args.modalities) and args.text_encoder is not None:
            self.text_encoder = Text_encoder(args, self.device)
            
        self.model = Fusion(args, self.ehr_encoder, self.cxr_encoder, self.text_encoder).to(self.device)
        
        # Load any existing checkpoint if provided
        self.load_model(args)  
        
        # For c-unimodal, we track minimum loss (rather than max AUROC)
        # as indicated in the original snippet:
        self.best_auroc = float('inf')  # used as "best loss" for c-unimodal
        self.best_stats = None
        
        self.loss = Loss(args)
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

    def pad_to_length(self, tensor, max_len=2646):
        # tensor shape: [batch_size, token_dim, num_classes]
        batch_size = tensor.size(0)
        curr_len = tensor.size(1)
        num_classes = tensor.size(2)
    
        # If already the desired size, just return
        if curr_len == max_len:
            return tensor
    
        # Otherwise, create a new zero tensor, then copy
        padded = torch.zeros(
            (batch_size, max_len, num_classes), 
             dtype=tensor.dtype, 
             device=tensor.device
        )
        padded[:, :curr_len, :] = tensor
        return padded

    
    def train_epoch(self, inference = False):
        print(f"Starting train epoch {self.epoch}")
        print(f"Starting {'inference' if inference else 'training'} epoch {self.epoch}")
        self.model.train(not inference)  # Set model to eval mode for inference
    
        outGT = []
        outPRED = []
        outPROB = []
        epoch_loss = 0
        
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, *_ ) in enumerate(self.val_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs, rr, dn)
            pred = output[self.args.fusion_type].squeeze()
            probs = torch.sigmoid(pred)
            
            if 'c-unimodal' in self.args.fusion_type:
                if self.args.task == 'phenotyping':
                    if 'c-unimodal_ehr' in self.args.fusion_type:
                        probs = self.pad_to_length(probs, max_len=48)
                    y = y.unsqueeze(1).repeat(1, pred.shape[1], 1)
                else:
                    y = y.unsqueeze(1).repeat(1, pred.shape[1])
    
            loss = self.loss(pred, y)
            epoch_loss += loss.item()
            
            # if 'c-unimodal_ehr' in self.args.fusion_type:
            #     batch_size, curr_len, num_classes = y.shape
            #     if curr_len < 2646:
            #         padded = torch.zeros(
            #             (batch_size, 2646, num_classes),
            #             dtype=y.dtype,
            #             device=y.device
            #         )
            #         padded[:, :curr_len, :] = y
            #         y = padded
            
            outPRED.append(pred.cpu())
            outGT.append(y.cpu())
            outPROB.append(probs.cpu())
            
            if not inference:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        outPROB = torch.cat(outPROB, dim=0)
        outGT = torch.cat(outGT, dim=0)
                
        return outPROB, outGT


    def eval(self):
        print("done!")

        return
    
    def compute_ece(self, probs, labels, n_bins=10):
        """
        Compute Expected Calibration Error (ECE) for each class independently, 
        but treating the prediction as a binary decision (class c vs not class c).
    
        `probs`: Tensor of shape (batch_size, num_classes) with predicted probabilities.
        `labels`: Tensor of shape (batch_size, num_classes) with binary ground truth labels.
        `n_bins`: Number of bins to discretize confidence scores.
    
        Returns:
            ece_per_class (Tensor): A tensor of shape (num_classes,) with ECE for each class.
        """
        num_classes = self.args.num_classes
        ece_per_class = torch.zeros(num_classes, device=probs.device)
    
        # Define bin boundaries
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    
        for c in range(num_classes):
            # Probability of class c
            class_probs = probs[:, c]
            # Ground truth for class c
            class_labels = labels[:, c]
    
            # Predicted label (1 if >= 0.5, else 0)
            predicted_label = (class_probs >= 0.5).long()
            # Confidence is max(p, 1-p)
            predicted_confidence = torch.where(predicted_label == 1, class_probs, 1.0 - class_probs)
    
            class_ece = torch.zeros(1, device=probs.device)
    
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Which samples fall into this confidence bin?
                in_bin = (predicted_confidence > bin_lower) & (predicted_confidence <= bin_upper)
                prop_in_bin = in_bin.float().mean()  # fraction of samples in bin
    
                if prop_in_bin.item() > 0:
                    # Average confidence within the bin
                    avg_confidence = predicted_confidence[in_bin].mean()
                    # Average accuracy (fraction correctly classified within the bin)
                    # Correct if predicted_label == actual_label
                    avg_accuracy = (predicted_label[in_bin] == class_labels[in_bin]).float().mean()
    
                    class_ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
            ece_per_class[c] = class_ece
    
        return ece_per_class

    def plot_calibration_curve(self, probs, labels, n_bins=10, save_path_prefix='calibration_curve'):
        """
        Plot one calibration curve per class, using symmetrical binning:
        Treat each class c vs not c, confidence = max(p, 1-p), then bin by that confidence
        and compute accuracy = fraction of correct predictions in that bin.
    
        Args:
            probs:  Tensor of shape (batch_size, num_classes) with predicted probabilities.
            labels: Tensor of shape (batch_size, num_classes) with ground-truth binary labels.
            n_bins: Number of bins for the calibration curve.
            save_path_prefix: Prefix for saving each class's calibration plot.
        """
    
        num_classes = self.args.num_classes
    
        # Prepare bin boundaries on the same device as probs
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        for c in range(num_classes):
            # -----------------------------
            # 1) Extract per-class data
            # -----------------------------
            class_probs = probs[:, c]    # Probability of class c
            class_labels = labels[:, c]  # Ground-truth labels for class c
            
            # Predicted label: 1 if p >= 0.5 else 0
            predicted_label = (class_probs >= 0.5).long()
            # Symmetrical confidence: p if predicting 1, else (1-p)
            predicted_conf = torch.where(predicted_label == 1, class_probs, 1.0 - class_probs)
    
            # To store bin-specific means
            bin_confs = []
            bin_accs = []
    
            # -----------------------------
            # 2) Bin samples by confidence
            # -----------------------------
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predicted_conf > bin_lower) & (predicted_conf <= bin_upper)
                prop_in_bin = in_bin.float().mean()  # fraction of samples in this bin
    
                if prop_in_bin > 0:
                    avg_conf = predicted_conf[in_bin].mean().item()
                    avg_acc = (predicted_label[in_bin] == class_labels[in_bin]).float().mean().item()
                else:
                    # No samples in this bin => store NaN
                    avg_conf = float('nan')
                    avg_acc = float('nan')
    
                bin_confs.append(avg_conf)
                bin_accs.append(avg_acc)
    
            bin_confs = np.array(bin_confs)
            bin_accs = np.array(bin_accs)
    
            # Filter out any bins where everything is NaN
            valid_mask = ~np.isnan(bin_confs)
            bin_confs = bin_confs[valid_mask]
            bin_accs = bin_accs[valid_mask]
    
            # -------------------------------------
            # 3) Plot calibration for this class
            # -------------------------------------
            plt.figure(figsize=(6, 6))
            plt.plot(bin_confs, bin_accs, marker='o', label=f"Class {c} Calibration")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
            
            plt.xlabel("Mean Predicted Confidence (max(p, 1-p))")
            plt.ylabel("Fraction Correct (TP+TN / total in bin)")
            plt.title(f"Calibration Curve - Class {c}")
            plt.legend(loc='best')
    
            # Save as "{prefix}_class_{c}.png"
            save_path = f"{save_path_prefix}_class_{c}.png"
            plt.savefig(save_path)
            plt.close()

    def train(self):
        """
        Main training loop specialized for 'c-unimodal' usage.
        """
        print(f'Running training for fusion_type {self.args.fusion_type}')
        # Compute ECE before training (inference mode)
        probs, labels = self.train_epoch(inference=True)
        pre_ece = self.compute_ece(probs, labels)
        self.plot_calibration_curve(probs, labels, 10, 'calibration_curve')
        
        print('Before Training')
        for index, class_ece in enumerate(pre_ece):
            line = f'{self.val_dl.dataset.CLASSES[index]: <90} & {class_ece:0.3f}'
            print(line)
        batch_size = probs.shape[0]
        token_dim = probs.shape[1]
        num_classes = probs.shape[2] if probs.dim() == 3 else 1

        # Initialize an empty table for storing ECE values
        ece_table = np.zeros((token_dim, num_classes))

        for t in range(token_dim):
            if num_classes > 1:
                token_probs = probs[:, t, :]        # shape: [batch_size, num_classes]
                token_labels = labels[:, t, :]
            else:
                token_probs = probs[:, t].unsqueeze(-1)
                token_labels = labels[:, t].unsqueeze(-1)

            token_ece = self.compute_ece(token_probs, token_labels, n_bins=10)
            ece_table[t, :] = token_ece.detach().cpu().numpy()

        token_ids = [f"Token {i}" for i in range(token_dim)]
        class_labels = [self.val_dl.dataset.CLASSES[index] for index in range(num_classes)]

        ece_df = pd.DataFrame(ece_table, index=token_ids, columns=class_labels)
        ece_filename = f'{self.args.modalities}_{self.args.task}_{self.args.lr}_ece_table_epoch_{self.epoch}.csv'
        ece_df.to_csv(ece_filename)
        print(f"ECE table saved to {ece_filename}")

        prob_data = []
        for sample_idx in range(batch_size):
            for token_idx in range(token_dim):
                row_data = {
                    'Sample_ID': sample_idx,
                    'Token_Position': token_idx
                }
                if num_classes > 1:
                    for class_idx, class_name in enumerate(self.val_dl.dataset.CLASSES):
                        prob = probs[sample_idx, token_idx, class_idx].item()
                        confidence = max(prob, 1.0 - prob)
                        row_data[f'Prob_{class_name}'] = confidence
                else:
                    prob = probs[sample_idx, token_idx].item()
                    confidence = max(prob, 1.0 - prob)
                    class_name = self.val_dl.dataset.CLASSES[0]
                    row_data[f'Prob_{class_name}'] = confidence
                prob_data.append(row_data)

        prob_df = pd.DataFrame(prob_data)
        prob_filename = f'per_token_probabilities_{self.args.modalities}_{self.args.task}_{self.args.lr}_epoch_{self.epoch}.csv'
        prob_df.to_csv(prob_filename, index=False)
        print(f"Saved per-token probabilities to {prob_filename}")
        self.best_auroc = pre_ece.mean()
        wandb.log({"epoch": self.epoch, "post_ece": pre_ece.mean()})
        for self.epoch in range(self.start_epoch, self.args.epochs):
            # Run actual training step
            probs, labels = self.train_epoch()
    
            # Compute ECE after training
            post_ece = self.compute_ece(probs, labels)
            print('After Training')
    
            # Log ECE values to Weights & Biases
            wandb.log({"epoch": self.epoch, "post_ece": post_ece.mean()})
            if post_ece.mean() < self.best_auroc:
                self.best_auroc = post_ece.mean()
                self.save_checkpoint()
                print('Checkpoint :)')
                batch_size = probs.shape[0]
                token_dim = probs.shape[1]
                num_classes = probs.shape[2] if probs.dim() == 3 else 1
        
                # Initialize an empty table for storing ECE values
                ece_table = np.zeros((token_dim, num_classes))
        
                for t in range(token_dim):
                    if num_classes > 1:
                        token_probs = probs[:, t, :]        # shape: [batch_size, num_classes]
                        token_labels = labels[:, t, :]
                    else:
                        token_probs = probs[:, t].unsqueeze(-1)
                        token_labels = labels[:, t].unsqueeze(-1)
        
                    token_ece = self.compute_ece(token_probs, token_labels, n_bins=10)
                    ece_table[t, :] = token_ece.detach().cpu().numpy()
        
                token_ids = [f"Token {i}" for i in range(token_dim)]
                class_labels = [self.val_dl.dataset.CLASSES[index] for index in range(num_classes)]
        
                ece_df = pd.DataFrame(ece_table, index=token_ids, columns=class_labels)
                ece_filename = f'final_{self.args.modalities}_{self.args.task}_{self.args.lr}_ece_table_epoch_{self.epoch}.csv'
                ece_df.to_csv(ece_filename)
                print(f"ECE table saved to {ece_filename}")
        return
