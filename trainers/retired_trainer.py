
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
        self.optimizer = optim.LBFGS(self.model.parameters(), lr=args.lr, max_iter=50)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

    def train_epoch(self, inference = False):
        """
        Training loop for LBFGS. 
        Collects the entire dataset from self.train_dl into one batch, then performs a single LBFGS step.
        """
        print(f"Starting train epoch {self.epoch}")
        print(f"Starting {'inference' if inference else 'training'} epoch {self.epoch}")
        self.model.train(not inference)  # Set model to eval mode for inference
    
        # === 1) Gather entire data from the DataLoader ===
        all_x, all_imgs, all_dn, all_rr = [], [], [], []
        all_y_list = []
        fixed_y = []
        all_seq_lengths, all_pairs = [], []
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, *_ ) in enumerate(self.val_dl):
            # Convert everything to Tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float)
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img, dtype=torch.float)
            if not isinstance(seq_lengths, torch.Tensor):
                seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
    
            # Decide which label (EHR vs. CXR) is appropriate
            y = self.get_gt(y_ehr, y_cxr)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float)
    
            # Accumulate in lists for later concatenation
            all_x.append(x)
            all_imgs.append(img)
            all_dn.extend(dn)
            all_rr.extend(rr)
            all_seq_lengths.append(seq_lengths)
            all_pairs.append(pairs)
            all_y_list.append(y)
    
        # === 2) Concatenate everything into a single large batch ===
        # Make sure each list is valid for concatenation (e.g., same shapes along dim=0)
        X = torch.cat(all_x, dim=0).to(self.device)
        IMG = torch.cat(all_imgs, dim=0).to(self.device)
        DN = all_dn
        RR = all_rr
        seq_lens = torch.cat(all_seq_lengths, dim=0)
        y_ = torch.cat(all_y_list, dim=0).to(self.device)
    
        # all_pairs might just be a list of items you pass directly to the model
        # If your model expects a tensor for 'pairs', you'd convert it similarly.
        Pairs = all_pairs
    
        # === 3) Define the closure (LBFGS will call it multiple times) ===
        def closure():
            self.optimizer.zero_grad()
            # Forward pass with the entire batch
            output_dict = self.model(X, seq_lens, IMG, Pairs, RR, DN)
            pred = output_dict[self.args.fusion_type].squeeze()
    
            if 'c-unimodal' in self.args.fusion_type:
                if self.args.task == 'phenotyping':
                    # E.g.: [batch_size, pred_dim, ???]
                    # Adjust if your data shape differs
                    Y = y_.unsqueeze(1).repeat(1, pred.shape[1], 1)
                else:
                    # e.g.: [batch_size, pred_dim]
                    Y = y_.unsqueeze(1).repeat(1, pred.shape[1])
            else:
                Y=y_
    
            loss_val = self.loss(pred, Y)
            loss_val.backward()
            return loss_val
    
        # === 4) LBFGS step ===
        if not inference:
            self.optimizer.step(closure)
    
        # === 5) One more forward pass for logging or returning predictions ===
        with torch.no_grad():
            out_dict = self.model(X, seq_lens, IMG, Pairs, RR, DN)
            final_preds = out_dict[self.args.fusion_type].squeeze()
            probs = torch.sigmoid(final_preds)
            if 'c-unimodal' in self.args.fusion_type:
                if self.args.task == 'phenotyping':
                    # [batch size, token_dim, num_classes]
                    # Adjust if your data shape differs
                    Y = y_.unsqueeze(1).repeat(1, final_preds.shape[1], 1)
                    #Y = Y.max(dim=-1).values
                else:
                    Y = y_.unsqueeze(1).repeat(1, final_preds.shape[1])
            else:
                Y= y_
    
        print("Train epoch done.")
        return probs, Y

    def train_epoch_batchwise_rr(self, inference=False):
        """
        Specialized training function for c-unimodal_rr that processes data in batches
        to avoid memory overflow from storing the entire radiology reports.
        """
        print(f"Starting {'inference' if inference else 'training'} epoch {self.epoch} (Batch-wise RR)")
        self.model.train(not inference)
    
        all_probs, all_labels = [], []
    
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, *_) in enumerate(self.val_dl):
            # Convert to tensors if not already
            x = torch.tensor(x, dtype=torch.float).to(self.device)
            img = torch.tensor(img, dtype=torch.float).to(self.device)
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
    
            y = self.get_gt(y_ehr, y_cxr)
            y = torch.tensor(y, dtype=torch.float).to(self.device)
    
            # Single batch closure function for LBFGS
            def closure():
                self.optimizer.zero_grad()
                output_dict = self.model(x, seq_lengths, img, pairs, rr, dn)
                pred = output_dict[self.args.fusion_type].squeeze()
    
                if self.args.task == 'phenotyping':
                    Y = y.unsqueeze(1).repeat(1, pred.shape[1], 1)
                else:
                    Y = y.unsqueeze(1).repeat(1, pred.shape[1])
    
                loss_val = self.loss(pred, Y)
                loss_val.backward()
                return loss_val
    
            # Perform LBFGS step per batch if not inference
            if not inference:
                self.optimizer.step(closure)
    
            with torch.no_grad():
                out_dict = self.model(x, seq_lengths, img, pairs, rr, dn)
                final_preds = out_dict[self.args.fusion_type].squeeze()
                probs = torch.sigmoid(final_preds)
    
                if self.args.task == 'phenotyping':
                    Y = y.unsqueeze(1).repeat(1, final_preds.shape[1], 1)
                else:
                    Y = y.unsqueeze(1).repeat(1, final_preds.shape[1])
    
                all_probs.append(probs.cpu())
                all_labels.append(Y.cpu())
    
        # Concatenate predictions and labels from all batches
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
    
        print("Batch-wise RR epoch done.")
        return all_probs, all_labels


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
    
        num_classes = probs.shape[1]
    
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
        if 'c-unimodal' in self.args.fusion_type:
            for self.epoch in range(self.start_epoch, self.args.epochs):
                # Compute ECE before training (inference mode)
                if 'c-unimodal_rr' in self.args.fusion_type:
                    probs, labels = self.train_epoch_batchwise_rr(inference=True)
                else:
                    probs, labels = self.train_epoch(inference=True)
        
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
                    ece_table[t, :] = token_ece.cpu().numpy()
        
                token_ids = [f"Token {i}" for i in range(token_dim)]
                class_labels = [self.val_dl.dataset.CLASSES[index] for index in range(num_classes)]
        
                ece_df = pd.DataFrame(ece_table, index=token_ids, columns=class_labels)
                ece_filename = f'{self.args.modalities}_{self.args.task}_ece_table_epoch_{self.epoch}.csv'
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
                prob_filename = f'per_token_probabilities_{self.args.modalities}_{self.args.task}_epoch_{self.epoch}.csv'
                prob_df.to_csv(prob_filename, index=False)
                print(f"Saved per-token probabilities to {prob_filename}")

        else:
            print(f'Running training for fusion_type {self.args.fusion_type}')
            for self.epoch in range(self.start_epoch, self.args.epochs):
                # Compute ECE before training (inference mode)
                probs, labels = self.train_epoch(inference=True)
                pre_ece = self.compute_ece(probs, labels)
                self.plot_calibration_curve(probs, labels, 10, 'calibration_curve')
                
                print('Before Training')
                for index, class_ece in enumerate(pre_ece):
                    line = f'{self.val_dl.dataset.CLASSES[index]: <90} & {class_ece:0.3f}'
                    print(line)
                    
        
                # Run actual training step
                probs, labels = self.train_epoch()
        
                # Compute ECE after training
                post_ece = self.compute_ece(probs, labels)
                print('After Training')
                for index, class_ece in enumerate(pre_ece):
                    line = f'{self.val_dl.dataset.CLASSES[index]: <90} & {class_ece:0.3f}'
                    print(line)
        
                # Log ECE values to Weights & Biases
                wandb.log({"epoch": self.epoch, "pre_ece": pre_ece, "post_ece": post_ece})
                if post_ece.mean() < pre_ece.mean(): 
                    self.save_checkpoint()
            
        # In c-unimodal, we don't track a best AUROC, so no final print of it.
        return


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

    # def compute_ece(self, probs, labels, n_bins=15):
    #     """
    #     Compute Expected Calibration Error (ECE) for multi-label outputs.
    
    #     `probs`: Tensor of shape (batch_size, num_classes) with predicted probabilities.
    #     `labels`: Tensor of shape (batch_size, num_classes) with binary ground truth labels.
    #     `n_bins`: Number of bins to discretize confidence scores.
    
    #     Returns:
    #         ece (float): Expected Calibration Error
    #     """
        
    #     if self.args.num_classes > 1:
    #         probs = probs.max(dim=-1).values
        
    #     # Flatten all predictions into a single 1D array (treat each prediction independently)
    #     probs = probs.view(-1)   # Shape: (batch_size * num_classes,)
    #     labels = labels.view(-1) # Shape: (batch_size * num_classes,)
    
    #     # Define bin boundaries
    #     bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    #     bin_lowers = bin_boundaries[:-1]
    #     bin_uppers = bin_boundaries[1:]
    
    #     ece = torch.zeros(1, device=probs.device)
    
    #     for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    #         in_bin = (probs > bin_lower) & (probs <= bin_upper)  # Find samples in this bin
    #         prop_in_bin = in_bin.float().mean()  # Fraction of samples in bin
    
    #         if prop_in_bin.item() > 0:  # Avoid division by zero
    #             avg_confidence = probs[in_bin].mean()
    #             avg_accuracy = labels[in_bin].float().mean()
    #             ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    #     return ece.item()
    
    # def compute_ece(self, probs, labels, n_bins=10):
    #     """
    #     Compute Expected Calibration Error (ECE) for each class independently.
    
    #     `probs`: Tensor of shape (batch_size, num_classes) with predicted probabilities.
    #     `labels`: Tensor of shape (batch_size, num_classes) with binary ground truth labels.
    #     `n_bins`: Number of bins to discretize confidence scores.
    
    #     Returns:
    #         ece_per_class (Tensor): A tensor of shape (num_classes,) with ECE for each class.
    #     """
    #     num_classes = self.args.num_classes
    #     ece_per_class = torch.zeros(num_classes, device=probs.device)
        
    #     # Define bin boundaries
    #     bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    #     bin_lowers = bin_boundaries[:-1]
    #     bin_uppers = bin_boundaries[1:]
        
    #     for c in range(num_classes):
    #         class_probs = probs[:, c]  # Get probabilities for class c
    #         class_labels = labels[:, c]  # Get labels for class c
            
    #         class_ece = torch.zeros(1, device=probs.device)
            
    #         for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    #             in_bin = (class_probs > bin_lower) & (class_probs <= bin_upper)  # Find samples in this bin
    #             prop_in_bin = in_bin.float().mean()  # Fraction of samples in bin
                
    #             if prop_in_bin.item() > 0:  # Avoid division by zero
    #                 avg_confidence = class_probs[in_bin].mean()
    #                 avg_accuracy = class_labels[in_bin].float().mean()
    #                 class_ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin
            
    #         ece_per_class[c] = class_ece
        
    #     return ece_per_class

class MSMA_Trainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl):
        
        super(MSMA_Trainer, self).__init__(args)
        run = wandb.init(project=f'MSMA_{self.args.fusion_type}_{self.args.task}', config=args)
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
        
        self.load_model(args)   
        
        self.loss = Loss(args)
            
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
    
    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, age, gender, ethnicity, hadm_id) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs, rr, dn)
            
            pred = output[self.args.fusion_type].squeeze()
            loss = self.loss(pred, y)
            epoch_loss += loss.item()
            if self.args.align > 0.0:
                loss = loss + self.args.align * output['align_loss']
                epoch_loss_align = epoch_loss_align + self.args.align * output['align_loss'].item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
        # self.epochs_stats['loss train'].append(epoch_loss/i)
        # self.epochs_stats['loss align train'].append(epoch_loss_align/i)
        wandb.log({
                'train_Loss': epoch_loss/i, 
                'train_AUC': ret['auroc_mean']
            })
        return ret
        
    def validate(self, dl):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0

        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        
        predictions_by_hadm = {}
        ground_truth_by_hadm = {}

        with torch.no_grad():
            for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, age, gender, ethnicity, hadm_id) in enumerate (dl):
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                output = self.model(x, seq_lengths, img, pairs, rr, dn)
                
                pred = output[self.args.fusion_type]
                
                if self.args.fusion_type != 'uni_cxr':
                    if len(pred.shape) > 1:
                         pred = pred.squeeze()
                           
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                if self.args.align > 0.0:
                    epoch_loss_align +=  output['align_loss'].item()
                    
                # # Collect predictions and true labels by hadm_id
                # if self.args.task == "phenotyping":
                #     hadm_id = hadm_id.cpu().numpy()
                #     pred = pred.cpu().numpy()
                #     y = y.cpu().numpy()
                #     for j, h_id in enumerate(hadm_id):
                #         # Ensure ground truth is consistent for each hadm_id
                #         if h_id not in ground_truth_by_hadm:
                #             # First occurrence of this hadm_id
                #             ground_truth_by_hadm[h_id] = y[j]
                #         else:
                #             # Check for consistency in ground truth labels
                #             assert np.array_equal(ground_truth_by_hadm[h_id], y[j]), \
                #                 f"Mismatch in true labels for hadm_id {h_id}. Expected {ground_truth_by_hadm[h_id]}, got {y[j]}"
                        
                #         # Add predictions for this hadm_id
                #         if h_id not in predictions_by_hadm:
                #             predictions_by_hadm[h_id] = []
                #         predictions_by_hadm[h_id].append(pred[j])
                # else:
                # Normal behavior for other tasks
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)
                
            # if self.args.task == "phenotyping":
            #     averaged_predictions = []
            #     true_labels = []
            #     for h_id in predictions_by_hadm:
            #         averaged_predictions.append(np.mean(predictions_by_hadm[h_id], axis=0))
            #         true_labels.append(ground_truth_by_hadm[h_id])
                
            #     outPRED = torch.tensor(averaged_predictions, device=self.device)
            #     outGT = torch.tensor(true_labels, device=self.device)
        
        self.scheduler.step(epoch_loss/len(self.val_dl))

        print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f} \t{epoch_loss_align/i:0.5f}")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
        np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy()) 
        np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy()) 


        # self.epochs_stats['auroc val'].append(ret['auroc_mean'])

        # self.epochs_stats['loss val'].append(epoch_loss/i)
        # self.epochs_stats['loss align val'].append(epoch_loss_align/i)
        avg_loss = epoch_loss/i
        
        return ret, avg_loss
        
    def eval(self):
        if self.args.mode == 'train':
            self.load_state(state_path=f'{self.args.save_dir}/{self.args.task}/{self.args.fusion_type}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.fusion_type}_{self.args.modalities}_{self.args.data_pairs}.pth.tar')
        
        self.epoch = 0
        self.model.eval() 

        ret, avg_loss = self.validate(self.test_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
        wandb.log({
                'test_auprc': ret['auprc_mean'], 
                'test_AUC': ret['auroc_mean']
            })
        return
    
    def train(self):
        print(f'running for fusion_type {self.args.fusion_type}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.eval()
            ret, avg_loss = self.validate(self.val_dl)
            wandb.log({
                'val_Loss': avg_loss,
                'val_AUC': ret['auroc_mean']
            })
    
            if self.best_auroc < ret['auroc_mean']:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_checkpoint()
                print("checkpoint")
                self.print_and_write(ret, isbest=True)
                self.patience = 0
            else:
                self.print_and_write(ret, isbest=False)
                self.patience+=1

            self.model.train()
            self.train_epoch()
            # self.plot_stats(key='loss', filename='loss.pdf')
            # self.plot_stats(key='auroc', filename='auroc.pdf')
            if self.patience >= self.args.patience:
                break
        self.print_and_write(self.best_stats , isbest=True)