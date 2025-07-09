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
        
        if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
            self.model.equalize()
            self.weights = nn.Parameter(torch.ones(3))
            self.optimizer = optim.Adam(list(self.model.parameters()) + [self.weights], args.lr, betas=(0.9, self.args.beta_1))
                
        else:
            self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        
        self.loss = Loss(args)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        if 'c-unimodal' not in self.args.fusion_type:
            self.best_auroc = 0
        else:
            self.best_auroc = float('inf')
        self.best_stats = None
        
                # Count parameters and exit if requested
        if self.args.inspect_model:
            model_params = list(self.model.parameters())
            extra_params = [self.weights] if hasattr(self, 'weights') else []
            all_params = model_params + extra_params

            total_params = sum(p.numel() for p in all_params)
            trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
            non_trainable_params = total_params - trainable_params

            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")
            print(f"Non-trainable parameters: {non_trainable_params}")
            sys.exit(0)

    
    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
            epoch_loss_high = 0
            epoch_loss_low = 0
            epoch_loss_late = 0
            outPRED_high = torch.FloatTensor().to(self.device)
            outPRED_low = torch.FloatTensor().to(self.device)
            outPRED_late = torch.FloatTensor().to(self.device)
            outPRED_combined = torch.FloatTensor().to(self.device)
            outPRED_less_combined = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, age, gender, ethnicity, hadm_id) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs, rr, dn)
            
            if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
                 # Extract predictions
                pred_high = output['high_conf'].squeeze()
                pred_low = output['low'].squeeze()
                pred_late = output['late'].squeeze()
    
                # Compute individual losses
                loss_high = self.loss(pred_high, y)
                loss_low = self.loss(pred_low, y)
                loss_late = self.loss(pred_late, y)
    
                if self.args.ablation == "without_joint_module":
                    loss = loss_late
                else:
                    weight_norm = torch.softmax(self.weights, dim=0)
                    loss = (weight_norm[0] * loss_high +
                            weight_norm[1] * loss_low +
                            weight_norm[2] * loss_late)
                epoch_loss += loss.item()
                epoch_loss_high += loss_high.item()
                epoch_loss_low += loss_low.item()
                epoch_loss_late += loss_late.item()
    
                # Collect predictions and ground truth
                outPRED_high = torch.cat((outPRED_high, pred_high), 0)
                outPRED_low = torch.cat((outPRED_low, pred_low), 0)
                outPRED_late = torch.cat((outPRED_late, pred_late), 0)
                pred_combined = (pred_high + pred_low + pred_late) / 3
                pred_less_combined = (pred_high + pred_late) / 2
                outPRED_combined = torch.cat((outPRED_combined, pred_combined), 0)
                outPRED_less_combined = torch.cat((outPRED_less_combined, pred_less_combined), 0)
                outGT = torch.cat((outGT, y), 0)
            else:
                pred = output[self.args.fusion_type].squeeze()
                if 'c-unimodal' in self.args.fusion_type:
                    if self.args.task == 'phenotyping':
                        y = y.unsqueeze(1).repeat(1, pred.shape[1], 1)
                    else:
                        y = y.unsqueeze(1).repeat(1, pred.shape[1])
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                if self.args.align > 0.0:
                    loss = loss + self.args.align * output['align_loss']
                    epoch_loss_align = epoch_loss_align + self.args.align * output['align_loss'].item()
                if 'c-unimodal' in self.args.fusion_type:
                    pred = torch.sigmoid(pred) 
                    if self.args.num_classes > 1:
                        pred = pred.max(dim=-1).values
                else: 
                    outPRED = torch.cat((outPRED, pred), 0)
                if 'c-unimodal' not in self.args.fusion_type: 
                    outGT = torch.cat((outGT, y), 0)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}")
        if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
            ret_high = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_high.data.cpu().numpy(), 'train')
            ret_low = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_low.data.cpu().numpy(), 'train')
            ret_late = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_late.data.cpu().numpy(), 'train')
            ret_combined = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_combined.data.cpu().numpy(), 'train')
            ret_less_combined = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_less_combined.data.cpu().numpy(), 'train')
    
            # Log results to WandB
            wandb.log({
                'train_Loss': epoch_loss / i,
                'train_Loss_High': epoch_loss_high / i,
                'train_Loss_Low': epoch_loss_low / i,
                'train_Loss_Late': epoch_loss_late / i,
                'train_AUC_High': ret_high['auroc_mean'],
                'train_AUC_Low': ret_low['auroc_mean'],
                'train_AUC_Late': ret_late['auroc_mean'],
                'train_AUC_Combined': ret_combined['auroc_mean'],
                'train_AUC_less_combined' : ret_less_combined['auroc_mean'],
            })
    
            return ret_combined
        else:
            if 'c-unimodal' not in self.args.fusion_type:
                ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
                wandb.log({
                    'train_Loss': epoch_loss/i, 
                    'train_AUC': ret['auroc_mean']
                })
            else:
                ret = None
                wandb.log({
                    'train_Loss': epoch_loss/i
                })
                # np.save(f'{self.args.save_dir}/{self.args.modalities}_train_confidences.npy', outPRED.data.cpu().numpy()) 
            # self.epochs_stats['loss train'].append(epoch_loss/i)
            # self.epochs_stats['loss align train'].append(epoch_loss_align/i)
        return ret
        
    def validate(self, dl, test):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0

        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
            epoch_loss_high = 0
            epoch_loss_low = 0
            epoch_loss_late = 0
            outPRED_high = torch.FloatTensor().to(self.device)
            outPRED_low = torch.FloatTensor().to(self.device)
            outPRED_late = torch.FloatTensor().to(self.device)
            outPRED_combined = torch.FloatTensor().to(self.device)
            outPRED_less_combined = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, age, gender, ethnicity, hadm_id) in enumerate (dl):
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                output = self.model(x, seq_lengths, img, pairs, rr, dn)
                
                if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
                     # Extract predictions
                    pred_high = output['high_conf'].squeeze()
                    pred_low = output['low'].squeeze()
                    pred_late = output['late'].squeeze()
        
                    # Compute individual losses
                    loss_high = self.loss(pred_high, y)
                    loss_low = self.loss(pred_low, y)
                    loss_late = self.loss(pred_late, y)
        
                    # Combine losses with weights
                    if self.args.ablation == "without_joint_module":
                        loss = loss_late
                    else:
                        weight_norm = torch.softmax(self.weights, dim=0)
                        
                        if self.args.fusion_type == 'c-e-msma':
                            weight_norm[0] = 0.9999833106994628
                            weight_norm[1] = 0.00001670859091973398
                            weight_norm[2] = 0.00000000467673544335
                        
                        loss = (weight_norm[0] * loss_high +
                                weight_norm[1] * loss_low +
                                weight_norm[2] * loss_late)
                    epoch_loss += loss.item()
                    epoch_loss_high += loss_high.item()
                    epoch_loss_low += loss_low.item()
                    epoch_loss_late += loss_late.item()
        
                    # Collect predictions and ground truth
                    outPRED_high = torch.cat((outPRED_high, pred_high), 0)
                    outPRED_low = torch.cat((outPRED_low, pred_low), 0)
                    outPRED_late = torch.cat((outPRED_late, pred_late), 0)
                    pred_combined = (pred_high + pred_low + pred_late) / 3
                    pred_less_combined = (pred_high + pred_late) / 2
                    outPRED_combined = torch.cat((outPRED_combined, pred_combined), 0)
                    outPRED_less_combined = torch.cat((outPRED_less_combined, pred_less_combined), 0)
                    outGT = torch.cat((outGT, y), 0)
                else:
                    pred = output[self.args.fusion_type]
                    if self.args.fusion_type != 'uni_cxr':
                        if len(pred.shape) > 1:
                             pred = pred.squeeze()
                    if 'c-unimodal' in self.args.fusion_type:
                        if self.args.task == 'phenotyping':
                            y = y.unsqueeze(1).repeat(1, pred.shape[1], 1)
                        else:
                            y = y.unsqueeze(1).repeat(1, pred.shape[1])
                    
                               
                    loss = self.loss(pred, y)
                    epoch_loss += loss.item()
                    if self.args.align > 0.0:
                        epoch_loss_align +=  output['align_loss'].item()
                        
                    if 'c-unimodal' in self.args.fusion_type:
                        pred = torch.sigmoid(pred) 
                        if self.args.num_classes > 1:
                            pred = pred.max(dim=-1).values
                        
                    else:
                        outPRED = torch.cat((outPRED, pred), 0)
                        outGT = torch.cat((outGT, y), 0)
        
        self.scheduler.step(epoch_loss/len(self.val_dl))
        
        print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f} \t{epoch_loss_align/i:0.5f}")
        
        if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
            ret_high = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_high.data.cpu().numpy(), 'train')
            ret_low = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_low.data.cpu().numpy(), 'train')
            ret_late = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_late.data.cpu().numpy(), 'train')
            ret_combined = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_combined.data.cpu().numpy(), 'train')
            ret_less_combined = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_less_combined.data.cpu().numpy(), 'train')
            avg_loss = epoch_loss/i
            
            if self.args.stat_test:
                # Identify the best AUROC
                all_rets = {
                    'high': (ret_high, outPRED_high),
                    'low': (ret_low, outPRED_low),
                    'late': (ret_late, outPRED_late),
                    'combined': (ret_combined, outPRED_combined),
                    'less_combined': (ret_less_combined, outPRED_less_combined)
                }
                best_key, (best_ret, best_pred) = max(all_rets.items(), key=lambda item: item[1][0]['auroc_mean'])
                
                # Save best predictions and ground truth
                base_name = f"{self.args.fusion_type}_{self.args.task}_{self.args.modalities}"
                np.save(f"{base_name}_best_pred.npy", best_pred.data.cpu().numpy())
                np.save(f"{base_name}_best_gt.npy", outGT.data.cpu().numpy())
                
    
            return ret_high, ret_low, ret_late, ret_combined, ret_less_combined, avg_loss
        else:
            if 'c-unimodal' not in self.args.fusion_type:
                ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            else:
                ret = None
            #     if test:
            #         np.save(f'{self.args.save_dir}/{self.args.modalities}_test_confidences.npy', outPRED.data.cpu().numpy())
            #     else:
            #         np.save(f'{self.args.save_dir}/{self.args.modalities}_val_confidences.npy', outPRED.data.cpu().numpy())
            np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy()) 
            
            if ret is not None:
                if self.args.stat_test:
                    base_name = f"{self.args.fusion_type}_{self.args.task}"
                    np.save(f"{base_name}_best_pred.npy", outPRED.data.cpu().numpy())
                    np.save(f"{base_name}_best_gt.npy", outGT.data.cpu().numpy())
    
            avg_loss = epoch_loss/i
        
            return ret, avg_loss
        
    def eval(self):
        #if self.args.mode == 'train':
        if self.args.fusion_type != 'late':
            self.load_state(state_path=f'{self.args.save_dir}/{self.args.task}/{self.args.fusion_type}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.fusion_type}_{self.args.modalities}_{self.args.data_pairs}.pth.tar')
        
        self.epoch = 0
        self.model.eval()
        
        if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
            ret_high, ret_low, ret_late, ret_combined, ret_less_combined, avg_loss = self.validate(self.test_dl, True)
            
            
            ret = max([ret_high, ret_low, ret_late, ret_combined, ret_less_combined], key=lambda x: x['auroc_mean'])
        
            self.print_and_write(ret, isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
            
            weight_norm = torch.softmax(self.weights, dim=0)
            
            wandb.log({
                'test_AUC_high': ret_high['auroc_mean'],
                'test_AUC_Late': ret_late['auroc_mean'],
                'test_AUC_Combined': ret_combined['auroc_mean'],
                'test_AUC_Less_Combined': ret_less_combined['auroc_mean'],
                'test_auprc_high': ret_high['auprc_mean'],
                'test_auprc_Late': ret_late['auprc_mean'],
                'test_auprc_Combined': ret_combined['auprc_mean'],
                'test_auprc_Less_Combined': ret_less_combined['auprc_mean'],
                'Weight_Low': weight_norm[1],
                'Weight_High': weight_norm[0],
                'Weight_Late': weight_norm[2]
            })
        else:
            ret, avg_loss = self.validate(self.test_dl, True)
            if 'c-unimodal' not in self.args.fusion_type:
                self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
                wandb.log({
                        'test_auprc': ret['auprc_mean'], 
                        'test_AUC': ret['auroc_mean']
                    })
            else:
                wandb.log({
                        'test_loss' : avg_loss
                    })
        return
    
    def train(self):
        test = False
        print(f'running for fusion_type {self.args.fusion_type}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.eval()
            if self.args.fusion_type == 'c-msma' or self.args.fusion_type == 'c-e-msma':
                ret_high, ret_low, ret_late, ret_combined, ret_less_combined, avg_loss = self.validate(self.val_dl,False)
                weight_norm = torch.softmax(self.weights, dim=0)
                wandb.log({
                    'val_Loss': avg_loss,
                    'val_AUC_High': ret_high['auroc_mean'],
                    'val_AUC_Low': ret_low['auroc_mean'],
                    'val_AUC_Late': ret_late['auroc_mean'],
                    'val_AUC_Combined': ret_combined['auroc_mean'],
                    'val_AUC_Less_Combined': ret_less_combined['auroc_mean'],
                    'Weight_Low': weight_norm[1],
                    'Weight_High': weight_norm[0],
                    'Weight_Late': weight_norm[2]
                })
                ret = max([ret_high, ret_low, ret_late, ret_combined, ret_less_combined], key=lambda x: x['auroc_mean'])
            else:
                ret, avg_loss = self.validate(self.val_dl, False)
            if 'c-unimodal' not in self.args.fusion_type:
                if self.args.fusion_type != 'c-msma' and self.args.fusion_type != 'c-e-msma':
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
            else:
                wandb.log({
                    'val_Loss': avg_loss
                })

                if self.best_auroc > avg_loss:
                    self.best_auroc = avg_loss
                    self.save_checkpoint()
                    print("checkpoint")
                    #self.print_and_write(ret, isbest=True)
                    self.patience = 0
                else:
                    print('Not the best')
                    self.patience+=1

            self.model.train()
            self.train_epoch()

            if self.patience >= self.args.patience:
                break
        if 'c-unimodal' not in self.args.fusion_type:
            self.print_and_write(self.best_stats , isbest=True)