from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
# import transformers.CompactTransformers.src as cct_models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.fusion_mmtm import FusionMMTM
from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from .trainer import Trainer

# from cxr_models import CNN


import numpy as np
from sklearn import metrics
import wandb
# from sklearn import metrics

# from common_utils import print_metrics_multilabel

class CustomBins:
    inf = 1e18
    bins = [(-1*inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]
             
    
def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]*24.0
        b = CustomBins.bins[i][1]*24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None
    
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


class MMTMTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl=None,
        train_iter=None,
        eval_iter=None,
        ):

        super(MMTMTrainer, self).__init__(args)
        
        run = wandb.init(project=f'Medfuse_{self.args.fusion_type}', config=args)
        
        self.eval_iter = eval_iter
        self.train_iter = train_iter
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.best_kappa = 0

        self.ehr_model = LSTM(args, input_dim=76, num_classes=args.num_classes, hidden_dim=args.dim, dropout=args.dropout, layers=args.layers).to(self.device)
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)



        self.model = FusionMMTM(args, self.ehr_model, self.cxr_model ).to(self.device)

        if self.args.task=="length-of-stay":
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.BCELoss()

        self.optimizer_visual = optim.Adam([{'params': self.model.cxr_model.parameters()},{'params': self.model.mmtm0.parameters()}], args.lr, betas=(0.9, self.args.beta_1))
        self.optimizer_ehr = optim.Adam([{'params': self.model.cxr_model.parameters()},{'params': self.model.mmtm0.parameters()}], args.lr, betas=(0.9, self.args.beta_1))
        self.optimizer_joint = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        self.optimizer_early = optim.Adam(self.model.joint_cls.parameters(), args.lr, betas=(0.9, self.args.beta_1))

        self.load_state()
        print(self.optimizer_visual)
        print(self.loss)
        self.scheduler_visual = ReduceLROnPlateau(self.optimizer_visual, factor=0.5, patience=10, mode='min')
        self.scheduler_ehr = ReduceLROnPlateau(self.optimizer_ehr, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {'loss train cxr': [], 'loss train ehr': [], 'loss val cxr': [], 'loss val ehr': [], 'auroc val cxr': [], 'auroc val ehr': [], 'auroc val avg': [],
        'auroc val joint': [], 'loss train joint': [], 'loss val joint': [], 'loss train align': []}
        if self.args.pretrained:
            self.load_ehr_pheno()
            self.load_cxr_pheno()
            self.load_state()
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99) 

    def step(self, optim, pred, y, key='ehr'):
        loss = self.loss(pred[key].squeeze(), y)
        pred['align_loss'] = self.args.align * pred['align_loss']
        if self.args.align > 0:
            loss = loss + pred['align_loss']
        
            
        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss

    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{prefix}_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar'
        torch.save(
            {
            'epoch': self.epoch, 
            'state_dict': self.model.state_dict(), 
            'best_auroc': self.best_auroc, 
            'optimizer_visual' : self.optimizer_visual.state_dict(),
            'optimizer_ehr' : self.optimizer_ehr.state_dict(),
            'epochs_stats': self.epochs_stats
            }, path)
        print(f"saving {prefix} checkpoint at epoch {self.epoch}")

    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        cxr_loss = 0
        ehr_loss = 0
        joint_loss = 0
        align_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            
            img = img.to(self.device)

            if self.args.task == "length-of-stay":
                y_true_bins = torch.tensor([get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()], dtype=torch.long).to(self.device)
                output = self.model(x, seq_lengths, img)
                loss_joint = self.step(self.optimizer_ehr, output, y_true_bins, key='ehr_only_scores')
    
                output = self.model(x, seq_lengths, img)
                loss_joint = self.step(self.optimizer_visual, output, y_true_bins, key='cxr_only_scores')
    
                output = self.model(x, seq_lengths, img)
                loss_joint = self.step(self.optimizer_joint, output, y_true_bins, key='joint_scores')
            else:
                output = self.model(x, seq_lengths, img)
                loss_joint = self.step(self.optimizer_ehr, output, y, key='ehr_only')
    
                output = self.model(x, seq_lengths, img)
                loss_joint = self.step(self.optimizer_visual, output, y, key='cxr_only')
    
                output = self.model(x, seq_lengths, img)
                loss_joint = self.step(self.optimizer_joint, output, y, key='joint')

            epoch_loss = epoch_loss + loss_joint.item() 
            joint_loss +=loss_joint.item()
            align_loss += output['align_loss'].item()

            if self.train_iter is not None and (i+1) % self.train_iter == 0:
                # print(f'evaluation after {i} iteration')
                # self.eval_script()
                break

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer_ehr.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} align loss {output['align_loss'].item():0.5f}")

        
        wandb.log({
                'train_Loss': epoch_loss/i
            })
    
    def validate(self, dl, full_run=False):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        ehr_loss = 0
        cxr_loss = 0
        joint_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        outPRED_cxr = torch.FloatTensor().to(self.device)
        outPRED_ehr = torch.FloatTensor().to(self.device)
        outPRED_joint = torch.FloatTensor().to(self.device)
        with torch.no_grad():
            for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (dl):
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)

                output = self.model(x, seq_lengths, img)
                if self.args.task == "length-of-stay":
                    pred = output['late_average_scores'].squeeze()
                    pred1 = output['cxr_only_scores'].squeeze()
                    pred2 = output['ehr_only_scores'].squeeze()
                    pred3 = output['joint_scores'].squeeze()
                    y_true_bins = torch.tensor([get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()], dtype=torch.long).to(self.device)
                    loss = self.loss(pred, y_true_bins)
                    cxr_loss += self.loss(pred1, y_true_bins).item()
                    ehr_loss += self.loss(pred2, y_true_bins).item()
                    joint_loss += self.loss(pred3, y_true_bins).item()
                else:
                    pred = output['late_average'].squeeze()
                    loss = self.loss(pred, y)
                    pred1 = output['cxr_only'].squeeze()
                    pred2 = output['ehr_only'].squeeze()
                    pred3 = output['joint'].squeeze()
                    cxr_loss += self.loss(pred1, y).item()
                    ehr_loss += self.loss(pred2, y).item()
                    joint_loss += self.loss(pred3, y).item()
                epoch_loss += (loss.item() )#+ loss2.item() + loss3.item())/3)
                outPRED = torch.cat((outPRED, pred), 0)

                outPRED_cxr = torch.cat((outPRED_cxr, pred1), 0)
                outPRED_ehr = torch.cat((outPRED_ehr, pred2), 0)
                outPRED_joint = torch.cat((outPRED_joint, pred3), 0)

                outGT = torch.cat((outGT, y), 0)

                if self.eval_iter is not None and (i+1) % self.eval_iter == 0 and not full_run:
                    break
        if self.args.task == "length-of-stay":
            with torch.no_grad():
                y_true_bins = [get_bin_custom(y_item.item(), CustomBins.nbins) for y_item in outGT.cpu().numpy()]
                pred_labels = torch.max(outPRED, 1)[1].cpu().numpy()  # Convert logits to predicted labels
                cf = metrics.confusion_matrix(y_true_bins, pred_labels)
                kappa = metrics.cohen_kappa_score(y_true_bins, pred_labels, weights='linear')
                mad = metrics.mean_absolute_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
                mse = metrics.mean_squared_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
                mape = mean_absolute_percentage_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
    
                best_stats = {"mad": mad, "mse": mse, "mape": mape, "kappa": kappa}
                wandb.log({
                    'val_mad': mad,
                    'val_mse': mse, 
                    'val_mape': mape,
                    'val_kappa': kappa
                })
            ret = best_stats
            return ret
        else:
            self.epochs_stats['loss val joint'].append(joint_loss/i)
            self.epochs_stats['loss val ehr'].append(ehr_loss/i)
            self.epochs_stats['loss val cxr'].append(cxr_loss/i)
    
            print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")
            ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            ret_ehr = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_ehr.data.cpu().numpy(), 'validation_ehr')
            ret_cxr = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_cxr.data.cpu().numpy(), 'validation_cxr')
            ret_joint = self.computeAUROC(outGT.data.cpu().numpy(), outPRED_joint.data.cpu().numpy(), 'validation_joint')
    
            self.epochs_stats['auroc val ehr'].append(ret_ehr['auroc_mean'])
            self.epochs_stats['auroc val cxr'].append(ret_cxr['auroc_mean'])
            self.epochs_stats['auroc val avg'].append(ret['auroc_mean'])
            self.epochs_stats['auroc val joint'].append(ret_joint['auroc_mean'])
            wandb.log({
                    'val_ehr_auroc': ret_ehr['auroc_mean'],
                    'val_cxr_auroc': ret_cxr['auroc_mean'], 
                    'val_avg_auroc': ret['auroc_mean'],
                    'val_joint_auroc': ret_joint['auroc_mean']
                })

            return {'ehr': ret_ehr, 'cxr': ret_cxr, 'late': ret, 'joint': ret_joint}

    
    def eval(self):
        print('validating ... ')
        self.load_ehr_pheno(load_state=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
        self.load_cxr_pheno(load_state=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
        self.load_state(state_path=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
        
        self.epoch = 0
        self.model.eval()
        # ret = self.validate(self.val_dl, full_run=True)
        # # import pdb; pdb.set_trace()
        # self.print_and_write(ret['joint'] , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val_joint.txt')
        # self.print_and_write(ret['late'] , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val_late.txt')
        # self.print_and_write(ret['cxr'] , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val_cxr.txt')
        # self.print_and_write(ret['ehr'] , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val_ehr.txt')
        # self.model.eval()
        
        ret = self.validate(self.test_dl, full_run=True)
        if self.args.task=="length-of-stay":
            wandb.log({
                    'test mad': ret['mad'], 
                    'test mse': ret['mse'], 
                    'test mape': ret['mape'],
                    'test kappa': ret['kappa']
                })
        else:
            ret_ehr = ret['ehr']
            ret_cxr = ret['cxr']
            ret_late = ret['late']
            ret_joint = ret['joint']
            wandb.log({
                    'test_ehr_auroc': ret_ehr['auroc_mean'],
                    'test_cxr_auroc': ret_cxr['auroc_mean'], 
                    'test_avg_auroc': ret_late['auroc_mean'],
                    'test_joint_auroc': ret_joint['auroc_mean'],
                    'test_ehr_auprc': ret_ehr['auprc_mean'],
                    'test_cxr_auprc': ret_cxr['auprc_mean'], 
                    'test_avg_auprc': ret_late['auprc_mean'],
                    'test_joint_auprc': ret_joint['auprc_mean']
                })
        
        return
        
        # self.print_and_write(ret['joint'] , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test_joint.txt')
        # self.print_and_write(ret['late'] , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test_late.txt')
        # self.print_and_write(ret['cxr'] , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test_cxr.txt')
        # self.print_and_write(ret['ehr'] , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test_ehr.txt')
        
        # return

    
      
    def train(self):
        print(f'running for fusion_type {self.args.fusion_type}')
        
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.eval()
            full_run = True if (self.args.task == 'decompensation' or self.args.task == 'length-of-stay') else True
            ret = self.validate(self.val_dl, full_run=full_run)
            self.model.train()
            self.train_epoch()
            self.save_checkpoint(prefix='last')

            if self.args.task=="length-of-stay":
                if self.best_kappa < ret['kappa']:
                    self.best_kappa = ret['kappa']
                    self.best_stats = ret
                    self.save_checkpoint()
                    # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
                    #self.print_and_write(ret, isbest=True)
                    self.patience = 0
                else:
                    #self.print_and_write(ret, isbest=False)
                    self.patience+=1
            else:
                intrabest = max([ret['late']['auroc_mean'], ret['cxr']['auroc_mean'], ret['ehr']['auroc_mean'], ret['joint']['auroc_mean']])
                if self.best_auroc < intrabest:
                    self.best_auroc = intrabest#ret['auroc_mean']
                    self.best_stats = ret
                    self.save_checkpoint()
                    # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
                    # self.print_and_write(ret['late'], prefix='vallate', isbest=True, filename='results_val_late.txt')
                    # self.print_and_write(ret['joint'], prefix='valjoint', isbest=True, filename='results_val_joint.txt')
                    # self.print_and_write(ret['ehr'], prefix='valehr', isbest=True, filename='results_val_ehr.txt')
                    # self.print_and_write(ret['cxr'], prefix='valcxr', isbest=True, filename='results_val_cxr.txt')
                    self.patience = 0
                else:
                    self.patience+=1
                    # self.print_and_write(ret['late'], prefix='val late', isbest=False)
            
            # self.plot_stats(key='loss', filename='loss.pdf')
            # self.plot_stats(key='auroc', filename='auroc.pdf')
            
            if self.patience >= self.args.patience:
                print("early stopped")
                break
        # self.print_and_write(self.best_stats['late'] , isbest=True, filename='results_val_late.txt')
        # self.print_and_write(self.best_stats['joint'], isbest=True, filename='results_val_joint.txt')
        # self.print_and_write(self.best_stats['ehr'] , isbest=True, filename='results_val_ehr.txt')
        # self.print_and_write(self.best_stats['cxr'] , isbest=True, filename='results_val_cxr.txt')


        
    

