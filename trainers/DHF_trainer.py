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

def relevancy_loss(y_fused_pred, y_true, r_scores, preds):
    # Calculate BCE for the fused prediction
    L_pred = F.binary_cross_entropy_with_logits(y_fused_pred, y_true)
    
    # Initialize total loss with L_pred
    total_loss = L_pred

    # Calculate and add L_{r_i} and BCE for each modality
    for modality in preds:
        y_pred = preds[modality]
        r_score = r_scores[modality].squeeze()

        # BCE for the modality prediction
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        
        # Ensure bce_loss and r_score have compatible shapes
        if bce_loss.shape != r_score.shape:
            r_score = r_score.view(bce_loss.shape)  # Adjust the shape of r_score if needed

        # Regression loss between BCE loss and r_score (assuming r_score is logits)
        regression_loss = F.mse_loss(bce_loss, r_score, reduction='none')
        
        # Accumulate the individual BCE losses
        total_loss += bce_loss.sum()

        # Accumulate the regression losses
        total_loss += 0.1*regression_loss.sum()

    return total_loss



class DHFTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl
        ):

        super(DHFTrainer, self).__init__(args)
        run = wandb.init(project=f'DHF_{self.args.H_mode}_{self.args.task}', config=args)
        self.epoch = 0 
        self.start_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.token_dim = 384
        
        self.seed = 1002
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.token_vector = torch.nn.Parameter(torch.randn(self.token_dim).to(self.device))
        #self.token_vector_expanded = self.token_vector.unsqueeze(0).repeat(self.args.batch_size,1, 1)
        
        self.cls_fusion = torch.nn.Parameter(torch.randn(1, 1, self.token_dim).to(self.device))
        self.cls_tokens_expanded = self.cls_fusion.expand(self.args.batch_size, -1, -1)


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
        
        self.ehr_r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.cxr_r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.rr_r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.dn_r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        self.r_classifier = MLPClassifier(input_dim=384, output_dim=1).to(self.device)
        
        self.ehr_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        self.cxr_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        self.rr_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        self.dn_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        
        self.classifier_1 = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        self.classifier_2 = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        self.classifier_3 = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        self.classifier_4 = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        
        self.classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        
        
        self.final_classifier = MLPClassifier(input_dim=384, output_dim=self.args.num_classes).to(self.device)
        
        # Initialize transformer layers
        self.transformer_layer1 = CustomTransformerLayer(input_dim=384, model_dim=384, nhead=2, num_layers=1).to(self.device)
        self.transformer_layer2 = CustomTransformerLayer(input_dim=384, model_dim=384, nhead=2, num_layers=1).to(self.device)
        if self.args.task == 'in-hospital-mortality':
            self.transformer_layer3 = CustomTransformerLayer(input_dim=384, model_dim=384, nhead=4, num_layers=1).to(self.device)
        else:
            self.transformer_layer3 = CustomTransformerLayer(input_dim=384, model_dim=384, nhead=2, num_layers=1).to(self.device)
        self.transformer_layer4 = CustomTransformerLayer(input_dim=384, model_dim=384, nhead=4, num_layers=1).to(self.device)

        if self.args.mode == 'relevancy-based-hierarchical':
            self.loss = relevancy_loss
            all_params = (
            list(self.ehr_encoder.parameters()) +
            list(self.cxr_encoder.parameters()) +
            list(self.dn_encoder.parameters()) +
            list(self.rr_encoder.parameters()) +
            list(self.ehr_r_classifier.parameters()) +
            list(self.cxr_r_classifier.parameters()) +
            list(self.dn_r_classifier.parameters()) +
            list(self.rr_r_classifier.parameters()) +
            list(self.r_classifier.parameters()) +
            list(self.classifier.parameters()) +
            list(self.ehr_classifier.parameters()) +
            list(self.cxr_classifier.parameters()) +
            list(self.dn_classifier.parameters()) +
            list(self.rr_classifier.parameters()) +
            list(self.transformer_layer1.parameters()) +
            list(self.transformer_layer2.parameters()) +
            list(self.transformer_layer3.parameters()) +
            list(self.transformer_layer4.parameters()) +
            list(self.final_classifier.parameters())+
            [self.token_vector, self.cls_fusion ]
        )
        else:
            self.loss = nn.BCEWithLogitsLoss()
            all_params = (
            list(self.ehr_encoder.parameters()) +
            list(self.cxr_encoder.parameters()) +
            list(self.dn_encoder.parameters()) +
            list(self.rr_encoder.parameters()) +
            list(self.transformer_layer1.parameters()) +
            list(self.transformer_layer2.parameters()) +
            list(self.transformer_layer3.parameters()) +
            list(self.transformer_layer4.parameters()) +
            list(self.classifier_1.parameters()) +
            list(self.classifier_2.parameters()) +
            list(self.classifier_3.parameters()) +
            list(self.classifier_4.parameters()) +
            list(self.final_classifier.parameters())+
            [self.token_vector, self.cls_fusion]
        )

        
        self.optimizer = optim.Adam(all_params, lr=args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
        
        if self.args.load_ehr:
            checkpoint = torch.load(args.load_ehr)
            self.ehr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.ehr_classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("ehr loaded")
        
        if self.args.load_cxr:
            checkpoint = torch.load(args.load_cxr)
            self.cxr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.cxr_classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("cxr loaded")
        
        if self.args.load_dn:
            checkpoint = torch.load(args.load_dn)
            self.dn_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.dn_classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("dd loaded")
        
        if self.args.load_rr:
            checkpoint = torch.load(args.load_rr)
            self.rr_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.rr_classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("rr loaded")

        
    def save_DHF_checkpoint(self):
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
            'ehr_r_classifier_state_dict': self.ehr_r_classifier.state_dict(),
            'cxr_r_classifier_state_dict': self.cxr_r_classifier.state_dict(),
            'dn_r_classifier_state_dict': self.dn_r_classifier.state_dict(),
            'rr_r_classifier_state_dict': self.rr_r_classifier.state_dict(),
            'ehr_classifier_state_dict': self.ehr_classifier.state_dict(),
            'cxr_classifier_state_dict': self.cxr_classifier.state_dict(),
            'dn_classifier_state_dict': self.dn_classifier.state_dict(),
            'rr_classifier_state_dict': self.rr_classifier.state_dict(),
            'r_classifier_state_dict': self.r_classifier.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'classifier_1_state_dict': self.classifier_1.state_dict(),
            'classifier_2_state_dict': self.classifier_2.state_dict(),
            'classifier_3_state_dict': self.classifier_3.state_dict(),
            'classifier_4_state_dict': self.classifier_4.state_dict(),
            'final_classifier_state_dict': self.final_classifier.state_dict(),
            'transformer_layer1_state_dict': self.transformer_layer1.state_dict(),
            'transformer_layer2_state_dict': self.transformer_layer2.state_dict(),
            'transformer_layer3_state_dict': self.transformer_layer3.state_dict(),
            'transformer_layer4_state_dict': self.transformer_layer4.state_dict(),
            'token_vector': self.token_vector,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.H_mode}_{self.args.order}_single_r.pth.tar')


    def load_DHF_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        self.ehr_encoder.load_state_dict(checkpoint['ehr_encoder_state_dict'])
        self.cxr_encoder.load_state_dict(checkpoint['cxr_encoder_state_dict'])
        self.dn_encoder.load_state_dict(checkpoint['dn_encoder_state_dict'])
        self.rr_encoder.load_state_dict(checkpoint['rr_encoder_state_dict'])
        self.ehr_r_classifier.load_state_dict(checkpoint['ehr_r_classifier_state_dict'])
        self.cxr_r_classifier.load_state_dict(checkpoint['cxr_r_classifier_state_dict'])
        self.dn_r_classifier.load_state_dict(checkpoint['dn_r_classifier_state_dict'])
        self.rr_r_classifier.load_state_dict(checkpoint['rr_r_classifier_state_dict'])
        self.ehr_classifier.load_state_dict(checkpoint['ehr_classifier_state_dict'])
        self.cxr_classifier.load_state_dict(checkpoint['cxr_classifier_state_dict'])
        self.dn_classifier.load_state_dict(checkpoint['dn_classifier_state_dict'])
        self.rr_classifier.load_state_dict(checkpoint['rr_classifier_state_dict'])
        self.r_classifier.load_state_dict(checkpoint['r_classifier_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier_1.load_state_dict(checkpoint['classifier_1_state_dict'])
        self.classifier_2.load_state_dict(checkpoint['classifier_2_state_dict'])
        self.classifier_3.load_state_dict(checkpoint['classifier_3_state_dict'])
        self.classifier_4.load_state_dict(checkpoint['classifier_4_state_dict'])
        self.final_classifier.load_state_dict(checkpoint['final_classifier_state_dict'])
        self.transformer_layer1.load_state_dict(checkpoint['transformer_layer1_state_dict'])
        self.transformer_layer2.load_state_dict(checkpoint['transformer_layer2_state_dict'])
        self.transformer_layer3.load_state_dict(checkpoint['transformer_layer3_state_dict'])
        self.transformer_layer4.load_state_dict(checkpoint['transformer_layer4_state_dict'])
        self.token_vector = checkpoint['token_vector']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def set_train_mode(self):
        """Set all neural network components to training mode."""
        self.ehr_encoder.train()
        self.cxr_encoder.train()
        self.dn_encoder.train()
        self.rr_encoder.train()

        self.ehr_r_classifier.train()
        self.cxr_r_classifier.train()
        self.dn_r_classifier.train()
        self.rr_r_classifier.train()
        self.r_classifier.train()
        self.classifier.train()
        
        self.classifier_1.train()
        self.classifier_2.train()
        self.classifier_3.train()
        self.classifier_4.train()

        self.ehr_classifier.train()
        self.cxr_classifier.train()
        self.dn_classifier.train()
        self.rr_classifier.train()
        self.final_classifier.train()

        self.transformer_layer1.train()
        self.transformer_layer2.train()
        self.transformer_layer3.train()
        self.transformer_layer4.train()

    def set_eval_mode(self):
        """Set all neural network components to evaluation mode."""
        self.ehr_encoder.eval()
        self.cxr_encoder.eval()
        self.dn_encoder.eval()
        self.rr_encoder.eval()

        self.ehr_r_classifier.eval()
        self.cxr_r_classifier.eval()
        self.dn_r_classifier.eval()
        self.rr_r_classifier.eval()
        self.r_classifier.eval()
        self.classifier.eval()

        self.ehr_classifier.eval()
        self.cxr_classifier.eval()
        self.dn_classifier.eval()
        self.rr_classifier.eval()
        self.final_classifier.eval()
        
        self.classifier_1.train()
        self.classifier_2.train()
        self.classifier_3.train()
        self.classifier_4.train()

        self.transformer_layer1.eval()
        self.transformer_layer2.eval()
        self.transformer_layer3.eval()
        self.transformer_layer4.eval()
    
    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
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
            
            vectors = {}
            r_scores = {}
            preds = {}
            stage_predictions = []

            if 'EHR' in self.args.modalities:
                v_ehr,cls_ehr = self.ehr_encoder(x)
                vectors['EHR'] = v_ehr
                r_ehr = self.r_classifier(cls_ehr)
                r_scores['EHR'] = r_ehr
                y_ehr_pred = self.classifier(cls_ehr)
                preds['EHR'] = y_ehr_pred
            if 'CXR' in self.args.modalities:
                v_cxr,cls_cxr = self.cxr_encoder(img)
                vectors['CXR'] = v_cxr
                r_cxr = self.r_classifier(cls_cxr)
                r_scores['CXR'] = r_cxr
                y_cxr_pred = self.classifier(cls_cxr)
                preds['CXR'] = y_cxr_pred
            if 'DN' in self.args.modalities:
                v_dn, cls_dn = self.dn_encoder(dn)
                vectors['DN'] = v_dn
                r_dn = self.r_classifier(cls_dn)
                r_scores['DN'] = r_dn
                y_dn_pred = self.classifier(cls_dn)
                preds['DN'] = y_dn_pred
            if 'RR' in self.args.modalities:
                v_rr, cls_rr = self.rr_encoder(rr)
                vectors['RR'] = v_rr
                r_rr = self.r_classifier(cls_rr)
                r_scores['RR'] = r_rr
                y_rr_pred = self.classifier(cls_rr)
                preds['RR'] = y_rr_pred

            if self.args.H_mode == 'relevancy-based-hierarchical':
                modalities_list = self.args.modalities.split('-')
                scores_tensor = torch.stack([r_scores[mod].squeeze() for mod in modalities_list], dim=0)
                sorted_scores, sorted_indices = torch.sort(scores_tensor, dim=0, descending=False)
                sorted_modalities = [modalities_list[idx] for idx in sorted_indices.cpu().numpy()]
                #print(sorted_modalities)
                
            elif self.args.H_mode == 'predefined-hierarchical' and self.args.order is not None:
                # Use predefined order
                order_list = self.args.order.split('-')
                #print(order_list)
                sorted_modalities = [mod for mod in order_list]
                
            batch_size = vectors[sorted_modalities[0]].size(0)
            token_vector_expanded = self.token_vector.unsqueeze(0).repeat(batch_size, 1, 1)
            fused_vector = torch.cat((vectors[sorted_modalities[0]], token_vector_expanded), dim=1)
                
            fused_vector = self.transformer_layer1(fused_vector)
            
            y_1 = self.classifier_1(fused_vector[:, 0, :])
            stage_predictions.append(y_1)


            for idx, modality in enumerate(sorted_modalities[1:], 2):
                # Concatenate the modality vector to the fused vector
                fused_vector = torch.cat((fused_vector, vectors[modality]), dim=1)
                
                # Get the current transformer layer
                transformer_layer = getattr(self, f'transformer_layer{idx}')
                
                # Apply the transformer layer
                fused_vector = transformer_layer(fused_vector)
                
                class_layer = getattr(self, f'classifier_{idx}')
                y_stage=class_layer(fused_vector[:, 0, :])
                stage_predictions.append(y_stage)
    
            # Final classifier
            #y_fused_pred = self.final_classifier(fused_vector[:, 0, :])
            
            y_mean_pred = torch.mean(torch.stack(stage_predictions), dim=0)
            
            if self.args.H_mode == 'relevancy-based-hierarchical':
                loss = relevancy_loss(y_fused_pred, y, r_scores, preds)
            else:
                loss = self.loss(y_mean_pred, y)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            outPRED = torch.cat((outPRED, y_mean_pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}")
        

        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
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
    
        with torch.no_grad():
            for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (dl):
                #print("seq_lengths:",seq_lengths)
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = x.to(self.device)
                y = y.to(self.device)
                if self.args.task == 'in-hospital-mortality':
                    y = y.unsqueeze(1)
                img = img.to(self.device)
                
                vectors = {}
                r_scores = {}
                preds = {}
                stage_predictions = []
    
                if 'EHR' in self.args.modalities:
                    v_ehr,cls_ehr = self.ehr_encoder(x)
                    vectors['EHR'] = v_ehr
                    r_ehr = self.r_classifier(cls_ehr)
                    r_scores['EHR'] = r_ehr
                    y_ehr_pred = self.classifier(cls_ehr)
                    preds['EHR'] = y_ehr_pred
                if 'CXR' in self.args.modalities:
                    v_cxr,cls_cxr = self.cxr_encoder(img)
                    vectors['CXR'] = v_cxr
                    r_cxr = self.r_classifier(cls_cxr)
                    r_scores['CXR'] = r_cxr
                    y_cxr_pred = self.classifier(cls_cxr)
                    preds['CXR'] = y_cxr_pred
                if 'DN' in self.args.modalities:
                    v_dn, cls_dn = self.dn_encoder(dn)
                    vectors['DN'] = v_dn
                    r_dn = self.r_classifier(cls_dn)
                    r_scores['DN'] = r_dn
                    y_dn_pred = self.classifier(cls_dn)
                    preds['DN'] = y_dn_pred
                if 'RR' in self.args.modalities:
                    v_rr, cls_rr = self.rr_encoder(rr)
                    vectors['RR'] = v_rr
                    r_rr = self.r_classifier(cls_rr)
                    r_scores['RR'] = r_rr
                    y_rr_pred = self.classifier(cls_rr)
                    preds['RR'] = y_rr_pred
    
                if self.args.H_mode == 'relevancy-based-hierarchical':
                    modalities_list = self.args.modalities.split('-')
                    scores_tensor = torch.stack([r_scores[mod].squeeze() for mod in modalities_list], dim=0)
                    sorted_scores, sorted_indices = torch.sort(scores_tensor, dim=0, descending=True)
                    sorted_modalities = [modalities_list[idx] for idx in sorted_indices.cpu().numpy()]
                    
                elif self.args.H_mode == 'predefined-hierarchical' and self.args.order is not None:
                    # Use predefined order
                    order_list = self.args.order.split('-')
                    #print(order_list)
                    sorted_modalities = [mod for mod in order_list]
                    
                #print(f"Shape of vectors[{sorted_modalities[0]}]: {vectors[sorted_modalities[0]].shape}")
                #print(f"Shape of self.token_vector_expanded: {self.token_vector_expanded.shape}")
                
                batch_size = vectors[sorted_modalities[0]].size(0)
                token_vector_expanded = self.token_vector.unsqueeze(0).repeat(batch_size, 1, 1)

                fused_vector = torch.cat((vectors[sorted_modalities[0]], token_vector_expanded), dim=1)
                

                
                fused_vector = self.transformer_layer1(fused_vector)
                
                
                y_1 = self.classifier_1(fused_vector[:, 0, :])
                stage_predictions.append(y_1)
    
    
                for idx, modality in enumerate(sorted_modalities[1:], 2):
                    # Concatenate the modality vector to the fused vector
                    fused_vector = torch.cat((fused_vector, vectors[modality]), dim=1)
                    
                    # Get the current transformer layer
                    transformer_layer = getattr(self, f'transformer_layer{idx}')
                    
                    # Apply the transformer layer
                    fused_vector = transformer_layer(fused_vector)
                    
                    class_layer = getattr(self, f'classifier_{idx}')
                    y_stage=class_layer(fused_vector[:, 0, :])
                    stage_predictions.append(y_stage)
        
                # Final classifier
                #y_fused_pred = self.final_classifier(fused_vector[:, 0, :])
                
                y_mean_pred = torch.mean(torch.stack(stage_predictions), dim=0)
                
                if self.args.H_mode == 'relevancy-based-hierarchical':
                    #print("ok r loss")
                    loss = relevancy_loss(y_fused_pred, y, r_scores, preds)
                else:
                    loss = self.loss(y_mean_pred, y)
                    
                epoch_loss += loss.item()
                outPRED = torch.cat((outPRED, y_mean_pred), 0)
                outGT = torch.cat((outGT, y), 0)
    
            print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")
    
            ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy())
            np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy())
            # self.epochs_stats['auroc val'].append(ret['auroc_mean'])
            # self.epochs_stats['loss val'].append(epoch_loss / i)
            wandb.log({
                'val_Loss': epoch_loss / i,
                'val_AUC': ret['auroc_mean']
            })
    
        return ret

    def eval(self):
        
        self.load_DHF_checkpoint(f'{self.args.save_dir}/{self.args.task}/{self.args.H_mode}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.H_mode}_{self.args.order}_single_r.pth.tar')
        
        self.epoch = 0
        self.set_eval_mode() 

        ret = self.validate(self.test_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
        wandb.log({
                'test_auprc': ret['auprc_mean'], 
                'test_AUC': ret['auroc_mean']
            })
            # if self.args.task!="length-of-stay":
            #     self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
        return
    
    def train(self):
        print(f'running for fusion_type {self.args.fusion_type}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            print(self.epoch)
            self.set_eval_mode() 
            ret = self.validate(self.val_dl)
            #self.save_checkpoint(prefix='last')
    
            if self.best_auroc < ret['auroc_mean']:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_DHF_checkpoint()
                print("checkpoint")
                self.patience = 0
            else:
                self.patience += 1
            
            if self.patience >= self.args.patience:
                break
            
            self.set_train_mode() 
            self.train_epoch()

        
    

