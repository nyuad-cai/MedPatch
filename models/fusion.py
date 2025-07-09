import torch
import torch.nn as nn
from .classifier import Classifier  # Import from classifier.py
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import timm
import random

from .ehr_models import EHR_encoder
from .CXR_models import CXR_encoder
from .text_models import Text_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from .healnet import HealNet
from typing import List, Dict, Tuple, Optional

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Fusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.text_model = text_model

        # Validate modalities in args
        assert hasattr(args, "modalities"), "args.modalities is required"
        self.modalities = args.modalities

        # Initialize based on fusion type
        fusion_type = args.fusion_type
        if fusion_type == 'early':
            self.fusion_model = EarlyFusion(args, ehr_model, cxr_model, text_model)
        elif fusion_type == 'joint':
            self.fusion_model = JointFusion(args, ehr_model, cxr_model, text_model)
        elif fusion_type == 'late':
            self.fusion_model = LateFusion(args, ehr_model, cxr_model, text_model)
        elif fusion_type == 'lstm':
            self.fusion_model = LSTMFusion(args, ehr_model, cxr_model, text_model)
        elif fusion_type == 'unimodal_ehr':
            self.fusion_model = UnimodalEHR(args, ehr_model)
        elif fusion_type == 'unimodal_cxr':
            self.fusion_model = UnimodalCXR(args, cxr_model)
        elif fusion_type == 'unimodal_rr':
            self.fusion_model = UnimodalRR(args, text_model)
        elif fusion_type == 'unimodal_dn':
            self.fusion_model = UnimodalDN(args, text_model)
        elif fusion_type == 'c-unimodal_ehr':
            self.fusion_model = UnimodalEHRConfidence(args, ehr_model)
        elif fusion_type == 'c-unimodal_cxr':
            self.fusion_model = UnimodalCXRConfidence(args, cxr_model)
        elif fusion_type == 'c-unimodal_rr':
            self.fusion_model = UnimodalRRConfidence(args, text_model)
        elif fusion_type == 'c-unimodal_dn':
            self.fusion_model = UnimodalDNConfidence(args, text_model)
        elif fusion_type == 'temp_c-unimodal_ehr':
            self.fusion_model = TempCUnimodalEHR(args, ehr_model)
        elif fusion_type == 'temp_c-unimodal_cxr':
            self.fusion_model = TempCUnimodalCXR(args, cxr_model)
        elif fusion_type == 'temp_c-unimodal_rr':
            self.fusion_model = TempCUnimodalRR(args, text_model)
        elif fusion_type == 'temp_c-unimodal_dn':
            self.fusion_model = TempCUnimodalDN(args, text_model)
        elif fusion_type == 'metra':
            self.fusion_model = MeTraTransformer(args, ehr_model, cxr_model, text_model)
        elif fusion_type == 'msma':
            if self.args.ablation:
                print('we are performing ', self.args.ablation)
                self.fusion_model = AblationMSMAFusion(args, ehr_model, cxr_model, text_model)
            else:
                self.fusion_model = NovelMSMAFusion(args, ehr_model, cxr_model, text_model)
        elif fusion_type == 'c-msma':
                self.fusion_model = CMSMAFusion(args, ehr_model, cxr_model, text_model)
        elif fusion_type == 'c-e-msma':
                self.fusion_model = EMSMAFusion(args, ehr_model, cxr_model, text_model)
        elif fusion_type == 'ensemble':
            self.fusion_model = EnsembleFusion(args)
        elif fusion_type == 'healnet-raw':
            # Define your modality specifics based on your models' outputs:
            modalities = args.modalities.split("-")
            n_modalities = len(modalities)
            channel_dims: List[int] = []
            num_spatial_axes: List[int] = []
            
            for mod in modalities:
                if mod == "EHR":
                    ehr_channels = 48 if args.task == "in-hospital-mortality" else 2646
                    channel_dims.append(ehr_channels)
                    num_spatial_axes.append(1)  # sequential
                elif mod == "CXR":
                    channel_dims.append(3)
                    num_spatial_axes.append(2)  # ViT-style patch sequence
                elif mod == "DN":
                    channel_dims.append(text_model.full_feats_dim_dn)
                    num_spatial_axes.append(1)  # 512 tokens
                elif mod == "RR":
                    channel_dims.append(text_model.full_feats_dim_rr)
                    num_spatial_axes.append(1)  # 512 tokens
                else:
                    raise ValueError(f"Unsupported modality: {mod}")
                    
            healnet_model = HealNet(
                n_modalities=n_modalities,
                channel_dims=channel_dims,
                num_spatial_axes=num_spatial_axes,
                out_dims=args.num_classes,  # task-specific output dimension
                depth=3,
                num_freq_bands=4,
                max_freq=10.,
                l_c=128,
                l_d=128,
                x_heads=8,
                l_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                attn_dropout=0.1,
                ff_dropout=0.1,
                weight_tie_layers=False,
                fourier_encode_data=True,
                self_per_cross_attn=1,
                final_classifier_head=True,
                snn=True
            )
          
            self.fusion_model = HealNetRawFusion(args, ehr_model=ehr_model, cxr_model=cxr_model, text_model=text_model, healnet=healnet_model, modalities=modalities)
            
        elif fusion_type == 'healnet':
            # Define your modality specifics based on your models' outputs:
            modalities = args.modalities.split("-")
            n_modalities = len(modalities)
            channel_dims: List[int] = []
            num_spatial_axes: List[int] = []
            
            for mod in modalities:
                if mod == "EHR":
                    channel_dims.append(ehr_model.full_feats_dim)
                    num_spatial_axes.append(1)  # sequential
                elif mod == "CXR":
                    channel_dims.append(cxr_model.full_feats_dim)
                    num_spatial_axes.append(1)  # ViT-style patch sequence
                elif mod == "DN":
                    channel_dims.append(text_model.full_feats_dim_dn)
                    num_spatial_axes.append(1)  # 512 tokens
                elif mod == "RR":
                    channel_dims.append(text_model.full_feats_dim_rr)
                    num_spatial_axes.append(1)  # 512 tokens
                else:
                    raise ValueError(f"Unsupported modality: {mod}")
            
            healnet_model = HealNet(
                n_modalities=n_modalities,
                channel_dims=channel_dims,
                num_spatial_axes=num_spatial_axes,
                out_dims=args.num_classes,  # task-specific output dimension
                depth=3,
                num_freq_bands=4,
                max_freq=10.,
                l_c=128,
                l_d=128,
                x_heads=8,
                l_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                attn_dropout=0.1,
                ff_dropout=0.1,
                weight_tie_layers=False,
                fourier_encode_data=True,
                self_per_cross_attn=1,
                final_classifier_head=True,
                snn=True
            )
            self.fusion_model = HealNetBaselineFusion(args, ehr_model=ehr_model, cxr_model=cxr_model, text_model=text_model, healnet=healnet_model, modalities=modalities)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, *args, **kwargs):
        return self.fusion_model(*args, **kwargs)
        
    def equalize(self):
        return self.fusion_model.equalize()


class EarlyFusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, text_model):
        super(EarlyFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities

        # Encoders
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['RR', 'DN']) else None

        # Freeze the encoders
        self.freeze_encoders()

        # Calculate total feature dimension
        total_feats_dim = 0
        
        if self.ehr_model:
            total_feats_dim += self.ehr_model.feats_dim
        if self.cxr_model:
            total_feats_dim += self.cxr_model.feats_dim
        if self.text_model:
            # Only add dimensions for available features
            if 'DN' in self.modalities:
                total_feats_dim += self.text_model.feats_dim_dn
            if 'RR' in self.modalities:
                total_feats_dim += self.text_model.feats_dim_rr
                
        self.total_feats_dim = total_feats_dim

        # Define classifier dynamically
        self.early_classifier = Classifier(total_feats_dim, self.args)

    def freeze_encoders(self):
        """Freeze all encoder parameters to prevent training."""
        if self.ehr_model:
            for param in self.ehr_model.parameters():
                param.requires_grad = False
        if self.cxr_model:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
        if self.text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        features = []

        # Get EHR features
        if self.ehr_model and x is not None:
            ehr_feats = self.ehr_model(x, seq_lengths)
            # ehr_feats = ehr_feats.unsqueeze(1)
            features.append(ehr_feats)

        # Get CXR features
        if self.cxr_model and img is not None:
            cxr_feats = self.cxr_model(img)
            if self.args.use_cls_token == 'cls':
                # Use the first token (CLS token)
                cxr_feats = cxr_feats[:, 0, :]
            else:
                # Use mean pooling
                cxr_feats = cxr_feats.mean(dim=1)
            features.append(cxr_feats)

        # Get text features
        if self.text_model:
            dn_feats, rr_feats = None, None
            if 'DN' in self.modalities and 'RR' in self.modalities:
                # print(f"DN: {dn}")
                # print(f"RR: {rr}")
                dn_feats, rr_feats = self.text_model(dn_notes=dn, rr_notes = rr)
            elif 'DN' in self.modalities and dn is not None:
                dn_feats, _ = self.text_model(dn_notes=dn)
                dn_feats = dn_feats.mean(dim=1)
                features.append(dn_feats)
            elif 'RR' in self.modalities and rr is not None:
                _, rr_feats = self.text_model(rr_notes = rr)
                rr_feats = rr_feats.mean(dim=1)
                features.append(rr_feats)

        # Combine all features
        if features:
            combined_feats = torch.cat(features, dim=1)
            if combined_feats.shape[1] < self.total_feats_dim:
                padding = torch.zeros(combined_feats.shape[0], self.total_feats_dim - combined_feats.shape[1]).to(combined_feats.device)
                combined_feats = torch.cat([combined_feats, padding], dim=1)
            output = self.early_classifier(combined_feats)
        else:
            raise ValueError("No modalities available for fusion!")

        return {'early': output, 'unified': output}

class JointFusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, text_model):
        super(JointFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities

        # Encoders
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['RR', 'DN']) else None

        # Calculate total feature dimension
        total_feats_dim = 0
        
        if self.ehr_model:
            total_feats_dim += self.ehr_model.feats_dim
        if self.cxr_model:
            total_feats_dim += self.cxr_model.feats_dim
        if self.text_model:
            # Only add dimensions for available features
            if 'DN' in self.modalities:
                total_feats_dim += self.text_model.feats_dim_dn
            if 'RR' in self.modalities:
                total_feats_dim += self.text_model.feats_dim_rr
                
        self.total_feats_dim = total_feats_dim

        # Define classifier dynamically
        self.joint_classifier = Classifier(total_feats_dim, self.args)

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        features = []

        # Get EHR features
        if self.ehr_model and x is not None:
            ehr_feats = self.ehr_model(x, seq_lengths)
            # ehr_feats = ehr_feats.unsqueeze(1)
            features.append(ehr_feats)

        # Get CXR features
        if self.cxr_model and img is not None:
            cxr_feats = self.cxr_model(img)
            if self.args.use_cls_token == 'cls':
                # Use the first token (CLS token)
                cxr_feats = cxr_feats[:, 0, :]
            else:
                # Use mean pooling
                cxr_feats = cxr_feats.mean(dim=1)
            features.append(cxr_feats)

        # Get text features
        if self.text_model:
            dn_feats, rr_feats = None, None
            if 'DN' in self.modalities and 'RR' in self.modalities:
                dn_feats, rr_feats = self.text_model(dn_notes=dn, rr_notes = rr)
            elif 'DN' in self.modalities and dn is not None:
                dn_feats, _ = self.text_model(dn_notes=dn)
                dn_feats = dn_feats.mean(dim=1)
                features.append(dn_feats)
            elif 'RR' in self.modalities and rr is not None:
                _, rr_feats = self.text_model(rr_notes = rr)
                rr_feats = rr_feats.mean(dim=1)
                features.append(rr_feats)

        # Combine all features
        if features:
            combined_feats = torch.cat(features, dim=1)
            if combined_feats.shape[1] < self.total_feats_dim:
                padding = torch.zeros(combined_feats.shape[0], self.total_feats_dim - combined_feats.shape[1]).to(combined_feats.device)
                combined_feats = torch.cat([combined_feats, padding], dim=1)
            output = self.joint_classifier(combined_feats)
        else:
            raise ValueError("No modalities available for fusion!")

        return {'joint': output, 'unified': output}

class LateFusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, text_model):
        super(LateFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities

        # Encoders
        self.ehr_model = ehr_model if 'EHR' in args.modalities else None
        self.cxr_model = cxr_model if 'CXR' in args.modalities else None
        self.text_model = text_model if any(m in args.modalities for m in ['RR', 'DN']) else None

        # Define classifiers for each modality
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, args) if self.ehr_model else None
        self.cxr_classifier = Classifier(self.cxr_model.feats_dim, args) if self.cxr_model else None
        self.dn_classifier = Classifier(self.text_model.feats_dim_dn, args) if self.text_model and 'DN' in args.modalities else None
        self.rr_classifier = Classifier(self.text_model.feats_dim_rr, args) if self.text_model and 'RR' in args.modalities else None

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        preds = []

        # Get EHR predictions
        if self.ehr_model and x is not None:
            ehr_feats = self.ehr_model(x, seq_lengths)
            ehr_pred = self.ehr_classifier(ehr_feats)
            preds.append(ehr_pred)

        # Get CXR predictions
        if self.cxr_model and img is not None:
            cxr_feats = self.cxr_model(img)
            if self.args.use_cls_token == 'cls':
                # Use the first token (CLS token)
                cxr_feats = cxr_feats[:, 0, :]
            else:
                # Use mean pooling
                cxr_feats = cxr_feats.mean(dim=1)
            cxr_pred = self.cxr_classifier(cxr_feats)
            preds.append(cxr_pred)

        # Get DN predictions
        if self.text_model and 'DN' in self.modalities and dn is not None:
            dn_feats, _ = self.text_model(dn_notes=dn)
            dn_feats = dn_feats.mean(dim=1)
            dn_pred = self.dn_classifier(dn_feats)
            preds.append(dn_pred)

        # Get RR predictions
        if self.text_model and 'RR' in self.modalities and rr is not None:
            _, rr_feats = self.text_model(rr_notes = rr)
            rr_feats = rr_feats.mean(dim=1)
            rr_pred = self.rr_classifier(rr_feats)
            preds.append(rr_pred)

        if preds:
            # Combine predictions by averaging
            combined_pred = sum(preds) / len(preds)
        else:
            raise ValueError("No modalities available for fusion!")

        return {'late': combined_pred, 'unified': combined_pred}

class LSTMFusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, text_model):
        super(LSTMFusion, self).__init__()
        self.args = args

        # Encoders
        self.ehr_model = ehr_model if 'EHR' in args.modalities else None
        self.cxr_model = cxr_model if 'CXR' in args.modalities else None
        self.text_model = text_model if any(m in args.modalities for m in ['RR', 'DN']) else None

        # Determine LSTM input size
        lstm_in = 0
        if self.ehr_model:
            lstm_in += self.ehr_model.feats_dim
        if self.cxr_model:
            lstm_in += self.cxr_model.feats_dim
        if self.text_model:
            if 'DN' in self.modalities:
                lstm_in += self.text_model.feats_dim_dn
            if 'RR' in self.modalities:
                lstm_in += self.text_model.feats_dim_rr

        self.lstm_fusion_layer = nn.LSTM(
            input_size=lstm_in,
            hidden_size=args.lstm_hidden_size,
            batch_first=True,
            dropout=args.dropout
        )

        # Define classifier dynamically
        self.lstm_classifier = Classifier(args.lstm_hidden_size, self.args)

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        features = []

        # Get EHR features
        if self.ehr_model and x is not None:
            ehr_feats = self.ehr_model(x, seq_lengths)
            features.append(ehr_feats)

        # Get CXR features
        if self.cxr_model and img is not None:
            cxr_feats = self.cxr_model(img)
            features.append(cxr_feats)

        # Get text features
        if self.text_model:
            dn_feats, rr_feats = None, None
            if 'DN' in self.modalities and 'RR' in self.modalities:
                dn_feats, rr_feats = self.text_model(dn_notes=dn, rr_notes = rr)
            elif 'DN' in self.modalities and dn is not None:
                dn_feats, _ = self.text_model(dn_notes=dn)
            elif 'RR' in self.modalities and rr is not None:
                _, rr_feats = self.text_model(rr_notes = rr)

            if dn_feats is not None and rr_feats is not None:
                text_feats = torch.cat((dn_feats, rr_feats), dim=1)
            elif dn_feats is not None:
                text_feats = dn_feats
            elif rr_feats is not None:
                text_feats = rr_feats
            else:
                text_feats = None

            if text_feats is not None:
                features.append(text_feats)

        # Create feature sequence for LSTM
        if features:
            features_seq = torch.stack(features, dim=1)
            _, (ht, _) = self.lstm_fusion_layer(features_seq)
            output = self.lstm_classifier(ht.squeeze(0))
        else:
            raise ValueError("No modalities available for fusion!")

        return {'lstm': output, 'unified': output}

class UnimodalEHR(nn.Module):
    def __init__(self, args, ehr_model):
        super(UnimodalEHR, self).__init__()
        self.args = args

        # EHR Encoder
        self.ehr_model = ehr_model
        if not self.ehr_model:
            raise ValueError("EHR encoder must be provided for UnimodalEHR!")
            
        # Define a classifier for EHR
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args)


    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if x is None:
            raise ValueError("EHR data (x) must be provided for UnimodalEHR!")

        # Pass data through the encoder
        ehr_feats = self.ehr_model(x, seq_lengths)

        # Generate predictions using the classifier
        output = self.ehr_classifier(ehr_feats)

        return {'unimodal_ehr': output, 'unified': output}

class UnimodalCXR(nn.Module):
    def __init__(self, args, cxr_model):
        super(UnimodalCXR, self).__init__()
        self.args = args
        
        # CXR Encoder
        self.cxr_model = cxr_model

        if not self.cxr_model:
            raise ValueError("CXR encoder must be provided for UnimodalCXR!")
        
        # Define a classifier for CXR
        self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        if img is None:
            raise ValueError("CXR data (img) must be provided for UnimodalCXR!")
        
        # Pass data through the encoder
        cxr_feats = self.cxr_model(img)  # Shape: [B, 577, feats_dim]
        if self.args.use_cls_token == 'cls':
            # Use the first token (CLS token)
            cxr_feats = cxr_feats[:, 0, :]
        else:
            # Use mean pooling
            cxr_feats = cxr_feats.mean(dim=1)
        
        output = self.cxr_classifier(cxr_feats)
        return {'unimodal_cxr': output, 'unified': output}
        
class UnimodalRR(nn.Module):
    def __init__(self, args, text_model):
        super(UnimodalRR, self).__init__()
        self.args = args

        # Text Encoder
        self.text_model = text_model
        if not self.text_model or 'RR' not in self.args.modalities:
            raise ValueError("Text encoder with 'RR' modality must be provided for UnimodalRR!")

        # Define a classifier for RR
        self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if rr is None:
            raise ValueError("RR data (rr) must be provided for UnimodalRR!")
            
        _, rr_feats = self.text_model(rr_notes=rr)
        if self.args.use_cls_token == 'cls':
            # Use the first token (CLS token)
            rr_feats = rr_feats[:, 0, :]
        else:
            # Use mean pooling
            rr_feats = rr_feats.mean(dim=1)

        # Generate predictions using the RR classifier
        output = self.rr_classifier(rr_feats)

        return {'unimodal_rr': output, 'unified': output}

class UnimodalDN(nn.Module):
    def __init__(self, args, text_model):
        super(UnimodalDN, self).__init__()
        self.args = args

        # Text Encoder
        self.text_model = text_model
        if not self.text_model or 'DN' not in self.args.modalities:
            raise ValueError("Text encoder with 'DN' modality must be provided for UnimodalDN!")

        # Define a classifier for DN
        self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if dn is None:
            raise ValueError("DN data (dn) must be provided for UnimodalDN!")

        # Pass DN data through the text encoder
        dn_feats, _ = self.text_model(dn_notes=dn)
        
        if self.args.use_cls_token == 'cls':
            # Use the first token (CLS token)
            dn_feats = dn_feats[:, 0, :]
        else:
            # Use mean pooling
            dn_feats = dn_feats.mean(dim=1)

        # Generate predictions using the DN classifier
        output = self.dn_classifier(dn_feats)

        return {'unimodal_dn': output, 'unified': output}
        
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MeTraTransformer(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super().__init__()

        self.args = args
        self.modalities = args.modalities
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['RR', 'DN']) else None
        
        image_size = self.args.image_size
        patch_size = self.args.patch_size
        emb_dropout = self.args.emb_dropout
        output_dim = self.args.output_dim
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) + 1 
        num_patches += 76 # Account for lab value
        
        max_len = num_patches + 1
        if 'DN' in self.modalities or 'RR' in self.modalities:
            max_len = 1000

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, output_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.args = args
        # Default values with fallback to args
        
        self.feats_dim = output_dim
        # depth = getattr(args, 'depth', 4)
        depth = 4
        heads = 4
        mlp_dim = 768
        dropout = 0.1
        dim_head = 128
        # heads = getattr(args, 'heads', 4)
        # mlp_dim = getattr(args, 'mlp_dim', 768)
        # dropout = getattr(args, 'dropout', 0.1)
        # dim_head = getattr(args, 'dim_head', 128)

        self.transformer = Transformer(output_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.metra_classifier = Classifier(args.output_dim, self.args)

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        features = []

        # Get EHR features
        if 'EHR' in self.args.modalities:
            #print('EHR here')
            ehr_feats = self.ehr_model(x, seq_lengths)
            features.append(ehr_feats)

        # Get CXR features
        if 'CXR' in self.args.modalities:
            cxr_feats = self.cxr_model(img)
            features.append(cxr_feats)

        # Get text features
        if self.text_model:
            dn_feats, rr_feats = None, None
            if 'DN' in self.modalities and 'RR' in self.modalities:
                dn_feats, rr_feats = self.text_model(dn_notes=dn, rr_notes = rr)
                features.append(dn_feats)
                features.append(rr_feats)
            elif 'DN' in self.modalities and dn is not None:
                dn_feats, _ = self.text_model(dn_notes=dn)
                features.append(dn_feats)
            elif 'RR' in self.modalities and rr is not None:
                _, rr_feats = self.text_model(rr_notes = rr)
                features.append(rr_feats)

        # Combine all features
        combined_feats = torch.cat(features, dim=1)

        b, n, _ = combined_feats.shape
        
        if n + 1 > self.pos_embedding.size(1):
            # Expand positional embeddings if src is longer
            repeat_factor = (n + 1) // self.pos_embedding.size(1) + 1
            expanded_pos_embedding = self.pos_embedding.repeat(1, repeat_factor, 1)
            pos_embedding = expanded_pos_embedding[:, :n + 1, :]
        else:
            # Use the existing positional embeddings
            pos_embedding = self.pos_embedding[:, :n + 1, :]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        combined_feats = torch.cat((cls_tokens, combined_feats), dim=1)
        combined_feats += self.pos_embedding[:, :(n + 1)]
        combined_feats = self.dropout(combined_feats)

        combined_feats = self.transformer(combined_feats)

        combined_feats = combined_feats[:, 0]

        combined_feats = self.to_latent(combined_feats)
        output = self.metra_classifier(combined_feats)
        return {'metra': output, 'unified': output}
        
class ConfidencePredictor(nn.Module):
    def __init__(self, args, embedding_dim):
        super(ConfidencePredictor, self).__init__()
        self.args = args
        self.confidence_layer = nn.Linear(embedding_dim, self.args.num_classes)  # Outputs a score for each token

    def forward(self, embeddings):
        # embeddings: [batch_size, seq_length, embedding_dim]
        confidence_scores = self.confidence_layer(embeddings)  # [batch_size, seq_length, 1]
        confidence_scores = confidence_scores.squeeze(-1)  # [batch_size, seq_length]
        return confidence_scores
        
class UnimodalEHRConfidence(nn.Module):
    def __init__(self, args, ehr_model):
        super(UnimodalEHRConfidence, self).__init__()
        self.args = args

        # EHR Encoder
        self.ehr_model = ehr_model
        if not self.ehr_model:
            raise ValueError("EHR encoder must be provided for UnimodalEHR!")
            
        # Define a classifier for EHR
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args)
        self.ehr_confidence_predictor = ConfidencePredictor(self.args, ehr_model.full_feats_dim)
        
        for param in self.ehr_model.parameters():
                param.requires_grad = False
        
        if self.args.freeze:
            for param in self.ehr_model.parameters():
                param.requires_grad = False
            for param in self.ehr_classifier.parameters():
                param.requires_grad = False
            for param in self.ehr_confidence_predictor.parameters():
                param.requires_grad = False

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if x is None:
            raise ValueError("EHR data (x) must be provided for UnimodalEHR!")

        _, full_ehr_feats = self.ehr_model(x, seq_lengths)
        ehr_confidences = self.ehr_confidence_predictor(full_ehr_feats)  # [batch_size, seq_length]
        return {
            'c-unimodal_ehr': ehr_confidences
        }
        
class UnimodalCXRConfidence(nn.Module):
    def __init__(self, args, cxr_model):
        super(UnimodalCXRConfidence, self).__init__()
        self.args = args
        self.cxr_model = cxr_model
        self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
        self.cxr_confidence_predictor = ConfidencePredictor(self.args, cxr_model.full_feats_dim)

        for param in self.cxr_model.parameters():
                param.requires_grad = False
                
        if self.args.freeze:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
            for param in self.cxr_classifier.parameters():
                param.requires_grad = False
            for param in self.cxr_confidence_predictor.parameters():
                param.requires_grad = False

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if img is None:
            return None

        _, full_cxr_feats = self.cxr_model(img)
        cxr_confidences = self.cxr_confidence_predictor(full_cxr_feats)  # [batch_size, seq_length]
        return {
            'c-unimodal_cxr': cxr_confidences
        }

class UnimodalDNConfidence(nn.Module):
    def __init__(self, args, text_model):
        super(UnimodalDNConfidence, self).__init__()
        self.args = args
        self.text_model = text_model
        self.dn_confidence_predictor = ConfidencePredictor(self.args, text_model.full_feats_dim_dn)
        self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)

        for param in self.text_model.parameters():
                param.requires_grad = False
        
        if self.args.freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.dn_classifier.parameters():
                param.requires_grad = False
            for param in self.dn_confidence_predictor.parameters():
                param.requires_grad = False

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if dn is None:
            return None

        _, full_dn_feats, _, _ = self.text_model(dn_notes=dn)
        dn_confidences = self.dn_confidence_predictor(full_dn_feats)  # [batch_size, seq_length]
        return {
            'c-unimodal_dn': dn_confidences
        }

class UnimodalRRConfidence(nn.Module):
    def __init__(self, args, text_model):
        super(UnimodalRRConfidence, self).__init__()
        self.args = args
        self.text_model = text_model
        self.rr_confidence_predictor = ConfidencePredictor(self.args, text_model.full_feats_dim_rr)
        self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)

        for param in self.text_model.parameters():
                param.requires_grad = False
                
        if self.args.freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.rr_classifier.parameters():
                param.requires_grad = False
            for param in self.rr_confidence_predictor.parameters():
                param.requires_grad = False

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if rr is None:
            return None

        _, _, _, full_rr_feats = self.text_model(rr_notes=rr)
        rr_confidences = self.rr_confidence_predictor(full_rr_feats)  # [batch_size, seq_length]
        return {
            'c-unimodal_rr': rr_confidences
        }
        
class TempCUnimodalEHR(nn.Module):
    """
    Same as c-unimodal EHR, but with an additional temperature parameter.
    Everything except that parameter is frozen.
    """
    def __init__(self, args, ehr_model):
        super().__init__()
        self.args = args
        self.ehr_model = ehr_model
        if not self.ehr_model:
            raise ValueError("EHR encoder must be provided for TempCUnimodalEHR!")

        # Same sub-modules as UnimodalEHRConfidence:
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args)
        self.ehr_confidence_predictor = ConfidencePredictor(self.args, ehr_model.full_feats_dim)
        if args.task == 'in-hospital-mortality':
            self.max_seq_len = 48
        else:
            self.max_seq_len = 2646
        self.num_classes = args.num_classes

        # Freeze all existing parameters
        for param in self.ehr_model.parameters():
            param.requires_grad = False
        for param in self.ehr_classifier.parameters():
            param.requires_grad = False
        for param in self.ehr_confidence_predictor.parameters():
            param.requires_grad = False

        # Add the temperature parameter (the only one we keep trainable)
        if args.task == 'in-hospital-mortality':
            self.ehr_temperature = nn.Parameter(torch.ones(self.max_seq_len))
        else:
            self.ehr_temperature = nn.Parameter(torch.ones(self.max_seq_len, self.num_classes))

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        # Exactly like c-unimodal EHR, but we scale the logits by 1/temperature
        if x is None:
            raise ValueError("EHR data (x) must be provided for TempCUnimodalEHR!")
        # Get the full token embeddings
        _, full_ehr_feats = self.ehr_model(x, seq_lengths)

        # Unscaled confidence logits
        ehr_confidences = self.ehr_confidence_predictor(full_ehr_feats)
        # Scale them by dividing by temperature
        seq_len = ehr_confidences.shape[1]
        temp_for_seq = self.ehr_temperature[:seq_len]
        temp_for_seq = temp_for_seq.clamp_min(1e-9)
        
        scaled_ehr_confidences = ehr_confidences / temp_for_seq.unsqueeze(0)
        
        return {
            'temp_c-unimodal_ehr': scaled_ehr_confidences
        }


class TempCUnimodalCXR(nn.Module):
    """
    Temperature-based version of c-unimodal CXR.
    """
    def __init__(self, args, cxr_model):
        super().__init__()
        self.args = args
        self.cxr_model = cxr_model
        if not self.cxr_model:
            raise ValueError("CXR encoder must be provided for TempCUnimodalCXR!")

        self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
        self.cxr_confidence_predictor = ConfidencePredictor(self.args, self.cxr_model.full_feats_dim)
        self.max_seq_len = 578
        self.num_classes = args.num_classes
        
        # Freeze everything
        for param in self.cxr_model.parameters():
            param.requires_grad = False
        for param in self.cxr_classifier.parameters():
            param.requires_grad = False
        for param in self.cxr_confidence_predictor.parameters():
            param.requires_grad = False

        # Only the temperature is learnable
        if args.task == 'in-hospital-mortality':
            self.cxr_temperature = nn.Parameter(torch.ones(self.max_seq_len))
        else:
            self.cxr_temperature = nn.Parameter(torch.ones(self.max_seq_len, self.num_classes))

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        if img is None:
            raise ValueError("CXR data (img) must be provided for TempCUnimodalCXR!")

        # Full token embeddings from the model
        _, full_cxr_feats = self.cxr_model(img)

        # Raw confidence logits
        cxr_confidences = self.cxr_confidence_predictor(full_cxr_feats)
        # Scale by temperature
        seq_len = cxr_confidences.shape[1]
        temp_for_seq = self.cxr_temperature[:seq_len]
        temp_for_seq = temp_for_seq.clamp_min(1e-9)
        
        scaled_cxr_confidences = cxr_confidences / temp_for_seq.unsqueeze(0)
        

        return {
            'temp_c-unimodal_cxr': scaled_cxr_confidences
        }


class TempCUnimodalDN(nn.Module):
    """
    Temperature-based version of c-unimodal DN.
    """
    def __init__(self, args, text_model):
        super().__init__()
        self.args = args
        self.text_model = text_model
        if not self.text_model:
            raise ValueError("Text encoder must be provided for TempCUnimodalDN!")

        self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)
        self.dn_confidence_predictor = ConfidencePredictor(self.args, self.text_model.full_feats_dim_dn)
        self.max_seq_len = 512
        self.num_classes = args.num_classes
        
        # Freeze everything
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.dn_classifier.parameters():
            param.requires_grad = False
        for param in self.dn_confidence_predictor.parameters():
            param.requires_grad = False

        # Temperature
        if args.task == 'in-hospital-mortality':
            self.dn_temperature = nn.Parameter(torch.ones(self.max_seq_len))
        else:
            self.dn_temperature = nn.Parameter(torch.ones(self.max_seq_len, self.num_classes))

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        if dn is None:
            raise ValueError("DN data (dn) must be provided for TempCUnimodalDN!")

        _, full_dn_feats, _, _ = self.text_model(dn_notes=dn)
        dn_confidences = self.dn_confidence_predictor(full_dn_feats)
        
        seq_len = dn_confidences.shape[1]
        temp_for_seq = self.dn_temperature[:seq_len]
        temp_for_seq = temp_for_seq.clamp_min(1e-9)
        
        scaled_dn_confidences = dn_confidences / temp_for_seq.unsqueeze(0)

        return {
            'temp_c-unimodal_dn': scaled_dn_confidences
        }


class TempCUnimodalRR(nn.Module):
    """
    Temperature-based version of c-unimodal RR that learns a separate temperature
    per token AND per class, assuming multi-class confidences of shape
    [B, seq_len, num_classes].
    """
    def __init__(self, args, text_model):
        super().__init__()
        self.args = args
        self.text_model = text_model
        if not self.text_model:
            raise ValueError("Text encoder must be provided for TempCUnimodalRR!")

        self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
        self.rr_confidence_predictor = ConfidencePredictor(self.args, self.text_model.full_feats_dim_rr)
        self.max_seq_len = 512
        self.num_classes = args.num_classes

        # Freeze everything
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.rr_classifier.parameters():
            param.requires_grad = False
        for param in self.rr_confidence_predictor.parameters():
            param.requires_grad = False

        # Temperature per (token, class) pair
        # Shape: [max_seq_len, num_classes]
        if args.task == 'in-hospital-mortality':
            self.rr_temperature = nn.Parameter(torch.ones(self.max_seq_len))
        else:
            self.rr_temperature = nn.Parameter(torch.ones(self.max_seq_len, self.num_classes))

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        if rr is None:
            raise ValueError("RR data (rr) must be provided for TempCUnimodalRR!")

        # full_rr_feats is typically [B, seq_len, feats_dim], e.g. after text_model
        # rr_confidences is [B, seq_len, num_classes]
        _, _, _, full_rr_feats = self.text_model(rr_notes=rr)
        rr_confidences = self.rr_confidence_predictor(full_rr_feats)

        # You must confirm rr_confidences indeed has shape [B, seq_len, num_classes].
        # Adjust your predictor if it isn’t already returning multi-class outputs.
        seq_len = rr_confidences.shape[1]

        # Slice out the relevant portion for the current sequence length
        # shape = [seq_len, num_classes]
        temp_for_seq = self.rr_temperature[:seq_len]
        temp_for_seq = temp_for_seq.clamp_min(1e-9)

        # Expand to [B, seq_len, num_classes] so it can divide rr_confidences
        scaled_rr_confidences = rr_confidences / temp_for_seq.unsqueeze(0)

        return {
            'temp_c-unimodal_rr': scaled_rr_confidences
        }

class CMSMAFusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(CMSMAFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities.split("-")  # Modalities we are considering
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        self.ehr_model_fixed = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model_fixed = cxr_model if 'CXR' in self.modalities else None
        self.text_model_fixed = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        # One output dimension assumed for simplicity. If multiclass, adjust accordingly.
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args) if 'EHR' in self.modalities else None
        self.ehr_confidence_predictor = ConfidencePredictor(self.args, self.ehr_model.full_feats_dim)
        self.ehr_temperature = nn.Parameter(torch.ones(48)) if args.task == 'in-hospital-mortality' else nn.Parameter(torch.ones(2646, args.num_classes))

        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
            self.cxr_confidence_predictor = ConfidencePredictor(self.args, self.cxr_model.full_feats_dim)
            d_in = self.cxr_model.cxr_encoder.projection_layer.in_features
            d_out = self.cxr_model.cxr_encoder.projection_layer.out_features
            
            self.cxr_high_proj = nn.Linear(d_in, d_out)
            self.cxr_low_proj  = nn.Linear(d_in, d_out)
            self.cxr_temperature = nn.Parameter(torch.ones(578)) if args.task == 'in-hospital-mortality' else nn.Parameter(torch.ones(578, args.num_classes))

                
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
            self.rr_confidence_predictor = ConfidencePredictor(self.args, self.text_model.full_feats_dim_rr)
            
            d_in = self.text_model.bert.config.hidden_size
            d_out = self.text_model.feats_dim_rr
            
            self.rr_high_proj = nn.Linear(d_in, d_out)
            self.rr_low_proj  = nn.Linear(d_in, d_out)
            self.rr_temperature = nn.Parameter(torch.ones(512)) if args.task == 'in-hospital-mortality' else nn.Parameter(torch.ones(512, args.num_classes))

            
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)
            self.dn_confidence_predictor = ConfidencePredictor(self.args, self.text_model.full_feats_dim_dn)
            
            d_in = self.text_model.bert.config.hidden_size
            d_out = self.text_model.feats_dim_dn
            
            self.dn_high_proj = nn.Linear(d_in, d_out)
            self.dn_low_proj  = nn.Linear(d_in, d_out)
            self.dn_temperature = nn.Parameter(torch.ones(512)) if args.task == 'in-hospital-mortality' else nn.Parameter(torch.ones(512, args.num_classes))

        # Missingness classifier: takes missingness vector [batch_size, num_modalities]
        self.missingness_classifier = Classifier(len(self.modalities), self.args)
        emb_dim = args.patch_output_dim   
        num_mods = len(self.modalities)  # number of modalities present

        if self.args.fuser is None:
            # In "none" mode, we simply concatenate the pooled representations.
            self.high_fuser_classifier = Classifier(input_dim=num_mods * self.args.num_classes * emb_dim, args=self.args)
            self.low_fuser_classifier  = Classifier(input_dim=num_mods * self.args.num_classes * emb_dim, args=self.args)
        elif self.args.fuser == "lstm":
            # In "lstm" mode, we process the sequence of modality embeddings.
            self.high_fuser_lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
            self.low_fuser_lstm  = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
            self.high_fuser_classifier = Classifier(input_dim=emb_dim, args=self.args)
            self.low_fuser_classifier  = Classifier(input_dim=emb_dim, args=self.args)
        elif self.args.fuser == "transformer":
            # In "transformer" mode, we use a transformer block.
            self.high_fuser_transformer = Transformer(dim=emb_dim, depth=1, heads=8, 
                                                        dim_head=emb_dim // 8, mlp_dim=emb_dim * 2, 
                                                        dropout=0.)
            self.low_fuser_transformer  = Transformer(dim=emb_dim, depth=1, heads=8, 
                                                        dim_head=emb_dim // 8, mlp_dim=emb_dim * 2, 
                                                        dropout=0.)
            self.high_fuser_classifier = Classifier(input_dim=emb_dim, args=self.args)
            self.low_fuser_classifier  = Classifier(input_dim=emb_dim, args=self.args)
        else:
            raise ValueError(f"Unsupported fuser type: {self.args.fuser}")
        # ---------------------------------------------------------

        self.num_modalities = num_mods

        if self.args.freeze:
            print('freeze activated')
            self.freeze_all()

        self.num_modalities = len(self.modalities)
        # Now we have unimodal preds, plus joint pred, plus missingness pred = num_modalities + 2
        if self.args.ablation == "without_joint_module":
            num_extra = 1  # only missingness prediction is added
        elif self.args.ablation == "without_missingness_module":
            num_extra = 2  # only low_conf and high_conf predictions are added
        elif self.args.ablation == "without_late_module":
            self.num_modalities = 0
            num_extra = 3
        else:
            num_extra = 3  # default: missingness, low_conf, and high_conf predictions all present
        
        # Initialize the weights parameter with dynamic dimension:
        self.weights = nn.Parameter(torch.ones(self.num_modalities + num_extra, self.args.num_classes))
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        # Freeze and count parameters
        if self.ehr_model_fixed:
            for p in self.ehr_model_fixed.parameters():
                p.requires_grad = False
            print(f"Frozen ehr_model_fixed: {count_parameters(self.ehr_model_fixed)} parameters")
        
        if self.cxr_model_fixed:
            for p in self.cxr_model_fixed.parameters():
                p.requires_grad = False
            print(f"Frozen cxr_model_fixed: {count_parameters(self.cxr_model_fixed)} parameters")
        
        if self.text_model_fixed:
            for p in self.text_model_fixed.parameters():
                p.requires_grad = False
            print(f"Frozen text_model_fixed: {count_parameters(self.text_model_fixed)} parameters")
        
        # Freeze confidence predictors
        if hasattr(self, 'ehr_confidence_predictor'):
            for p in self.ehr_confidence_predictor.parameters():
                p.requires_grad = False
            print(f"Frozen ehr_confidence_predictor: {count_parameters(self.ehr_confidence_predictor)} parameters")
        
        if hasattr(self, 'cxr_confidence_predictor'):
            for p in self.cxr_confidence_predictor.parameters():
                p.requires_grad = False
            print(f"Frozen cxr_confidence_predictor: {count_parameters(self.cxr_confidence_predictor)} parameters")
        
        if hasattr(self, 'rr_confidence_predictor'):
            for p in self.rr_confidence_predictor.parameters():
                p.requires_grad = False
            print(f"Frozen rr_confidence_predictor: {count_parameters(self.rr_confidence_predictor)} parameters")
        
        if hasattr(self, 'dn_confidence_predictor'):
            for p in self.dn_confidence_predictor.parameters():
                p.requires_grad = False
            print(f"Frozen dn_confidence_predictor: {count_parameters(self.dn_confidence_predictor)} parameters")
        
        # Freeze temperatures (usually scalar tensors, not modules)
        def count_tensor_params(tensor):
            return tensor.numel() if tensor is not None else 0
        
        if hasattr(self, 'ehr_temperature'):
            self.ehr_temperature.requires_grad = False
            print(f"Frozen ehr_temperature: {count_tensor_params(self.ehr_temperature)} parameters")
        
        if hasattr(self, 'cxr_temperature'):
            self.cxr_temperature.requires_grad = False
            print(f"Frozen cxr_temperature: {count_tensor_params(self.cxr_temperature)} parameters")
        
        if hasattr(self, 'rr_temperature'):
            self.rr_temperature.requires_grad = False
            print(f"Frozen rr_temperature: {count_tensor_params(self.rr_temperature)} parameters")
        
        if hasattr(self, 'dn_temperature'):
            self.dn_temperature.requires_grad = False
            print(f"Frozen dn_temperature: {count_tensor_params(self.dn_temperature)} parameters")
                
        if self.args.freeze:
            print('freeze activated')
            self.freeze_all()

    def freeze_all(self):
        for param in self.ehr_model.parameters():
            param.requires_grad = False
        if self.ehr_classifier:
            for param in self.ehr_classifier.parameters():
                param.requires_grad = False
        if 'CXR' in self.modalities:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
            for param in self.cxr_classifier.parameters():
                param.requires_grad = False
        if 'RR' in self.modalities:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.rr_classifier.parameters():
                param.requires_grad = False
        if 'DN' in self.modalities:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.dn_classifier.parameters():
                param.requires_grad = False
                
    def detect_missingness_batch(self, batch_size, cxr=None, dn=None, rr=None):
        missingness_matrix = torch.ones(batch_size, len(self.modalities), device='cpu')
        if 'CXR' in self.modalities and cxr is not None:
            cxr_missing_mask = torch.zeros(3, 384, 384, device=cxr.device)
            cxr_is_missing = (cxr == cxr_missing_mask).view(cxr.size(0), -1).all(dim=1)
            missingness_matrix[:, self.modalities.index('CXR')] = ~cxr_is_missing
        if 'DN' in self.modalities and dn is not None:
            dn_is_missing = torch.tensor([len(note.strip()) == 0 for note in dn], device='cpu')
            missingness_matrix[:, self.modalities.index('DN')] = ~dn_is_missing
        if 'RR' in self.modalities and rr is not None:
            rr_is_missing = torch.tensor([len(note.strip()) == 0 for note in rr], device='cpu')
            missingness_matrix[:, self.modalities.index('RR')] = ~rr_is_missing
        return missingness_matrix
        
    def equalize(self):
        """Ensures projection layers have the same weights and biases and aligns fixed models."""
        with torch.no_grad():
            if 'EHR' in self.modalities:
                if self.ehr_model_fixed:
                    self.ehr_model_fixed.load_state_dict(self.ehr_model.state_dict())
            if 'CXR' in self.modalities:
                self.cxr_high_proj.weight.copy_(self.cxr_model.cxr_encoder.projection_layer.weight)
                self.cxr_high_proj.bias.copy_(self.cxr_model.cxr_encoder.projection_layer.bias)
                self.cxr_low_proj.weight.copy_(self.cxr_model.cxr_encoder.projection_layer.weight)
                self.cxr_low_proj.bias.copy_(self.cxr_model.cxr_encoder.projection_layer.bias)
                
                if self.cxr_model_fixed:
                    self.cxr_model_fixed.load_state_dict(self.cxr_model.state_dict())

            if 'RR' in self.modalities:
                self.rr_high_proj.weight.copy_(self.text_model.fc_rr.weight)
                self.rr_high_proj.bias.copy_(self.text_model.fc_rr.bias)
                self.rr_low_proj.weight.copy_(self.text_model.fc_rr.weight)
                self.rr_low_proj.bias.copy_(self.text_model.fc_rr.bias)
                
                if self.text_model_fixed:
                    self.text_model_fixed.load_state_dict(self.text_model.state_dict())
                
            if 'DN' in self.modalities:
                self.dn_high_proj.weight.copy_(self.text_model.fc_dn.weight)
                self.dn_high_proj.bias.copy_(self.text_model.fc_dn.bias)
                self.dn_low_proj.weight.copy_(self.text_model.fc_dn.weight)
                self.dn_low_proj.bias.copy_(self.text_model.fc_dn.bias)
                
                if self.text_model_fixed:
                    self.text_model_fixed.load_state_dict(self.text_model.state_dict())
                    
            if self.ehr_model:
                for param in self.ehr_model_fixed.parameters():
                    param.requires_grad = False
            if self.cxr_model:
                for param in self.cxr_model_fixed.parameters():
                    param.requires_grad = False
            if self.text_model:
                for param in self.text_model_fixed.parameters():
                    param.requires_grad = False
    
    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        #print("[Forward] Starting forward pass")
        batch_size = None
        features = []
        preds = []
        high_pool, high_conf_scores = [], []  # lists of tensors per modality
        low_pool,  low_conf_scores  = [], []
        modality_names = [] 
        

        # Process each modality
        if 'EHR' in self.modalities:
            ehr_feats, full_ehr_feats = self.ehr_model(x, seq_lengths)
            _, full_ehr_conf_feats = self.ehr_model_fixed(x, seq_lengths)
            # print("[EHR] ehr_feats shape:", ehr_feats.shape)
            # print("[EHR] full_ehr_feats shape:", full_ehr_feats.shape)
            
            if len(ehr_feats.shape) > 2:  # Only apply pooling if there are more than 2 dimensions
                ehr_feats = ehr_feats.mean(dim=1)
            features.append(ehr_feats)
            batch_size = ehr_feats.size(0)
            missingness_matrix = self.detect_missingness_batch(batch_size, cxr=img, dn=dn, rr=rr)
            
            # print("[EHR] batch_size determined to be:", batch_size)
            #print("EHR feature shape:", ehr_feats.shape)
            ehr_pred = self.ehr_classifier(ehr_feats)
            # print("[EHR] ehr_pred shape:", ehr_pred.shape)
            preds.append(ehr_pred)
            
            ehr_conf_logits = self.ehr_confidence_predictor(full_ehr_conf_feats)
            seq_len = ehr_conf_logits.shape[1]
            temp_ehr = self.ehr_temperature[:seq_len].clamp_min(1e-9)
            scaled_ehr_conf_logits = ehr_conf_logits / temp_ehr.unsqueeze(0)

            # 3) take the maximum across classes -> [B, L]
            if self.args.num_classes > 1:
                ehr_conf_probs = torch.sigmoid(ehr_conf_logits)
                # ehr_confidences = torch.max(ehr_conf_probs, 1 - ehr_conf_probs).max(dim=-1).values
                ehr_confidences = torch.max(ehr_conf_probs, 1 - ehr_conf_probs)
            else:
                ehr_conf_probs = torch.sigmoid(scaled_ehr_conf_logits)
                ehr_confidences = torch.where(ehr_conf_probs > (1 - ehr_conf_probs), ehr_conf_probs, 1 - ehr_conf_probs)
            # print("[EHR] ehr_confidences shape:", ehr_confidences.shape)
            
            mask_ehr_high = (ehr_confidences >= self.args.ehr_confidence_threshold)
            mask_ehr_low  = ~mask_ehr_high
            
            if self.args.num_classes > 1:
                full_ehr_feats_exp = full_ehr_feats.unsqueeze(2)  
            
                # Expand masks to [B, L, num_classes, 1]
                mask_ehr_high_f = mask_ehr_high.float().unsqueeze(-1)
                mask_ehr_low_f  = mask_ehr_low.float().unsqueeze(-1)
                
                # Compute per-class weighted sums and averages
                high_sum = (full_ehr_feats_exp * mask_ehr_high_f).sum(dim=1)  # [B, num_classes, feat_dim]
                low_sum  = (full_ehr_feats_exp * mask_ehr_low_f).sum(dim=1)
            else:
                # Expand masks to [B, L_ehr, 1]
                mask_ehr_high_f = mask_ehr_high.unsqueeze(-1).float()
                mask_ehr_low_f  = mask_ehr_low.unsqueeze(-1).float()
                
                
        
                high_sum = (full_ehr_feats * mask_ehr_high_f).sum(dim=1)
                low_sum  = (full_ehr_feats * mask_ehr_low_f).sum(dim=1)
            count_high = mask_ehr_high_f.sum(dim=1).clamp_min(1e-9)
            count_low  = mask_ehr_low_f.sum(dim=1).clamp_min(1e-9)
            pooled_high = high_sum / count_high  # [B, emb_dim]
            pooled_low  = low_sum  / count_low   # [B, emb_dim]
            

            # Also compute average token confidence for the high tokens (for ordering later)
            conf_high = (ehr_confidences * mask_ehr_high.float()).sum(dim=1) / (mask_ehr_high.float().sum(dim=1).clamp_min(1e-9))
            high_pool.append(pooled_high)
            high_conf_scores.append(conf_high)
            low_pool.append(pooled_low)
            conf_low = (ehr_confidences * mask_ehr_low.float()).sum(dim=1) / (mask_ehr_low.float().sum(dim=1).clamp_min(1e-9))
            low_conf_scores.append(conf_low)
            # print("EHR mask_high sum:", mask_ehr_high.sum(), "mask_low sum:", mask_ehr_low.sum())
            
        if 'CXR' in self.modalities:
            # Get features and predictions
            cxr_feats, full_cxr_feats = self.cxr_model(img)
            _, full_cxr_conf_feats = self.cxr_model_fixed(img)
            cxr_feats = cxr_feats[:, 0, :]  # [B, feat_dim]
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            preds.append(cxr_pred)
            
            # Confidence computation
            cxr_conf_logits = self.cxr_confidence_predictor(full_cxr_conf_feats)
            seq_len = cxr_conf_logits.shape[1]
            temp_cxr = self.cxr_temperature[:seq_len].clamp_min(1e-9)
            scaled_cxr_conf_logits = cxr_conf_logits / temp_cxr.unsqueeze(0)
            cxr_conf_probs = torch.sigmoid(scaled_cxr_conf_logits)  # [B, L, num_classes] if num_classes > 1
            
            if self.args.num_classes > 1:
                cxr_confidences = torch.max(cxr_conf_probs, 1 - cxr_conf_probs)  # [B, L, num_classes]
            else:
                cxr_confidences = torch.where(cxr_conf_probs > (1 - cxr_conf_probs), cxr_conf_probs, 1 - cxr_conf_probs)  # [B, L, 1]
            
            # Get the projection outputs (note: projection is always applied)
            B, L_cxr, D_in = full_cxr_feats.shape
            flattened_cxr = full_cxr_feats.reshape(B * L_cxr, D_in)
            cxr_high_proj_out = self.cxr_high_proj(flattened_cxr).view(B, L_cxr, -1)
            cxr_low_proj_out  = self.cxr_low_proj(flattened_cxr).view(B, L_cxr, -1)
            
            # For num_classes > 1, we expand along a new dimension; else we leave as-is.
            if self.args.num_classes > 1:
                feats_exp_high = cxr_high_proj_out.unsqueeze(2)  # [B, L, 1, D_out]
                feats_exp_low  = cxr_low_proj_out.unsqueeze(2)
            else:
                feats_exp_high = cxr_high_proj_out  # [B, L, D_out]
                feats_exp_low  = cxr_low_proj_out
        
            # Create per-token masks (they already have the proper shape if num_classes==1)
            mask_high = (cxr_confidences >= self.args.cxr_confidence_threshold)  # [B, L, (num_classes)]
            mask_low  = ~mask_high
        
            # Expand the mask only if needed
            if self.args.num_classes > 1:
                mask_high_f = mask_high.float().unsqueeze(-1)  # [B, L, num_classes, 1]
                mask_low_f  = mask_low.float().unsqueeze(-1)
            else:
                mask_high_f = mask_high.unsqueeze(-1).float()   # [B, L, 1]
                mask_low_f  = mask_low.unsqueeze(-1).float()
            
            # Pool features over the token dimension (dim=1)
            high_sum = (feats_exp_high * mask_high_f).sum(dim=1)
            low_sum  = (feats_exp_low  * mask_low_f).sum(dim=1)
            count_high = mask_high_f.sum(dim=1).clamp_min(1e-9)
            count_low  = mask_low_f.sum(dim=1).clamp_min(1e-9)
            pooled_high = high_sum / count_high  # [B, (num_classes), D_out]
            pooled_low  = low_sum  / count_low
        
            # Compute average confidence per token (for ordering if needed)
            conf_high = (cxr_confidences * mask_high.float()).sum(dim=1) / (
                            mask_high.float().sum(dim=1).clamp_min(1e-9))
            conf_low  = (cxr_confidences * mask_low.float()).sum(dim=1) / (
                            mask_low.float().sum(dim=1).clamp_min(1e-9))
            
            high_pool.append(pooled_high)
            high_conf_scores.append(conf_high)
            low_pool.append(pooled_low)
            low_conf_scores.append(conf_low)
            # print("CXR mask_high sum:", mask_high.sum(), "mask_low sum:", mask_low.sum())
        

        if 'DN' in self.modalities and dn is not None:
            dn_feats, full_dn_feats, _, _ = self.text_model(dn_notes=dn)
            _, full_dn_conf_feats, _, _ = self.text_model_fixed(dn_notes=dn)
            dn_feats = dn_feats.mean(dim=1)
            features.append(dn_feats)
            dn_pred = self.dn_classifier(dn_feats)
            preds.append(dn_pred)
            
            dn_conf_logits = self.dn_confidence_predictor(full_dn_conf_feats)
            seq_len = dn_conf_logits.shape[1]
            temp_dn = self.dn_temperature[:seq_len].clamp_min(1e-9)
            scaled_dn_conf_logits = dn_conf_logits / temp_dn.unsqueeze(0)
            dn_conf_probs = torch.sigmoid(scaled_dn_conf_logits)
            
            if self.args.num_classes > 1:
                dn_confidences = torch.max(dn_conf_probs, 1 - dn_conf_probs)
            else:
                dn_confidences = torch.where(dn_conf_probs > (1 - dn_conf_probs),
                                             dn_conf_probs, 1 - dn_conf_probs)
            
            # Apply the DN projection layers to map to the common embedding dimension:
            B, L_dn, D_in = full_dn_feats.shape
            flattened_dn = full_dn_feats.reshape(B * L_dn, D_in)
            dn_high_proj_out = self.dn_high_proj(flattened_dn).view(B, L_dn, -1)
            dn_low_proj_out  = self.dn_low_proj(flattened_dn).view(B, L_dn, -1)
            
            # Expand if multi-class
            if self.args.num_classes > 1:
                feats_exp = dn_high_proj_out.unsqueeze(2)  # For high pooling, shape: [B, L, 1, feat_dim]
                feats_exp_low = dn_low_proj_out.unsqueeze(2)
            else:
                feats_exp = dn_high_proj_out
                feats_exp_low = dn_low_proj_out
        
            mask_high = (dn_confidences >= self.args.dn_confidence_threshold)
            mask_low  = ~mask_high
            
            if self.args.num_classes > 1:
                mask_high_f = mask_high.float().unsqueeze(-1)  # [B, L, num_classes, 1]
                mask_low_f  = mask_low.float().unsqueeze(-1)
            else:
                mask_high_f = mask_high.unsqueeze(-1).float()
                mask_low_f  = mask_low.unsqueeze(-1).float()
            
            high_sum = (feats_exp * mask_high_f).sum(dim=1)
            low_sum  = (feats_exp_low * mask_low_f).sum(dim=1)
            count_high = mask_high_f.sum(dim=1).clamp_min(1e-9)
            count_low  = mask_low_f.sum(dim=1).clamp_min(1e-9)
            pooled_high = high_sum / count_high
            pooled_low  = low_sum  / count_low
            
            conf_high = (dn_confidences * mask_high.float()).sum(dim=1) / (mask_high.float().sum(dim=1).clamp_min(1e-9))
            conf_low = (dn_confidences * mask_low.float()).sum(dim=1) / (mask_low.float().sum(dim=1).clamp_min(1e-9))
            
            high_pool.append(pooled_high)
            high_conf_scores.append(conf_high)
            low_pool.append(pooled_low)
            low_conf_scores.append(conf_low)
            #print("DN mask_high sum:", mask_high.sum(), "mask_low sum:", mask_low.sum())



        if 'RR' in self.modalities and rr is not None:
            _, _, rr_feats, full_rr_feats = self.text_model(rr_notes=rr)
            _, _, _, full_rr_conf_feats = self.text_model_fixed(rr_notes=rr)
            rr_feats = rr_feats.mean(dim=1)
            features.append(rr_feats)
            rr_pred = self.rr_classifier(rr_feats)
            preds.append(rr_pred)
            
            rr_conf_logits = self.rr_confidence_predictor(full_rr_conf_feats)
            seq_len = rr_conf_logits.shape[1]
            temp_rr = self.rr_temperature[:seq_len].clamp_min(1e-9)
            scaled_rr_conf_logits = rr_conf_logits / temp_rr.unsqueeze(0)
            rr_conf_probs = torch.sigmoid(scaled_rr_conf_logits)
            
            if self.args.num_classes > 1:
                rr_confidences = torch.max(rr_conf_probs, 1 - rr_conf_probs)
            else:
                rr_confidences = torch.where(rr_conf_probs > (1 - rr_conf_probs),
                                             rr_conf_probs, 1 - rr_conf_probs)
            
            # Apply the RR projection layers to map features to the common dimension:
            B, L_rr, D_in = full_rr_feats.shape
            flattened_rr = full_rr_feats.reshape(B * L_rr, D_in)
            rr_high_proj_out = self.rr_high_proj(flattened_rr).view(B, L_rr, -1)
            rr_low_proj_out  = self.rr_low_proj(flattened_rr).view(B, L_rr, -1)
            
            if self.args.num_classes > 1:
                feats_exp = rr_high_proj_out.unsqueeze(2)  # [B, L, 1, feat_dim]
                feats_exp_low = rr_low_proj_out.unsqueeze(2)
            else:
                feats_exp = rr_high_proj_out
                feats_exp_low = rr_low_proj_out
        
            mask_high = (rr_confidences >= self.args.rr_confidence_threshold)
            mask_low  = ~mask_high
            
            if self.args.num_classes > 1:
                mask_high_f = mask_high.float().unsqueeze(-1)  # [B, L, num_classes, 1]
                mask_low_f  = mask_low.float().unsqueeze(-1)
            else:
                mask_high_f = mask_high.unsqueeze(-1).float()
                mask_low_f  = mask_low.unsqueeze(-1).float()
            
            high_sum = (feats_exp * mask_high_f).sum(dim=1)
            low_sum  = (feats_exp_low * mask_low_f).sum(dim=1)
            count_high = mask_high_f.sum(dim=1).clamp_min(1e-9)
            count_low  = mask_low_f.sum(dim=1).clamp_min(1e-9)
            pooled_high = high_sum / count_high
            pooled_low  = low_sum  / count_low
            
            conf_high = (rr_confidences * mask_high.float()).sum(dim=1) / (mask_high.float().sum(dim=1).clamp_min(1e-9))
            conf_low = (rr_confidences * mask_low.float()).sum(dim=1) / (mask_low.float().sum(dim=1).clamp_min(1e-9))
            
            high_pool.append(pooled_high)
            high_conf_scores.append(conf_high)
            low_pool.append(pooled_low)
            low_conf_scores.append(conf_low)
            #print("RR mask_high sum:", mask_high.sum(), "mask_low sum:", mask_low.sum())


    
        high_tensor = torch.stack(high_pool, dim=1)  # high modality representations
        low_tensor  = torch.stack(low_pool, dim=1)
        high_conf_tensor = torch.stack(high_conf_scores, dim=1)  # [B, num_mods]
        low_conf_tensor  = torch.stack(low_conf_scores, dim=1)
        
        if self.args.num_classes > 1:
            B = high_tensor.size(0)
            emb_dim = high_tensor.size(-1)
            high_tensor = high_tensor.view(B, -1, emb_dim)  # shape: [B, num_modalities*num_classes, emb_dim]
            low_tensor = low_tensor.view(B, -1, emb_dim)
            high_conf_tensor = high_conf_tensor.view(B, -1)    # shape: [B, num_modalities*num_classes]
            low_conf_tensor = low_conf_tensor.view(B, -1)

        # Depending on fuser type, fuse the modality representations:
        if self.args.ablation != "without_joint_module":
            if self.args.fuser is None:
                # Simply concatenate along the feature dimension.
                high_joint = high_tensor.view(batch_size, -1)
                low_joint  = low_tensor.view(batch_size, -1)
                high_conf_pred = self.high_fuser_classifier(high_joint)
                low_conf_pred  = self.low_fuser_classifier(low_joint)
            elif self.args.fuser == "lstm":
                # For LSTM, we first sort modalities by confidence so that the highest confidence modality is last.
                # (Sorting is done per sample.)
                # For high tokens:
                sort_idx = torch.argsort(high_conf_tensor, dim=1, descending=False)
                # Gather sorted high modality embeddings.
                sorted_high = torch.gather(high_tensor, 1, sort_idx.unsqueeze(-1).expand(-1, -1, high_tensor.size(-1)))
                lstm_out, _ = self.high_fuser_lstm(sorted_high)  # lstm_out shape: [B, num_mods, emb_dim]
                high_rep = lstm_out[:, -1, :]  # take the last time-step
                high_conf_pred = self.high_fuser_classifier(high_rep)
                # For low tokens:
                sort_idx = torch.argsort(low_conf_tensor, dim=1, descending=False)
                sorted_low = torch.gather(low_tensor, 1, sort_idx.unsqueeze(-1).expand(-1, -1, low_tensor.size(-1)))
                lstm_out, _ = self.low_fuser_lstm(sorted_low)
                low_rep = lstm_out[:, -1, :]
                low_conf_pred = self.low_fuser_classifier(low_rep)
            elif self.args.fuser == "transformer":
                # For transformer fusion, we again sort modalities (if desired) and feed the sequence into the transformer.
                sort_idx = torch.argsort(high_conf_tensor, dim=1, descending=False)
                sorted_high = torch.gather(high_tensor, 1, sort_idx.unsqueeze(-1).expand(-1, -1, high_tensor.size(-1)))
                trans_out = self.high_fuser_transformer(sorted_high)  # [B, num_mods, emb_dim]
                high_rep = trans_out.mean(dim=1)
                high_conf_pred = self.high_fuser_classifier(high_rep)
    
                sort_idx = torch.argsort(low_conf_tensor, dim=1, descending=False)
                sorted_low = torch.gather(low_tensor, 1, sort_idx.unsqueeze(-1).expand(-1, -1, low_tensor.size(-1)))
                trans_out = self.low_fuser_transformer(sorted_low)
                low_rep = trans_out.mean(dim=1)
                low_conf_pred = self.low_fuser_classifier(low_rep)
            else:
                raise ValueError(f"Unsupported fuser type: {self.args.fuser}")
        
        missingness_matrix_device = missingness_matrix.to(preds[0].device)
        missingness_preds = self.missingness_classifier(missingness_matrix_device.float())
        
        # First, compute the base modality predictions.
        modality_preds = torch.cat(preds, dim=1)  # Shape: [batch_size, num_modalities]
        
        # Decide what additional predictions to include based on self.args.ablation.
        # Also, record the number of extra columns (ones) to be appended in the missingness matrix.
        if self.args.ablation == "without_joint_module":
            # Without joint module: exclude high and low confidence predictions.
            # Only include the missingness prediction.
            preds = torch.cat([modality_preds, missingness_preds], dim=1)
            extra_cols = 1  # one column corresponds to missingness_preds only
            high_conf_pred = missingness_preds
            low_conf_pred = missingness_preds
        elif self.args.ablation == "without_missingness_module":
            # Without missingness module: exclude missingness prediction.
            # Only include low and high confidence predictions.
            preds = torch.cat([modality_preds, low_conf_pred, high_conf_pred], dim=1)
            extra_cols = 2  # two columns: one for low_conf and one for high_conf
        elif self.args.ablation == "without_late_module":
            preds = torch.cat([missingness_preds, low_conf_pred, high_conf_pred], dim=1)
            extra_cols = 3
        else:
            # Default: include all (missingness, low_conf, high_conf)
            preds = torch.cat([modality_preds, missingness_preds, low_conf_pred, high_conf_pred], dim=1)
            extra_cols = 3  # three extra columns for missingness, low_conf, and high_conf
        
        # Build the extended missingness matrix.
        # Start with the base missingness matrix and append 'extra_cols' columns of ones.
        extended_missingness_matrix = torch.cat(
            [missingness_matrix_device] +
            [torch.ones(batch_size, 1, device=modality_preds.device) for _ in range(extra_cols)],
            dim=1
        )
        
        # Expand the missingness matrix to cover the class dimension.
        extended_missingness_matrix = extended_missingness_matrix.unsqueeze(-1).repeat(1, 1, self.args.num_classes)
        extended_missingness_matrix = extended_missingness_matrix.view(batch_size, -1)
        
        # Compute the normalized weights from learnable parameters.
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Apply the missingness mask to predictions.
        if self.args.ablation == "without_late_module":
            masked_preds = preds
        else:
            masked_preds = preds * extended_missingness_matrix
        
        # Expand the normalized weights to match the batch size.
        normalized_weights = normalized_weights.unsqueeze(0).repeat(batch_size, 1, 1)  # Expected shape: [batch_size, modalities+extra, num_classes]
        # Flatten the weight tensor for further processing.
        normalized_weights = normalized_weights.view(batch_size, -1)
        
        # Compute the modality weights using the extended missingness mask.
        if self.args.ablation == "without_late_module":
            modality_weights = normalized_weights
        else:
            modality_weights = normalized_weights * extended_missingness_matrix
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Reshape predictions and weights to separate the class dimension.
        masked_preds = masked_preds.view(batch_size, -1, self.args.num_classes)
        modality_weights = modality_weights.view(batch_size, -1, self.args.num_classes)
        
        # Fuse the predictions using a weighted sum.
        fused_preds_final = torch.sum(masked_preds * modality_weights, dim=1)
        
        return {'high_conf': high_conf_pred, 'low': low_conf_pred, 'late': fused_preds_final }
    
class EnsembleFusion(nn.Module):
    def __init__(self, args):
        """
        Args:
            args: Argument namespace with model configurations.
                - args.load_model_1, args.load_model_2, args.load_model_3: Paths to model checkpoints.
                - args.ensemble_type: Specifies the ensemble type - "early", "joint", "late", or "mixed".
        """
        super(EnsembleFusion, self).__init__()
        self.args = args
        self.ensemble_type = args.load_early  # "early", "joint", "late", or "mixed"

        # Load models based on the ensemble type
        self.model1 = self._load_model(args.load_model_1, args.load_ehr_1, args.load_cxr_1, args.load_rr_1, args.load_dn_1, "model1")
        self.model2 = self._load_model(args.load_model_2, args.load_ehr_2, args.load_cxr_2, args.load_rr_2, args.load_dn_2, "model2")
        self.model3 = self._load_model(args.load_model_3, args.load_ehr_3, args.load_cxr_3, args.load_rr_3, args.load_dn_3, "model3")

    def _load_model(self, checkpoint_path, load_ehr, load_cxr, load_rr, load_dn, model_name):
        """
        Load the appropriate fusion model based on the ensemble type and checkpoint path.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model_name (str): Name of the model being loaded (e.g., "model1", "model2").

        Returns:
            A fully initialized and checkpoint-loaded fusion model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Step 1: Create fresh encoder instances
        ehr_model = EHR_encoder(self.args) if 'EHR' in self.args.modalities else None
        cxr_model = CXR_encoder(self.args) if 'CXR' in self.args.modalities else None
        text_model = Text_encoder(self.args, self.device) if any(m in self.args.modalities for m in ['RR', 'DN']) else None

        # Step 2: Instantiate the fusion model based on ensemble_type
        if self.ensemble_type == "early":
            model = EarlyFusion(self.args, ehr_model, cxr_model, text_model)
        elif self.ensemble_type == "joint":
            model = JointFusion(self.args, ehr_model, cxr_model, text_model)
        elif self.ensemble_type == "late":
            model = LateFusion(self.args, ehr_model, cxr_model, text_model)
        elif self.ensemble_type == "mixed":
            # Mixed ensemble - load specific fusion model for each model name
            if "model1" in model_name:
                model = EarlyFusion(self.args, ehr_model, cxr_model, text_model)
            elif "model2" in model_name:
                model = JointFusion(self.args, ehr_model, cxr_model, text_model)
            elif "model3" in model_name:
                model = LateFusion(self.args, ehr_model, cxr_model, text_model)
            else:
                raise ValueError("Invalid model name for mixed ensemble.")
        else:
            raise ValueError("Unknown ensemble type specified!")

        # Step 3: Load checkpoint and handle state_dict
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
            # Extract state_dict and clean up any unnecessary prefixes
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
    
            # Clean state_dict: Remove 'fusion_model.' prefix if it exists
            cleaned_state_dict = self._clean_state_dict(state_dict, prefix="fusion_model.")
    
            # Load the state_dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
            print(f"Model {model_name} loaded from {checkpoint_path}")
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        else:
            if load_ehr:
                checkpoint = torch.load(load_ehr, map_location=self.device)
                # Extract state_dict and clean up any unnecessary prefixes
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
        
                # Clean state_dict: Remove 'fusion_model.' prefix if it exists
                cleaned_state_dict = self._clean_state_dict(state_dict, prefix="fusion_model.")
        
                # Load the state_dict into the model
                missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
                print(f"loaded from {load_ehr}")
                print("Missing keys:", missing_keys)
                print("Unexpected keys:", unexpected_keys)
            if load_cxr:
                checkpoint = torch.load(load_cxr, map_location=self.device)
                # Extract state_dict and clean up any unnecessary prefixes
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
        
                # Clean state_dict: Remove 'fusion_model.' prefix if it exists
                cleaned_state_dict = self._clean_state_dict(state_dict, prefix="fusion_model.")
        
                # Load the state_dict into the model
                missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
                print(f"loaded from {load_cxr}")
                print("Missing keys:", missing_keys)
                print("Unexpected keys:", unexpected_keys)
            if load_rr:
                checkpoint = torch.load(load_rr, map_location=self.device)
                # Extract state_dict and clean up any unnecessary prefixes
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
        
                # Clean state_dict: Remove 'fusion_model.' prefix if it exists
                cleaned_state_dict = self._clean_state_dict(state_dict, prefix="fusion_model.")
        
                # Load the state_dict into the model
                missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
                print(f"loaded from {load_rr}")
                print("Missing keys:", missing_keys)
                print("Unexpected keys:", unexpected_keys)
            if load_dn:
                checkpoint = torch.load(load_dn, map_location=self.device)
                # Extract state_dict and clean up any unnecessary prefixes
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
        
                # Clean state_dict: Remove 'fusion_model.' prefix if it exists
                cleaned_state_dict = self._clean_state_dict(state_dict, prefix="fusion_model.")
        
                # Load the state_dict into the model
                missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
                print(f"loaded from {load_dn}")
                print("Missing keys:", missing_keys)
                print("Unexpected keys:", unexpected_keys)

        model.to(self.device)
        model.eval()  # Set model to evaluation mode
        return model

    @staticmethod
    def _clean_state_dict(state_dict, prefix="fusion_model."):
        """
        Remove the specified prefix from state_dict keys.

        Args:
            state_dict (dict): The state_dict to clean.
            prefix (str): The prefix to remove.

        Returns:
            dict: The cleaned state_dict.
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[len(prefix):] if k.startswith(prefix) else k
            new_state_dict[new_key] = v
        return new_state_dict

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        """
        Forward pass through the ensemble.

        Args:
            x: EHR data.
            seq_lengths: Sequence lengths for EHR.
            img: CXR image data.
            pairs: Additional modality pairs.
            rr: Text data for RR modality.
            dn: Text data for DN modality.

        Returns:
            A dictionary containing the ensemble prediction.
        """
        predictions = []

        # Forward pass through each model in the ensemble
        for model in [self.model1, self.model2, self.model3]:
            if isinstance(model, EarlyFusion):
                pred = model(x, seq_lengths, img, pairs, rr, dn)['early']
            elif isinstance(model, JointFusion):
                pred = model(x, seq_lengths, img, pairs, rr, dn)['joint']
            elif isinstance(model, LateFusion):
                pred = model(x, seq_lengths, img, pairs, rr, dn)['late']
            else:
                raise ValueError("Unknown fusion model type.")
            predictions.append(pred)

        # Average predictions across all models
        ensemble_output = torch.stack(predictions, dim=0).mean(dim=0)

        return {'ensemble': ensemble_output, 'unified': ensemble_output}    
        
class HealNetBaselineFusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, text_model, healnet: HealNet, modalities: List[str]):
        super().__init__()
        self.args = args
        self.modalities = modalities
        self.ehr_model = ehr_model if 'EHR' in modalities else None
        self.cxr_model = cxr_model if 'CXR' in modalities else None
        self.text_model = text_model if any(m in modalities for m in ['DN', 'RR']) else None
        self.healnet = healnet

        # Store placeholders for comparison
        self.ehr_placeholder = torch.zeros(1, 10)  # adjust if your collate_fn uses something else
        self.cxr_placeholder = torch.zeros(3, 384, 384)  # as in your collate_fn
        self.text_placeholder = ""  # DN and RR use empty string

    def forward(self, x=None, seq_lengths=None, img=None,
                pairs=None, rr=None, dn=None):

        B = len(seq_lengths)
        tensor_batches = [[] for _ in range(B)]   # final per-sample list

        # ---------- 1.   build boolean masks ---------------------------------
        ehr_present = ~((x == 0).flatten(1).all(dim=1))       # (B,)
        cxr_present = ~((img == 0).flatten(1).all(dim=1))    # (B,)
        dn_present  = torch.tensor([note.strip() != "" for note in dn], device=x.device)
        rr_present  = torch.tensor([note.strip() != "" for note in rr], device=x.device)

        # ---------- 2.   EHR --------------------------------------------------
        if 'EHR' in self.modalities:
            _, ehr_feats_full = self.ehr_model(x, seq_lengths)                                            # (N_pres, T, D)
            idx = 0
            for i in range(B):
                if ehr_present[i]:
                    tensor_batches[i].append(ehr_feats_full[idx]); idx += 1
                else:
                    tensor_batches[i].append(None)

        # ---------- 3.   CXR --------------------------------------------------
        if 'CXR' in self.modalities:
            if cxr_present.any():
                _, cxr_feats_full = self.cxr_model(img[cxr_present])  # (N_pres, T, D)
            idx = 0
            for i in range(B):
                if cxr_present[i]:
                    tensor_batches[i].append(cxr_feats_full[idx]); idx += 1
                else:
                    tensor_batches[i].append(None)

        # ---------- 4.   DN ---------------------------------------------------
        if 'DN' in self.modalities:
            if dn_present.any():
                _, dn_feats_full, _, _ = self.text_model(
                    dn_notes=[dn[j] for j in range(B) if dn_present[j]]
                )                                           # (N_pres, 512, D)
            idx = 0
            for i in range(B):
                if dn_present[i]:
                    tensor_batches[i].append(dn_feats_full[idx]); idx += 1
                else:
                    tensor_batches[i].append(None)

        # ---------- 5.   RR ---------------------------------------------------
        if 'RR' in self.modalities:
            if rr_present.any():
                _, _, _, rr_feats_full = self.text_model(
                    rr_notes=[rr[j] for j in range(B) if rr_present[j]]
                )
            idx = 0
            for i in range(B):
                if rr_present[i]:
                    tensor_batches[i].append(rr_feats_full[idx]); idx += 1
                else:
                    tensor_batches[i].append(None)

        # ---------- 6.   HealNet call ----------------------------------------
        logits = []
        for sample in tensor_batches:
            batched = [None if t is None else t.unsqueeze(0) for t in sample]
            logits.append(self.healnet(batched))
        logits = torch.cat(logits, dim=0)    # (B, num_classes)

        return {'healnet': logits, 'unified': logits}
        
class HealNetRawFusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, text_model, healnet: HealNet, modalities: List[str]):
        super().__init__()
        self.args = args
        self.modalities = modalities
        self.ehr_model = ehr_model if 'EHR' in modalities else None
        self.cxr_model = cxr_model if 'CXR' in modalities else None
        self.text_model = text_model if any(m in modalities for m in ['DN', 'RR']) else None
        self.healnet = healnet

        # Store placeholders for comparison
        self.ehr_placeholder = torch.zeros(1, 10)  # adjust if your collate_fn uses something else
        self.cxr_placeholder = torch.zeros(3, 384, 384)  # as in your collate_fn
        self.text_placeholder = ""  # DN and RR use empty string

    def forward(self, x=None, seq_lengths=None, img=None,
                pairs=None, rr=None, dn=None):

        B = len(seq_lengths)
        tensor_batches = [[] for _ in range(B)]   # final per-sample list

        # ---------- 1.   build boolean masks ---------------------------------
        ehr_present = ~((x == 0).flatten(1).all(dim=1))       # (B,)
        cxr_present = ~((img == 0).flatten(1).all(dim=1))    # (B,)
        dn_present  = torch.tensor([note.strip() != "" for note in dn], device=x.device)
        rr_present  = torch.tensor([note.strip() != "" for note in rr], device=x.device)

        # ---------- 2.   EHR --------------------------------------------------
        if 'EHR' in self.modalities:                                         # (N_pres, T, D)
            for i in range(B):
                if ehr_present[i]:
                    tensor_batches[i].append(x[i])
                else:
                    tensor_batches[i].append(None)

        # ---------- 3.   CXR --------------------------------------------------
        if 'CXR' in self.modalities:
            img_raw = img.permute(0, 2, 3, 1).contiguous()
            for i in range(B):
                if cxr_present[i]:
                    tensor_batches[i].append(img_raw[i])
                else:
                    tensor_batches[i].append(None)

        # ---------- 4.   DN ---------------------------------------------------
        if 'DN' in self.modalities:
            if dn_present.any():
                _, dn_feats_full, _, _ = self.text_model(
                    dn_notes=[dn[j] for j in range(B) if dn_present[j]]
                )                                           # (N_pres, 512, D)
            idx = 0
            for i in range(B):
                if dn_present[i]:
                    tensor_batches[i].append(dn_feats_full[idx]); idx += 1
                else:
                    tensor_batches[i].append(None)

        # ---------- 5.   RR ---------------------------------------------------
        if 'RR' in self.modalities:
            if rr_present.any():
                _, _, _, rr_feats_full = self.text_model(
                    rr_notes=[rr[j] for j in range(B) if rr_present[j]]
                )
            idx = 0
            for i in range(B):
                if rr_present[i]:
                    tensor_batches[i].append(rr_feats_full[idx]); idx += 1
                else:
                    tensor_batches[i].append(None)

        # ---------- 6.   HealNet call ----------------------------------------
        logits = []
        for sample in tensor_batches:
            batched = [None if t is None else t.unsqueeze(0) for t in sample]
            logits.append(self.healnet(batched))
        logits = torch.cat(logits, dim=0)    # (B, num_classes)

        return {'healnet-raw': logits, 'unified': logits}
        
class EMSMAFusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(EMSMAFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities.split("-")  # Modalities we are considering
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        self.ehr_model_fixed = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model_fixed = cxr_model if 'CXR' in self.modalities else None
        self.text_model_fixed = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        # One output dimension assumed for simplicity. If multiclass, adjust accordingly.
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args) if 'EHR' in self.modalities else None
        self.ehr_confidence_predictor = ConfidencePredictor(self.args, self.ehr_model.full_feats_dim)
        self.ehr_temperature = nn.Parameter(torch.ones(48)) if args.task == 'in-hospital-mortality' else nn.Parameter(torch.ones(2646, args.num_classes))

        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
            self.cxr_confidence_predictor = ConfidencePredictor(self.args, self.cxr_model.full_feats_dim)
            d_in = self.cxr_model.cxr_encoder.projection_layer.in_features
            d_out = self.cxr_model.cxr_encoder.projection_layer.out_features
            
            self.cxr_high_proj = nn.Linear(d_in, d_out)
            self.cxr_low_proj  = nn.Linear(d_in, d_out)
            self.cxr_temperature = nn.Parameter(torch.ones(578)) if args.task == 'in-hospital-mortality' else nn.Parameter(torch.ones(578, args.num_classes))

                
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
            self.rr_confidence_predictor = ConfidencePredictor(self.args, self.text_model.full_feats_dim_rr)
            
            d_in = self.text_model.bert.config.hidden_size
            d_out = self.text_model.feats_dim_rr
            
            self.rr_high_proj = nn.Linear(d_in, d_out)
            self.rr_low_proj  = nn.Linear(d_in, d_out)
            self.rr_temperature = nn.Parameter(torch.ones(512)) if args.task == 'in-hospital-mortality' else nn.Parameter(torch.ones(512, args.num_classes))

            
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)
            self.dn_confidence_predictor = ConfidencePredictor(self.args, self.text_model.full_feats_dim_dn)
            
            d_in = self.text_model.bert.config.hidden_size
            d_out = self.text_model.feats_dim_dn
            
            self.dn_high_proj = nn.Linear(d_in, d_out)
            self.dn_low_proj  = nn.Linear(d_in, d_out)
            self.dn_temperature = nn.Parameter(torch.ones(512)) if args.task == 'in-hospital-mortality' else nn.Parameter(torch.ones(512, args.num_classes))

        # Missingness classifier: takes missingness vector [batch_size, num_modalities]
        self.missingness_classifier = Classifier(len(self.modalities), self.args)
        emb_dim = args.patch_output_dim   
        num_mods = len(self.modalities)  # number of modalities present

        if self.args.fuser is None:
            # In "none" mode, we simply concatenate the pooled representations.
            self.high_fuser_classifier = Classifier(input_dim=num_mods * self.args.num_classes * emb_dim, args=self.args)
            self.low_fuser_classifier  = Classifier(input_dim=num_mods * self.args.num_classes * emb_dim, args=self.args)
        elif self.args.fuser == "lstm":
            # In "lstm" mode, we process the sequence of modality embeddings.
            self.high_fuser_lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
            self.low_fuser_lstm  = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
            self.high_fuser_classifier = Classifier(input_dim=emb_dim, args=self.args)
            self.low_fuser_classifier  = Classifier(input_dim=emb_dim, args=self.args)
        elif self.args.fuser == "transformer":
            # In "transformer" mode, we use a transformer block.
            self.high_fuser_transformer = Transformer(dim=emb_dim, depth=1, heads=8, 
                                                        dim_head=emb_dim // 8, mlp_dim=emb_dim * 2, 
                                                        dropout=0.)
            self.low_fuser_transformer  = Transformer(dim=emb_dim, depth=1, heads=8, 
                                                        dim_head=emb_dim // 8, mlp_dim=emb_dim * 2, 
                                                        dropout=0.)
            self.high_fuser_classifier = Classifier(input_dim=emb_dim, args=self.args)
            self.low_fuser_classifier  = Classifier(input_dim=emb_dim, args=self.args)
        else:
            raise ValueError(f"Unsupported fuser type: {self.args.fuser}")
        # ---------------------------------------------------------

        self.num_modalities = num_mods

        if self.args.freeze:
            print('freeze activated')
            self.freeze_all()

        self.num_modalities = len(self.modalities)
        # Now we have unimodal preds, plus joint pred, plus missingness pred = num_modalities + 2
        if self.args.ablation == "without_joint_module":
            num_extra = 1  # only missingness prediction is added
        elif self.args.ablation == "without_missingness_module":
            num_extra = 2  # only low_conf and high_conf predictions are added
        elif self.args.ablation == "without_late_module":
            self.num_modalities = 0
            num_extra = 3
        else:
            num_extra = 3  # default: missingness, low_conf, and high_conf predictions all present
        
        # Initialize the weights parameter with dynamic dimension:
        self.weights = nn.Parameter(torch.ones(self.num_modalities + num_extra, self.args.num_classes))
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        # Freeze and count parameters
        if self.ehr_model_fixed:
            for p in self.ehr_model_fixed.parameters():
                p.requires_grad = False
            print(f"Frozen ehr_model_fixed: {count_parameters(self.ehr_model_fixed)} parameters")
        
        if self.cxr_model_fixed:
            for p in self.cxr_model_fixed.parameters():
                p.requires_grad = False
            print(f"Frozen cxr_model_fixed: {count_parameters(self.cxr_model_fixed)} parameters")
        
        if self.text_model_fixed:
            for p in self.text_model_fixed.parameters():
                p.requires_grad = False
            print(f"Frozen text_model_fixed: {count_parameters(self.text_model_fixed)} parameters")
        
        # Freeze confidence predictors
        if hasattr(self, 'ehr_confidence_predictor'):
            for p in self.ehr_confidence_predictor.parameters():
                p.requires_grad = False
            print(f"Frozen ehr_confidence_predictor: {count_parameters(self.ehr_confidence_predictor)} parameters")
        
        if hasattr(self, 'cxr_confidence_predictor'):
            for p in self.cxr_confidence_predictor.parameters():
                p.requires_grad = False
            print(f"Frozen cxr_confidence_predictor: {count_parameters(self.cxr_confidence_predictor)} parameters")
        
        if hasattr(self, 'rr_confidence_predictor'):
            for p in self.rr_confidence_predictor.parameters():
                p.requires_grad = False
            print(f"Frozen rr_confidence_predictor: {count_parameters(self.rr_confidence_predictor)} parameters")
        
        if hasattr(self, 'dn_confidence_predictor'):
            for p in self.dn_confidence_predictor.parameters():
                p.requires_grad = False
            print(f"Frozen dn_confidence_predictor: {count_parameters(self.dn_confidence_predictor)} parameters")
        
        # Freeze temperatures (usually scalar tensors, not modules)
        def count_tensor_params(tensor):
            return tensor.numel() if tensor is not None else 0
        
        if hasattr(self, 'ehr_temperature'):
            self.ehr_temperature.requires_grad = False
            print(f"Frozen ehr_temperature: {count_tensor_params(self.ehr_temperature)} parameters")
        
        if hasattr(self, 'cxr_temperature'):
            self.cxr_temperature.requires_grad = False
            print(f"Frozen cxr_temperature: {count_tensor_params(self.cxr_temperature)} parameters")
        
        if hasattr(self, 'rr_temperature'):
            self.rr_temperature.requires_grad = False
            print(f"Frozen rr_temperature: {count_tensor_params(self.rr_temperature)} parameters")
        
        if hasattr(self, 'dn_temperature'):
            self.dn_temperature.requires_grad = False
            print(f"Frozen dn_temperature: {count_tensor_params(self.dn_temperature)} parameters")
                
        if self.args.freeze:
            print('freeze activated')
            self.freeze_all()

    def freeze_all(self):
        for param in self.ehr_model.parameters():
            param.requires_grad = False
        if self.ehr_classifier:
            for param in self.ehr_classifier.parameters():
                param.requires_grad = False
        if 'CXR' in self.modalities:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
            for param in self.cxr_classifier.parameters():
                param.requires_grad = False
        if 'RR' in self.modalities:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.rr_classifier.parameters():
                param.requires_grad = False
        if 'DN' in self.modalities:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.dn_classifier.parameters():
                param.requires_grad = False
                
    def detect_missingness_batch(self, batch_size, cxr=None, dn=None, rr=None):
        missingness_matrix = torch.ones(batch_size, len(self.modalities), device='cpu')
        if 'CXR' in self.modalities and cxr is not None:
            cxr_missing_mask = torch.zeros(3, 384, 384, device=cxr.device)
            cxr_is_missing = (cxr == cxr_missing_mask).view(cxr.size(0), -1).all(dim=1)
            missingness_matrix[:, self.modalities.index('CXR')] = ~cxr_is_missing
        if 'DN' in self.modalities and dn is not None:
            dn_is_missing = torch.tensor([len(note.strip()) == 0 for note in dn], device='cpu')
            missingness_matrix[:, self.modalities.index('DN')] = ~dn_is_missing
        if 'RR' in self.modalities and rr is not None:
            rr_is_missing = torch.tensor([len(note.strip()) == 0 for note in rr], device='cpu')
            missingness_matrix[:, self.modalities.index('RR')] = ~rr_is_missing
        return missingness_matrix
        
    def equalize(self):
        """Ensures projection layers have the same weights and biases and aligns fixed models."""
        with torch.no_grad():
            if 'EHR' in self.modalities:
                if self.ehr_model_fixed:
                    self.ehr_model_fixed.load_state_dict(self.ehr_model.state_dict())
            if 'CXR' in self.modalities:
                self.cxr_high_proj.weight.copy_(self.cxr_model.cxr_encoder.projection_layer.weight)
                self.cxr_high_proj.bias.copy_(self.cxr_model.cxr_encoder.projection_layer.bias)
                self.cxr_low_proj.weight.copy_(self.cxr_model.cxr_encoder.projection_layer.weight)
                self.cxr_low_proj.bias.copy_(self.cxr_model.cxr_encoder.projection_layer.bias)
                
                if self.cxr_model_fixed:
                    self.cxr_model_fixed.load_state_dict(self.cxr_model.state_dict())

            if 'RR' in self.modalities:
                self.rr_high_proj.weight.copy_(self.text_model.fc_rr.weight)
                self.rr_high_proj.bias.copy_(self.text_model.fc_rr.bias)
                self.rr_low_proj.weight.copy_(self.text_model.fc_rr.weight)
                self.rr_low_proj.bias.copy_(self.text_model.fc_rr.bias)
                
                if self.text_model_fixed:
                    self.text_model_fixed.load_state_dict(self.text_model.state_dict())
                
            if 'DN' in self.modalities:
                self.dn_high_proj.weight.copy_(self.text_model.fc_dn.weight)
                self.dn_high_proj.bias.copy_(self.text_model.fc_dn.bias)
                self.dn_low_proj.weight.copy_(self.text_model.fc_dn.weight)
                self.dn_low_proj.bias.copy_(self.text_model.fc_dn.bias)
                
                if self.text_model_fixed:
                    self.text_model_fixed.load_state_dict(self.text_model.state_dict())
                    
            if self.ehr_model:
                for param in self.ehr_model_fixed.parameters():
                    param.requires_grad = False
            if self.cxr_model:
                for param in self.cxr_model_fixed.parameters():
                    param.requires_grad = False
            if self.text_model:
                for param in self.text_model_fixed.parameters():
                    param.requires_grad = False
    
    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        #print("[Forward] Starting forward pass")
        batch_size = None
        features = []
        preds = []
        high_pool, high_conf_scores = [], []  # lists of tensors per modality
        low_pool,  low_conf_scores  = [], []
        modality_names = [] 
        

        # Process each modality
        if 'EHR' in self.modalities:
            ehr_feats, full_ehr_feats = self.ehr_model(x, seq_lengths)
            _, full_ehr_conf_feats = self.ehr_model_fixed(x, seq_lengths)
            # print("[EHR] ehr_feats shape:", ehr_feats.shape)
            # print("[EHR] full_ehr_feats shape:", full_ehr_feats.shape)
            
            if len(ehr_feats.shape) > 2:  # Only apply pooling if there are more than 2 dimensions
                ehr_feats = ehr_feats.mean(dim=1)
            features.append(ehr_feats)
            batch_size = ehr_feats.size(0)
            missingness_matrix = self.detect_missingness_batch(batch_size, cxr=img, dn=dn, rr=rr)
            
            # print("[EHR] batch_size determined to be:", batch_size)
            #print("EHR feature shape:", ehr_feats.shape)
            ehr_pred = self.ehr_classifier(ehr_feats)
            # print("[EHR] ehr_pred shape:", ehr_pred.shape)
            preds.append(ehr_pred)
            
            ehr_conf_logits = self.ehr_confidence_predictor(full_ehr_conf_feats)
            seq_len = ehr_conf_logits.shape[1]
            temp_ehr = self.ehr_temperature[:seq_len].clamp_min(1e-9)
            scaled_ehr_conf_logits = ehr_conf_logits / temp_ehr.unsqueeze(0)

            # 3) take the maximum across classes -> [B, L]
            if self.args.num_classes > 1:
                ehr_conf_probs = torch.sigmoid(ehr_conf_logits)
                # ehr_confidences = torch.max(ehr_conf_probs, 1 - ehr_conf_probs).max(dim=-1).values
                ehr_confidences = - (ehr_conf_probs * torch.log(ehr_conf_probs + 1e-9) + (1 - ehr_conf_probs) * torch.log(1 - ehr_conf_probs + 1e-9))
            else:
                ehr_conf_probs = torch.sigmoid(scaled_ehr_conf_logits)
                ehr_confidences = - (ehr_conf_probs * torch.log(ehr_conf_probs + 1e-9) + (1 - ehr_conf_probs) * torch.log(1 - ehr_conf_probs + 1e-9))
            # print("[EHR] ehr_confidences shape:", ehr_confidences.shape)
            
            mask_ehr_high = (ehr_confidences >= self.args.ehr_confidence_threshold)
            mask_ehr_low  = ~mask_ehr_high
            
            if self.args.num_classes > 1:
                full_ehr_feats_exp = full_ehr_feats.unsqueeze(2)  
            
                # Expand masks to [B, L, num_classes, 1]
                mask_ehr_high_f = mask_ehr_high.float().unsqueeze(-1)
                mask_ehr_low_f  = mask_ehr_low.float().unsqueeze(-1)
                
                # Compute per-class weighted sums and averages
                high_sum = (full_ehr_feats_exp * mask_ehr_high_f).sum(dim=1)  # [B, num_classes, feat_dim]
                low_sum  = (full_ehr_feats_exp * mask_ehr_low_f).sum(dim=1)
            else:
                # Expand masks to [B, L_ehr, 1]
                mask_ehr_high_f = mask_ehr_high.unsqueeze(-1).float()
                mask_ehr_low_f  = mask_ehr_low.unsqueeze(-1).float()
                
                
        
                high_sum = (full_ehr_feats * mask_ehr_high_f).sum(dim=1)
                low_sum  = (full_ehr_feats * mask_ehr_low_f).sum(dim=1)
            count_high = mask_ehr_high_f.sum(dim=1).clamp_min(1e-9)
            count_low  = mask_ehr_low_f.sum(dim=1).clamp_min(1e-9)
            pooled_high = high_sum / count_high  # [B, emb_dim]
            pooled_low  = low_sum  / count_low   # [B, emb_dim]
            

            # Also compute average token confidence for the high tokens (for ordering later)
            conf_high = (ehr_confidences * mask_ehr_high.float()).sum(dim=1) / (mask_ehr_high.float().sum(dim=1).clamp_min(1e-9))
            high_pool.append(pooled_high)
            high_conf_scores.append(conf_high)
            low_pool.append(pooled_low)
            conf_low = (ehr_confidences * mask_ehr_low.float()).sum(dim=1) / (mask_ehr_low.float().sum(dim=1).clamp_min(1e-9))
            low_conf_scores.append(conf_low)
            # print("EHR mask_high sum:", mask_ehr_high.sum(), "mask_low sum:", mask_ehr_low.sum())
            
        if 'CXR' in self.modalities:
            # Get features and predictions
            cxr_feats, full_cxr_feats = self.cxr_model(img)
            _, full_cxr_conf_feats = self.cxr_model_fixed(img)
            cxr_feats = cxr_feats[:, 0, :]  # [B, feat_dim]
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            preds.append(cxr_pred)
            
            # Confidence computation
            cxr_conf_logits = self.cxr_confidence_predictor(full_cxr_conf_feats)
            seq_len = cxr_conf_logits.shape[1]
            temp_cxr = self.cxr_temperature[:seq_len].clamp_min(1e-9)
            scaled_cxr_conf_logits = cxr_conf_logits / temp_cxr.unsqueeze(0)
            cxr_conf_probs = torch.sigmoid(scaled_cxr_conf_logits)  # [B, L, num_classes] if num_classes > 1
            
            if self.args.num_classes > 1:
                cxr_confidences = - (cxr_conf_probs * torch.log(cxr_conf_probs + 1e-9) + (1 - cxr_conf_probs) * torch.log(1 - cxr_conf_probs + 1e-9))  # [B, L, num_classes]
            else:
                cxr_confidences = - (cxr_conf_probs * torch.log(cxr_conf_probs + 1e-9) + (1 - cxr_conf_probs) * torch.log(1 - cxr_conf_probs + 1e-9)) # [B, L, 1]
            
            # Get the projection outputs (note: projection is always applied)
            B, L_cxr, D_in = full_cxr_feats.shape
            flattened_cxr = full_cxr_feats.reshape(B * L_cxr, D_in)
            cxr_high_proj_out = self.cxr_high_proj(flattened_cxr).view(B, L_cxr, -1)
            cxr_low_proj_out  = self.cxr_low_proj(flattened_cxr).view(B, L_cxr, -1)
            
            # For num_classes > 1, we expand along a new dimension; else we leave as-is.
            if self.args.num_classes > 1:
                feats_exp_high = cxr_high_proj_out.unsqueeze(2)  # [B, L, 1, D_out]
                feats_exp_low  = cxr_low_proj_out.unsqueeze(2)
            else:
                feats_exp_high = cxr_high_proj_out  # [B, L, D_out]
                feats_exp_low  = cxr_low_proj_out
        
            # Create per-token masks (they already have the proper shape if num_classes==1)
            mask_high = (cxr_confidences >= self.args.cxr_confidence_threshold)  # [B, L, (num_classes)]
            mask_low  = ~mask_high
        
            # Expand the mask only if needed
            if self.args.num_classes > 1:
                mask_high_f = mask_high.float().unsqueeze(-1)  # [B, L, num_classes, 1]
                mask_low_f  = mask_low.float().unsqueeze(-1)
            else:
                mask_high_f = mask_high.unsqueeze(-1).float()   # [B, L, 1]
                mask_low_f  = mask_low.unsqueeze(-1).float()
            
            # Pool features over the token dimension (dim=1)
            high_sum = (feats_exp_high * mask_high_f).sum(dim=1)
            low_sum  = (feats_exp_low  * mask_low_f).sum(dim=1)
            count_high = mask_high_f.sum(dim=1).clamp_min(1e-9)
            count_low  = mask_low_f.sum(dim=1).clamp_min(1e-9)
            pooled_high = high_sum / count_high  # [B, (num_classes), D_out]
            pooled_low  = low_sum  / count_low
        
            # Compute average confidence per token (for ordering if needed)
            conf_high = (cxr_confidences * mask_high.float()).sum(dim=1) / (
                            mask_high.float().sum(dim=1).clamp_min(1e-9))
            conf_low  = (cxr_confidences * mask_low.float()).sum(dim=1) / (
                            mask_low.float().sum(dim=1).clamp_min(1e-9))
            
            high_pool.append(pooled_high)
            high_conf_scores.append(conf_high)
            low_pool.append(pooled_low)
            low_conf_scores.append(conf_low)
            # print("CXR mask_high sum:", mask_high.sum(), "mask_low sum:", mask_low.sum())
        

        if 'DN' in self.modalities and dn is not None:
            dn_feats, full_dn_feats, _, _ = self.text_model(dn_notes=dn)
            _, full_dn_conf_feats, _, _ = self.text_model_fixed(dn_notes=dn)
            dn_feats = dn_feats.mean(dim=1)
            features.append(dn_feats)
            dn_pred = self.dn_classifier(dn_feats)
            preds.append(dn_pred)
            
            dn_conf_logits = self.dn_confidence_predictor(full_dn_conf_feats)
            seq_len = dn_conf_logits.shape[1]
            temp_dn = self.dn_temperature[:seq_len].clamp_min(1e-9)
            scaled_dn_conf_logits = dn_conf_logits / temp_dn.unsqueeze(0)
            dn_conf_probs = torch.sigmoid(scaled_dn_conf_logits)
            
            if self.args.num_classes > 1:
                dn_confidences = - (dn_conf_probs * torch.log(dn_conf_probs + 1e-9) + (1 - dn_conf_probs) * torch.log(1 - dn_conf_probs + 1e-9))
            else:
                dn_confidences = - (dn_conf_probs * torch.log(dn_conf_probs + 1e-9) + (1 - dn_conf_probs) * torch.log(1 - dn_conf_probs + 1e-9))
                
            # Apply the DN projection layers to map to the common embedding dimension:
            B, L_dn, D_in = full_dn_feats.shape
            flattened_dn = full_dn_feats.reshape(B * L_dn, D_in)
            dn_high_proj_out = self.dn_high_proj(flattened_dn).view(B, L_dn, -1)
            dn_low_proj_out  = self.dn_low_proj(flattened_dn).view(B, L_dn, -1)
            
            # Expand if multi-class
            if self.args.num_classes > 1:
                feats_exp = dn_high_proj_out.unsqueeze(2)  # For high pooling, shape: [B, L, 1, feat_dim]
                feats_exp_low = dn_low_proj_out.unsqueeze(2)
            else:
                feats_exp = dn_high_proj_out
                feats_exp_low = dn_low_proj_out
        
            mask_high = (dn_confidences >= self.args.dn_confidence_threshold)
            mask_low  = ~mask_high
            
            if self.args.num_classes > 1:
                mask_high_f = mask_high.float().unsqueeze(-1)  # [B, L, num_classes, 1]
                mask_low_f  = mask_low.float().unsqueeze(-1)
            else:
                mask_high_f = mask_high.unsqueeze(-1).float()
                mask_low_f  = mask_low.unsqueeze(-1).float()
            
            high_sum = (feats_exp * mask_high_f).sum(dim=1)
            low_sum  = (feats_exp_low * mask_low_f).sum(dim=1)
            count_high = mask_high_f.sum(dim=1).clamp_min(1e-9)
            count_low  = mask_low_f.sum(dim=1).clamp_min(1e-9)
            pooled_high = high_sum / count_high
            pooled_low  = low_sum  / count_low
            
            conf_high = (dn_confidences * mask_high.float()).sum(dim=1) / (mask_high.float().sum(dim=1).clamp_min(1e-9))
            conf_low = (dn_confidences * mask_low.float()).sum(dim=1) / (mask_low.float().sum(dim=1).clamp_min(1e-9))
            
            high_pool.append(pooled_high)
            high_conf_scores.append(conf_high)
            low_pool.append(pooled_low)
            low_conf_scores.append(conf_low)
            #print("DN mask_high sum:", mask_high.sum(), "mask_low sum:", mask_low.sum())



        if 'RR' in self.modalities and rr is not None:
            _, _, rr_feats, full_rr_feats = self.text_model(rr_notes=rr)
            _, _, _, full_rr_conf_feats = self.text_model_fixed(rr_notes=rr)
            rr_feats = rr_feats.mean(dim=1)
            features.append(rr_feats)
            rr_pred = self.rr_classifier(rr_feats)
            preds.append(rr_pred)
            
            rr_conf_logits = self.rr_confidence_predictor(full_rr_conf_feats)
            seq_len = rr_conf_logits.shape[1]
            temp_rr = self.rr_temperature[:seq_len].clamp_min(1e-9)
            scaled_rr_conf_logits = rr_conf_logits / temp_rr.unsqueeze(0)
            rr_conf_probs = torch.sigmoid(scaled_rr_conf_logits)
            
            if self.args.num_classes > 1:
                rr_confidences = - (rr_conf_probs * torch.log(rr_conf_probs + 1e-9) + (1 - rr_conf_probs) * torch.log(1 - rr_conf_probs + 1e-9))
            else:
                rr_confidences = - (rr_conf_probs * torch.log(rr_conf_probs + 1e-9) + (1 - rr_conf_probs) * torch.log(1 - rr_conf_probs + 1e-9))
            
            # Apply the RR projection layers to map features to the common dimension:
            B, L_rr, D_in = full_rr_feats.shape
            flattened_rr = full_rr_feats.reshape(B * L_rr, D_in)
            rr_high_proj_out = self.rr_high_proj(flattened_rr).view(B, L_rr, -1)
            rr_low_proj_out  = self.rr_low_proj(flattened_rr).view(B, L_rr, -1)
            
            if self.args.num_classes > 1:
                feats_exp = rr_high_proj_out.unsqueeze(2)  # [B, L, 1, feat_dim]
                feats_exp_low = rr_low_proj_out.unsqueeze(2)
            else:
                feats_exp = rr_high_proj_out
                feats_exp_low = rr_low_proj_out
        
            mask_high = (rr_confidences >= self.args.rr_confidence_threshold)
            mask_low  = ~mask_high
            
            if self.args.num_classes > 1:
                mask_high_f = mask_high.float().unsqueeze(-1)  # [B, L, num_classes, 1]
                mask_low_f  = mask_low.float().unsqueeze(-1)
            else:
                mask_high_f = mask_high.unsqueeze(-1).float()
                mask_low_f  = mask_low.unsqueeze(-1).float()
            
            high_sum = (feats_exp * mask_high_f).sum(dim=1)
            low_sum  = (feats_exp_low * mask_low_f).sum(dim=1)
            count_high = mask_high_f.sum(dim=1).clamp_min(1e-9)
            count_low  = mask_low_f.sum(dim=1).clamp_min(1e-9)
            pooled_high = high_sum / count_high
            pooled_low  = low_sum  / count_low
            
            conf_high = (rr_confidences * mask_high.float()).sum(dim=1) / (mask_high.float().sum(dim=1).clamp_min(1e-9))
            conf_low = (rr_confidences * mask_low.float()).sum(dim=1) / (mask_low.float().sum(dim=1).clamp_min(1e-9))
            
            high_pool.append(pooled_high)
            high_conf_scores.append(conf_high)
            low_pool.append(pooled_low)
            low_conf_scores.append(conf_low)
            #print("RR mask_high sum:", mask_high.sum(), "mask_low sum:", mask_low.sum())


    
        high_tensor = torch.stack(high_pool, dim=1)  # high modality representations
        low_tensor  = torch.stack(low_pool, dim=1)
        high_conf_tensor = torch.stack(high_conf_scores, dim=1)  # [B, num_mods]
        low_conf_tensor  = torch.stack(low_conf_scores, dim=1)
        
        if self.args.num_classes > 1:
            B = high_tensor.size(0)
            emb_dim = high_tensor.size(-1)
            high_tensor = high_tensor.view(B, -1, emb_dim)  # shape: [B, num_modalities*num_classes, emb_dim]
            low_tensor = low_tensor.view(B, -1, emb_dim)
            high_conf_tensor = high_conf_tensor.view(B, -1)    # shape: [B, num_modalities*num_classes]
            low_conf_tensor = low_conf_tensor.view(B, -1)

        # Depending on fuser type, fuse the modality representations:
        if self.args.ablation != "without_joint_module":
            if self.args.fuser is None:
                # Simply concatenate along the feature dimension.
                high_joint = high_tensor.view(batch_size, -1)
                low_joint  = low_tensor.view(batch_size, -1)
                high_conf_pred = self.high_fuser_classifier(high_joint)
                low_conf_pred  = self.low_fuser_classifier(low_joint)
            elif self.args.fuser == "lstm":
                # For LSTM, we first sort modalities by confidence so that the highest confidence modality is last.
                # (Sorting is done per sample.)
                # For high tokens:
                sort_idx = torch.argsort(high_conf_tensor, dim=1, descending=False)
                # Gather sorted high modality embeddings.
                sorted_high = torch.gather(high_tensor, 1, sort_idx.unsqueeze(-1).expand(-1, -1, high_tensor.size(-1)))
                lstm_out, _ = self.high_fuser_lstm(sorted_high)  # lstm_out shape: [B, num_mods, emb_dim]
                high_rep = lstm_out[:, -1, :]  # take the last time-step
                high_conf_pred = self.high_fuser_classifier(high_rep)
                # For low tokens:
                sort_idx = torch.argsort(low_conf_tensor, dim=1, descending=False)
                sorted_low = torch.gather(low_tensor, 1, sort_idx.unsqueeze(-1).expand(-1, -1, low_tensor.size(-1)))
                lstm_out, _ = self.low_fuser_lstm(sorted_low)
                low_rep = lstm_out[:, -1, :]
                low_conf_pred = self.low_fuser_classifier(low_rep)
            elif self.args.fuser == "transformer":
                # For transformer fusion, we again sort modalities (if desired) and feed the sequence into the transformer.
                sort_idx = torch.argsort(high_conf_tensor, dim=1, descending=False)
                sorted_high = torch.gather(high_tensor, 1, sort_idx.unsqueeze(-1).expand(-1, -1, high_tensor.size(-1)))
                trans_out = self.high_fuser_transformer(sorted_high)  # [B, num_mods, emb_dim]
                high_rep = trans_out.mean(dim=1)
                high_conf_pred = self.high_fuser_classifier(high_rep)
    
                sort_idx = torch.argsort(low_conf_tensor, dim=1, descending=False)
                sorted_low = torch.gather(low_tensor, 1, sort_idx.unsqueeze(-1).expand(-1, -1, low_tensor.size(-1)))
                trans_out = self.low_fuser_transformer(sorted_low)
                low_rep = trans_out.mean(dim=1)
                low_conf_pred = self.low_fuser_classifier(low_rep)
            else:
                raise ValueError(f"Unsupported fuser type: {self.args.fuser}")
        
        missingness_matrix_device = missingness_matrix.to(preds[0].device)
        missingness_preds = self.missingness_classifier(missingness_matrix_device.float())
        
        # First, compute the base modality predictions.
        modality_preds = torch.cat(preds, dim=1)  # Shape: [batch_size, num_modalities]
        
        # Decide what additional predictions to include based on self.args.ablation.
        # Also, record the number of extra columns (ones) to be appended in the missingness matrix.
        if self.args.ablation == "without_joint_module":
            # Without joint module: exclude high and low confidence predictions.
            # Only include the missingness prediction.
            preds = torch.cat([modality_preds, missingness_preds], dim=1)
            extra_cols = 1  # one column corresponds to missingness_preds only
            high_conf_pred = missingness_preds
            low_conf_pred = missingness_preds
        elif self.args.ablation == "without_missingness_module":
            # Without missingness module: exclude missingness prediction.
            # Only include low and high confidence predictions.
            preds = torch.cat([modality_preds, low_conf_pred, high_conf_pred], dim=1)
            extra_cols = 2  # two columns: one for low_conf and one for high_conf
        elif self.args.ablation == "without_late_module":
            preds = torch.cat([missingness_preds, low_conf_pred, high_conf_pred], dim=1)
            extra_cols = 3
        else:
            # Default: include all (missingness, low_conf, high_conf)
            preds = torch.cat([modality_preds, missingness_preds, low_conf_pred, high_conf_pred], dim=1)
            extra_cols = 3  # three extra columns for missingness, low_conf, and high_conf
        
        # Build the extended missingness matrix.
        # Start with the base missingness matrix and append 'extra_cols' columns of ones.
        extended_missingness_matrix = torch.cat(
            [missingness_matrix_device] +
            [torch.ones(batch_size, 1, device=modality_preds.device) for _ in range(extra_cols)],
            dim=1
        )
        
        # Expand the missingness matrix to cover the class dimension.
        extended_missingness_matrix = extended_missingness_matrix.unsqueeze(-1).repeat(1, 1, self.args.num_classes)
        extended_missingness_matrix = extended_missingness_matrix.view(batch_size, -1)
        
        # Compute the normalized weights from learnable parameters.
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Apply the missingness mask to predictions.
        if self.args.ablation == "without_late_module":
            masked_preds = preds
        else:
            masked_preds = preds * extended_missingness_matrix
        
        # Expand the normalized weights to match the batch size.
        normalized_weights = normalized_weights.unsqueeze(0).repeat(batch_size, 1, 1)  # Expected shape: [batch_size, modalities+extra, num_classes]
        # Flatten the weight tensor for further processing.
        normalized_weights = normalized_weights.view(batch_size, -1)
        
        # Compute the modality weights using the extended missingness mask.
        if self.args.ablation == "without_late_module":
            modality_weights = normalized_weights
        else:
            modality_weights = normalized_weights * extended_missingness_matrix
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Reshape predictions and weights to separate the class dimension.
        masked_preds = masked_preds.view(batch_size, -1, self.args.num_classes)
        modality_weights = modality_weights.view(batch_size, -1, self.args.num_classes)
        
        # Fuse the predictions using a weighted sum.
        fused_preds_final = torch.sum(masked_preds * modality_weights, dim=1)
        
        return {'high_conf': high_conf_pred, 'low': low_conf_pred, 'late': fused_preds_final }

