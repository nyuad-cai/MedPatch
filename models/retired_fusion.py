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
        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
            self.cxr_confidence_predictor = ConfidencePredictor(self.args, self.cxr_model.full_feats_dim)
            d_in = self.cxr_model.cxr_encoder.projection_layer.in_features
            d_out = self.cxr_model.cxr_encoder.projection_layer.out_features
            
            self.cxr_high_proj = nn.Linear(d_in, d_out)
            self.cxr_low_proj  = nn.Linear(d_in, d_out)
                
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
            self.rr_confidence_predictor = ConfidencePredictor(self.args, self.text_model.full_feats_dim_rr)
            
            d_in = self.text_model.bert.config.hidden_size
            d_out = self.text_model.feats_dim_rr
            
            self.rr_high_proj = nn.Linear(d_in, d_out)
            self.rr_low_proj  = nn.Linear(d_in, d_out)
            
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)
            self.dn_confidence_predictor = ConfidencePredictor(self.args, self.text_model.full_feats_dim_dn)
            
            d_in = self.text_model.bert.config.hidden_size
            d_out = self.text_model.feats_dim_dn
            
            self.dn_high_proj = nn.Linear(d_in, d_out)
            self.dn_low_proj  = nn.Linear(d_in, d_out)
        
        # if self.ehr_model:
        #     for param in self.ehr_model.parameters():
        #         param.requires_grad = False
        # if self.cxr_model:
        #     for param in self.cxr_model.parameters():
        #         param.requires_grad = False
        # if self.text_model:
        #     for param in self.text_model.parameters():
        #         param.requires_grad = False

        # Missingness classifier: takes missingness vector [batch_size, num_modalities]
        self.missingness_classifier = Classifier(len(self.modalities), self.args)
        self.high_conf_classifier = Classifier(args.patch_output_dim, self.args)
        self.low_conf_classifier = Classifier(args.patch_output_dim, self.args)

        self.num_modalities = len(self.modalities)
        # Now we have unimodal preds, plus joint pred, plus missingness pred = num_modalities + 2
        self.weights = nn.Parameter(torch.ones(self.num_modalities + 1, args.num_classes))  # Dynamic classes
        
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
        all_high_proj_outputs = []  # Will hold [B, L_any, D_out] from each modality
        all_low_proj_outputs  = []
        all_high_masks        = []  # Will hold [B, L_any, 1] from each modality
        all_low_masks         = []
        

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
            high_conf_tokens = [[] for _ in range(batch_size)]
            low_conf_tokens = [[] for _ in range(batch_size)]
            # print("[EHR] batch_size determined to be:", batch_size)
            #print("EHR feature shape:", ehr_feats.shape)
            ehr_pred = self.ehr_classifier(ehr_feats)
            # print("[EHR] ehr_pred shape:", ehr_pred.shape)
            preds.append(ehr_pred)
            
            ehr_conf_logits = self.ehr_confidence_predictor(full_ehr_conf_feats)  # [batch_size, seq_len]
            ehr_conf_probs = torch.sigmoid(ehr_conf_logits)                   # [B, L, num_classes]

            # 3) take the maximum across classes -> [B, L]
            if self.args.num_classes > 1:
                ehr_confidences = ehr_conf_probs.max(dim=-1).values
            else:
                ehr_confidences = torch.where(ehr_conf_probs > (1 - ehr_conf_probs), ehr_conf_probs, 1 - ehr_conf_probs)
            # print("[EHR] ehr_confidences shape:", ehr_confidences.shape)
            
            mask_ehr_high = (ehr_confidences >= self.args.ehr_confidence_threshold)
            mask_ehr_low  = ~mask_ehr_high
    
            # Expand masks to [B, L_ehr, 1]
            mask_ehr_high_f = mask_ehr_high.unsqueeze(-1).float()
            mask_ehr_low_f  = mask_ehr_low.unsqueeze(-1).float()
    
            # Accumulate in our lists
            all_high_proj_outputs.append(full_ehr_feats)
            all_low_proj_outputs.append(full_ehr_feats)
            all_high_masks.append(mask_ehr_high_f)
            all_low_masks.append(mask_ehr_low_f)

        if 'CXR' in self.modalities:
            cxr_feats, full_cxr_feats = self.cxr_model(img)
            _, full_cxr_conf_feats = self.cxr_model(img)
            cxr_feats = cxr_feats[:, 0, :]
            # print("[CXR] After slicing [:, 0, :], cxr_feats shape:", cxr_feats.shape)
            # print("[CXR] full_cxr_feats shape:", full_cxr_feats.shape)
            
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            # print("[CXR] cxr_pred shape:", cxr_pred.shape)
            preds.append(cxr_pred)
            
            cxr_conf_logits = self.cxr_confidence_predictor(full_cxr_conf_feats)  # [batch_size, seq_len]
            cxr_conf_probs = torch.sigmoid(cxr_conf_logits)                   # [B, L, num_classes]

            # 3) take the maximum across classes -> [B, L]
            if self.args.num_classes > 1:
                cxr_confidences = cxr_conf_probs.max(dim=-1).values
            else:
                cxr_confidences = torch.where(cxr_conf_probs > (1 - cxr_conf_probs), cxr_conf_probs, 1 - cxr_conf_probs)
            
            # print("[CXR] cxr_confidences shape:", cxr_confidences.shape)
            
            mask_cxr_high = (cxr_confidences >= self.args.cxr_confidence_threshold)
            mask_cxr_low  = ~mask_cxr_high
    
            # Flatten for projection
            B, L_cxr, D_in = full_cxr_feats.shape
            flattened_cxr = full_cxr_feats.reshape(B * L_cxr, D_in)
    
            # Two separate projection layers
            cxr_high_proj_out = self.cxr_high_proj(flattened_cxr)  # [B*L_cxr, D_out]
            cxr_low_proj_out  = self.cxr_low_proj(flattened_cxr)   # [B*L_cxr, D_out]
    
            # Reshape back
            cxr_high_proj_out = cxr_high_proj_out.view(B, L_cxr, -1)
            cxr_low_proj_out  = cxr_low_proj_out.view(B, L_cxr, -1)
    
            # Expand masks
            mask_cxr_high_f = mask_cxr_high.unsqueeze(-1).float()
            mask_cxr_low_f  = mask_cxr_low.unsqueeze(-1).float()
    
            # Accumulate
            all_high_proj_outputs.append(cxr_high_proj_out)
            all_low_proj_outputs.append(cxr_low_proj_out)
            all_high_masks.append(mask_cxr_high_f)
            all_low_masks.append(mask_cxr_low_f)

        if 'DN' in self.modalities and dn is not None:
            dn_feats, full_dn_feats, _, _ = self.text_model(dn_notes=dn)
            _, full_dn_conf_feats, _, _ = self.text_model_fixed(dn_notes=dn)
            dn_feats = dn_feats.mean(dim=1)
            # print("[DN] dn_feats shape:", dn_feats.shape)
            # print("[DN] full_dn_feats shape:", full_dn_feats.shape)
            
            features.append(dn_feats)
            dn_pred = self.dn_classifier(dn_feats)
            # print("[DN] dn_pred shape:", dn_pred.shape)
            preds.append(dn_pred)
            
            dn_conf_logits = self.dn_confidence_predictor(full_dn_conf_feats)  # [batch_size, seq_len]
            dn_conf_probs = torch.sigmoid(dn_conf_logits)                   # [B, L, num_classes]

            # 3) take the maximum across classes -> [B, L]
            if self.args.num_classes > 1:
                dn_confidences = dn_conf_probs.max(dim=-1).values
            else:
                dn_confidences = torch.where(dn_conf_probs > (1 - dn_conf_probs), dn_conf_probs, 1 - dn_conf_probs)
            
            # print("[DN] dn_confidences shape:", dn_confidences.shape)
            
            mask_dn_high = (dn_confidences >= self.args.dn_confidence_threshold)
            mask_dn_low  = ~mask_dn_high
    
            B, L_dn, D_in = full_dn_feats.shape
            flattened_dn = full_dn_feats.reshape(B * L_dn, D_in)
    
            # Project for high/low
            dn_high_proj_out = self.dn_high_proj(flattened_dn)  # [B*L_dn, D_out]
            dn_low_proj_out  = self.dn_low_proj(flattened_dn)   # [B*L_dn, D_out]
    
            # Reshape
            dn_high_proj_out = dn_high_proj_out.view(B, L_dn, -1)
            dn_low_proj_out  = dn_low_proj_out.view(B, L_dn, -1)
    
            # Masks
            mask_dn_high_f = mask_dn_high.unsqueeze(-1).float()
            mask_dn_low_f  = mask_dn_low.unsqueeze(-1).float()
    
            all_high_proj_outputs.append(dn_high_proj_out)
            all_low_proj_outputs.append(dn_low_proj_out)
            all_high_masks.append(mask_dn_high_f)
            all_low_masks.append(mask_dn_low_f)

        if 'RR' in self.modalities and rr is not None:
            _, _, rr_feats, full_rr_feats = self.text_model(rr_notes=rr)
            _, _, _, full_rr_conf_feats = self.text_model(rr_notes=rr)
            rr_feats = rr_feats.mean(dim=1)
            # print("[RR] rr_feats shape:", rr_feats.shape)
            # print("[RR] full_rr_feats shape:", full_rr_feats.shape)
            
            features.append(rr_feats)
            rr_pred = self.rr_classifier(rr_feats)
            # print("[RR] rr_pred shape:", rr_pred.shape)
            preds.append(rr_pred)
            
            rr_conf_logits = self.rr_confidence_predictor(full_rr_conf_feats)  # [batch_size, seq_len]
            rr_conf_probs = torch.sigmoid(rr_conf_logits)                   # [B, L, num_classes]

            # 3) take the maximum across classes -> [B, L]
            if self.args.num_classes > 1:
                rr_confidences = rr_conf_probs.max(dim=-1).values
            else:
                rr_confidences = torch.where(rr_conf_probs > (1 - rr_conf_probs), rr_conf_probs, 1 - rr_conf_probs)
            
            # print("[RR] rr_confidences shape:", rr_confidences.shape)

            mask_rr_high = (rr_confidences >= self.args.rr_confidence_threshold)
            mask_rr_low  = ~mask_rr_high
    
            B, L_rr, D_in = full_rr_feats.shape
            flattened_rr = full_rr_feats.view(B * L_rr, D_in)
    
            rr_high_proj_out = self.rr_high_proj(flattened_rr)  # [B*L_rr, D_out]
            rr_low_proj_out  = self.rr_low_proj(flattened_rr)   # [B*L_rr, D_out]
    
            # Reshape
            rr_high_proj_out = rr_high_proj_out.view(B, L_rr, -1)
            rr_low_proj_out  = rr_low_proj_out.view(B, L_rr, -1)
    
            mask_rr_high_f = mask_rr_high.unsqueeze(-1).float()
            mask_rr_low_f  = mask_rr_low.unsqueeze(-1).float()
    
            all_high_proj_outputs.append(rr_high_proj_out)
            all_low_proj_outputs.append(rr_low_proj_out)
            all_high_masks.append(mask_rr_high_f)
            all_low_masks.append(mask_rr_low_f)
    
        all_high_proj = torch.cat(all_high_proj_outputs, dim=1)  # [B, L_total, D_out]
        all_low_proj  = torch.cat(all_low_proj_outputs,  dim=1)  # [B, L_total, D_out]
    
        all_high_mask = torch.cat(all_high_masks, dim=1)         # [B, L_total, 1]
        all_low_mask  = torch.cat(all_low_masks,  dim=1)         # [B, L_total, 1]
    
        # -------------------------------------------------------------------------
        # 6) Mean‐pool for high‐confidence and low‐confidence
        # -------------------------------------------------------------------------
        masked_high = all_high_proj * all_high_mask  # [B, L_total, D_out]
        masked_low  = all_low_proj  * all_low_mask   # [B, L_total, D_out]
    
        high_sum = masked_high.sum(dim=1)  # [B, D_out]
        low_sum  = masked_low.sum(dim=1)   # [B, D_out]
    
        high_count = all_high_mask.sum(dim=1).clamp_min(1e-9)  # [B, 1]
        low_count  = all_low_mask.sum(dim=1).clamp_min(1e-9)   # [B, 1]
    
        high_conf_batch = high_sum / high_count     # [B, D_out]
        low_conf_batch  = low_sum  / low_count      # [B, D_out]
        
        # print("[Forward] high_conf_batch shape:", high_conf_batch.shape)
        # print("[Forward] low_conf_batch  shape:", low_conf_batch.shape)
        
        missingness_matrix_device = missingness_matrix.to(preds[0].device)
        missingness_preds = self.missingness_classifier(missingness_matrix_device.float())
        
        # print("[Forward] missingness_preds shape:", missingness_preds.shape)
        
        high_conf_pred = self.high_conf_classifier(high_conf_batch)
        low_conf_pred =  self.low_conf_classifier(low_conf_batch)
        preds = torch.cat(preds, dim=1)  # [batch_size, num_modalities]
        # Add joint prediction and missingness prediction
        preds = torch.cat([preds, missingness_preds], dim=1)
        # preds shape: [batch_size, num_modalities + 1]
        
        extended_missingness_matrix = torch.cat(
            [missingness_matrix_device, 
             torch.ones(batch_size, 1, device=preds[0].device)], # missingness pred always present
            dim=1
        )
        # print("[Forward] missingness matrix shape:", extended_missingness_matrix.shape)
        extended_missingness_matrix = extended_missingness_matrix.unsqueeze(-1).repeat(1, 1, self.args.num_classes)
        # print("[Forward] missingness matrix shape:", extended_missingness_matrix.shape)
        extended_missingness_matrix = extended_missingness_matrix.view(batch_size, -1)
        # print("[Forward] missingness matrix shape:", extended_missingness_matrix.shape)
        
        normalized_weights = F.softmax(self.weights, dim=0)
        
        masked_preds = preds * extended_missingness_matrix
        
        # Expand weights to match batch size
        normalized_weights = normalized_weights.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, modalities + 2, num_classes]
        
        # Reshape weights to flatten class dimensions
        normalized_weights = normalized_weights.view(batch_size, -1)  # [batch_size, modalities * num_classes]
        
        modality_weights = normalized_weights * extended_missingness_matrix
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        masked_preds = masked_preds.view(batch_size, -1, self.args.num_classes)
        modality_weights = modality_weights.view(batch_size, -1, self.args.num_classes)
        fused_preds_final = torch.sum(masked_preds * modality_weights, dim=1)
        
        # print("[Forward] high conf shape:", high_conf_pred.shape)
        # print("[Forward] low conf shape:", low_conf_pred.shape)
        # print("[Forward] late preds shape:", fused_preds_final.shape)
        
        return {'high_conf': high_conf_pred, 'low': low_conf_pred, 'late': fused_preds_final }

class MSMAFusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(MSMAFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities.split("-")  # Modalities we are considering
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        # if self.ehr_model:
        #     for param in self.ehr_model.parameters():
        #         param.requires_grad = False
        # if self.cxr_model:
        #     for param in self.cxr_model.parameters():
        #         param.requires_grad = False
        # if self.text_model:
        #     for param in self.text_model.parameters():
        #         param.requires_grad = False
        
        
        # if 'CXR' in self.modalities:
        #     self.cxr_positional_embedding = nn.Parameter(torch.randn(1, 577, self.cxr_model.feats_dim))  # [1, seq_len, feats_dim]
        # if 'RR' in self.modalities:
        #     self.rr_positional_embedding = nn.Parameter(torch.randn(1, 512, self.text_model.feats_dim_rr))  # [1, seq_len, feats_dim]
        # if 'DN' in self.modalities:
        #     self.dn_positional_embedding = nn.Parameter(torch.randn(1, 512, self.text_model.feats_dim_dn))  # [1, seq_len, feats_dim]


        self.msma_joint_classifier = Classifier(args.output_dim + len(self.modalities), self.args)
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args)
        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)
        self.num_modalities = len(self.modalities)
        self.weights = nn.Parameter(torch.ones(self.num_modalities + 1))
        
        if self.args.freeze:
            self.freeze_all()
    
    def freeze_all(self):
        """Freeze all encoder and classifier parameters to prevent training."""
        # Freeze encoders
        for param in self.ehr_model.parameters():
                param.requires_grad = False
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
        missingness_matrix = torch.ones(batch_size, len(self.modalities), device='cpu')  # Start with all ones (present)
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

        #print("Missingness Matrix Shape:", missingness_matrix.shape)
        return missingness_matrix

    def similarity_calculator(self, embeddings, missingness_matrix):
        # Get the device from the embeddings
        device = embeddings[0].device
        # Ensure the missingness_matrix is on the same device
        missingness_matrix = missingness_matrix.to(device)
        batch_size = missingness_matrix.size(0)
        embedding_dim = embeddings[0].size(1)  # Assuming all embeddings have the same dimension
        num_modalities = len(self.modalities)
    
        # Final tensor to hold fused representations
        final_fused_representation = torch.zeros(batch_size, embedding_dim, device=embeddings[0].device)
    
        # Helper to compute cosine similarity and inverse similarity
        def cosine_and_inverse(emb1, emb2):
            sim = F.cosine_similarity(emb1, emb2, dim=1)  # Shape: [num_valid_samples]
            inv_sim = 1 / (1 + sim)  # Inverse similarity for weighting
            return sim, inv_sim
    
        # Initialize masks to track handled samples
        handled = torch.zeros(batch_size, dtype=torch.bool, device=embeddings[0].device)
    
        # Step 1: Process samples with all four modalities
        if num_modalities >= 4:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    for k in range(j + 1, num_modalities):
                        for l in range(k + 1, num_modalities):
                            valid_indices = (
                                (missingness_matrix[:, i] == 1) &
                                (missingness_matrix[:, j] == 1) &
                                (missingness_matrix[:, k] == 1) &
                                (missingness_matrix[:, l] == 1)
                            )
                            if valid_indices.any():
                                emb_i = embeddings[i][valid_indices]
                                emb_j = embeddings[j][valid_indices]
                                emb_k = embeddings[k][valid_indices]
                                emb_l = embeddings[l][valid_indices]
    
                                # Pairwise cosine similarities
                                sim_ij, inv_sim_ij = cosine_and_inverse(emb_i, emb_j)
                                sim_ik, inv_sim_ik = cosine_and_inverse(emb_i, emb_k)
                                sim_il, inv_sim_il = cosine_and_inverse(emb_i, emb_l)
                                sim_jk, inv_sim_jk = cosine_and_inverse(emb_j, emb_k)
                                sim_jl, inv_sim_jl = cosine_and_inverse(emb_j, emb_l)
                                sim_kl, inv_sim_kl = cosine_and_inverse(emb_k, emb_l)
    
                                # Triplet fused representations and average cosine similarity
                                fused_ijk = (emb_i + emb_j + emb_k) / 3
                                fused_ijl = (emb_i + emb_j + emb_l) / 3
                                fused_ikl = (emb_i + emb_k + emb_l) / 3
                                fused_jkl = (emb_j + emb_k + emb_l) / 3
                                avg_sim_triplet_ijk = (sim_ij + sim_ik + sim_jk) / 3
                                avg_sim_triplet_ijl = (sim_ij + sim_il + sim_jl) / 3
                                avg_sim_triplet_ikl = (sim_ik + sim_il + sim_kl) / 3
                                avg_sim_triplet_jkl = (sim_jk + sim_jl + sim_kl) / 3
                                
                                inv_sim_triplet_ijk = 1 / (1 + avg_sim_triplet_ijk)
                                inv_sim_triplet_ijl = 1 / (1 + avg_sim_triplet_ijl)
                                inv_sim_triplet_ikl = 1 / (1 + avg_sim_triplet_ikl)
                                inv_sim_triplet_jkl = 1 / (1 + avg_sim_triplet_jkl)
                                
                                avg_sim_quadruplet = (sim_ij + sim_ik + sim_il + sim_jk + sim_jl + sim_kl) / 6
                                inv_sim_quadruplet = 1 / (1 + avg_sim_quadruplet)
    
                                # Quadruplet fused representation
                                fused_quadruplet = (emb_i + emb_j + emb_k + emb_l) / 4
    
                                # Combine all fused representations using inverse similarities as weights
                                total_weight = (
                                    inv_sim_ij + inv_sim_ik + inv_sim_il +
                                    inv_sim_jk + inv_sim_jl + inv_sim_kl +
                                    inv_sim_triplet_ijk + inv_sim_triplet_ijl + 
                                    inv_sim_triplet_ikl + inv_sim_triplet_jkl +
                                    inv_sim_quadruplet
                                )
                                weighted_fused = (
                                    (emb_i + emb_j) / 2 * inv_sim_ij.unsqueeze(1) +
                                    (emb_i + emb_k) / 2 * inv_sim_ik.unsqueeze(1) +
                                    (emb_i + emb_l) / 2 * inv_sim_il.unsqueeze(1) +
                                    (emb_j + emb_k) / 2 * inv_sim_jk.unsqueeze(1) +
                                    (emb_j + emb_l) / 2 * inv_sim_jl.unsqueeze(1) +
                                    (emb_k + emb_l) / 2 * inv_sim_kl.unsqueeze(1) +
                                    fused_ijk * inv_sim_triplet_ijk.unsqueeze(1) +
                                    fused_ijl * inv_sim_triplet_ijl.unsqueeze(1) +
                                    fused_ikl * inv_sim_triplet_ikl.unsqueeze(1) +
                                    fused_jkl * inv_sim_triplet_jkl.unsqueeze(1) +
                                    fused_quadruplet * inv_sim_quadruplet.unsqueeze(1)
                                ) / total_weight.unsqueeze(1)
    
                                # Update the final representation
                                final_fused_representation[valid_indices] = weighted_fused
                                handled[valid_indices] = True
    
        # Step 2: Process samples with three modalities
        if num_modalities >= 3:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    for k in range(j + 1, num_modalities):
                        valid_indices = (
                            (missingness_matrix[:, i] == 1) &
                            (missingness_matrix[:, j] == 1) &
                            (missingness_matrix[:, k] == 1) &
                            ~handled
                        )
                        if valid_indices.any():
                            emb_i = embeddings[i][valid_indices]
                            emb_j = embeddings[j][valid_indices]
                            emb_k = embeddings[k][valid_indices]
                            sim_ij, inv_sim_ij = cosine_and_inverse(emb_i, emb_j)
                            sim_ik, inv_sim_ik = cosine_and_inverse(emb_i, emb_k)
                            sim_jk, inv_sim_jk = cosine_and_inverse(emb_j, emb_k)
                            avg_sim_triplet = (sim_ij + sim_ik + sim_jk) / 3
                            inv_sim_triplet = 1 / (1 + avg_sim_triplet)
                            fused_triplet = (emb_i + emb_j + emb_k) / 3
                            total_weight = inv_sim_ij + inv_sim_ik + inv_sim_jk + inv_sim_triplet
                            weighted_fused = (
                                (emb_i + emb_j) / 2 * inv_sim_ij.unsqueeze(1) +
                                (emb_i + emb_k) / 2 * inv_sim_ik.unsqueeze(1) +
                                (emb_j + emb_k) / 2 * inv_sim_jk.unsqueeze(1) +
                                fused_triplet * inv_sim_triplet.unsqueeze(1)
                            ) / total_weight.unsqueeze(1)
                            final_fused_representation[valid_indices] = weighted_fused
                            handled[valid_indices] = True
    
        # Step 3: Process samples with two modalities
        if num_modalities >= 2:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    valid_indices = (
                        (missingness_matrix[:, i] == 1) &
                        (missingness_matrix[:, j] == 1) &
                        ~handled
                    )
                    if valid_indices.any():
                        emb_i = embeddings[i][valid_indices]
                        emb_j = embeddings[j][valid_indices]
                        _, inv_sim = cosine_and_inverse(emb_i, emb_j)
                        fused = (emb_i + emb_j) / 2
                        final_fused_representation[valid_indices] = fused
                        handled[valid_indices] = True
        
        # Step 4: Process samples with one modality
        for i in range(num_modalities):
            valid_indices = (missingness_matrix[:, i] == 1) & ~handled
            if valid_indices.any():
                emb_i = embeddings[i][valid_indices]
                final_fused_representation[valid_indices] = emb_i
                handled[valid_indices] = True

        print('shape of final representation:', final_fused_representation.shape)
        return final_fused_representation


    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        batch_size = None
        features = []
        preds = []

        if 'EHR' in self.modalities:
            if x is not None:
                ehr_feats = self.ehr_model(x, seq_lengths)
                if len(ehr_feats.shape) > 2:  # Only apply pooling if there are more than 2 dimensions
                    ehr_feats = ehr_feats.mean(dim=1)
                features.append(ehr_feats)
                batch_size = ehr_feats.size(0)
                #print("EHR feature shape:", ehr_feats.shape)
                ehr_pred = self.ehr_classifier(ehr_feats)
                preds.append(ehr_pred)
                print('ehr_pred',ehr_pred.shape)
            else:
                raise ValueError("EHR data is required.")

        missingness_matrix = self.detect_missingness_batch(batch_size, cxr=img, dn=dn, rr=rr)

        if 'CXR' in self.modalities and img is not None:
            cxr_feats = self.cxr_model(img)
            #print("CXR feature shape:", cxr_feats.shape)
            # cxr_feats = cxr_feats + self.cxr_positional_embedding
            # cxr_feats = cxr_feats.mean(dim=1)  # Apply mean pooling to get [16, 384]

            #print(f"Reduced CXR feature shape: {cxr_feats.shape}")
            cxr_feats = cxr_feats[:, 0, :]
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            print('cxr_pred',cxr_pred.shape)
            preds.append(cxr_pred)

        if self.text_model:
            if 'DN' in self.modalities and dn is not None:
                dn_feats, _ = self.text_model(dn_notes=dn)
                #dn_feats = dn_feats + self.dn_positional_embedding
                dn_feats = dn_feats.mean(dim=1)
                features.append(dn_feats)
                #print("DN feature shape:", dn_feats.shape)
                dn_pred = self.dn_classifier(dn_feats)
                preds.append(dn_pred)

            if 'RR' in self.modalities and rr is not None:
                _, rr_feats = self.text_model(rr_notes=rr)
                #rr_feats = rr_feats + self.rr_positional_embedding
                rr_feats = rr_feats.mean(dim=1)
                features.append(rr_feats)
                #print("RR feature shape:", rr_feats.shape)
                rr_pred = self.rr_classifier(rr_feats)
                preds.append(rr_pred)

        final_fused_representation = self.similarity_calculator(features, missingness_matrix)

        missingness_matrix = missingness_matrix.to(final_fused_representation.device)
        fused_with_missingness = torch.cat((final_fused_representation, missingness_matrix), dim=1)
        #print("Final fused representation shape:", fused_with_missingness.shape)

        for i, pred in enumerate(preds):
            print(f"Prediction {i} shape: {pred.shape}")
                # Convert preds (list of [batch_size, 1]) into a tensor [batch_size, num_modalities]
        preds = torch.cat(preds, dim=1)  # Shape: [batch_size, num_modalities]
        print("Preds concatenated shape:", preds.shape)

        # Generate a missingness matrix for the joint output (always present)
        joint_output = self.msma_joint_classifier(fused_with_missingness)  # Shape: [batch_size, 1]
        #print("Joint output shape:", joint_output.shape)
        joint_output_missingness = torch.ones(batch_size, 1, device=preds.device)  # Always included

        # Concatenate joint output to preds and extend missingness matrix
        preds = torch.cat([preds, joint_output], dim=1)  # Shape: [batch_size, num_modalities + 1]
        extended_missingness_matrix = torch.cat([missingness_matrix, joint_output_missingness], dim=1)  # Shape: [batch_size, num_modalities + 1]
        print("Extended preds shape:", preds.shape)
        print("Extended missingness matrix shape:", extended_missingness_matrix.shape)

        # Mask predictions for missing modalities (joint output is always included)
        masked_preds = preds * extended_missingness_matrix  # Shape: [batch_size, num_modalities + 1]
        # print("Masked preds shape:", masked_preds.shape)

        # Adjust weights to consider only non-missing modalities
        normalized_weights = F.softmax(self.weights, dim=0)  # Shape: [num_modalities + 1]
        modality_weights = normalized_weights * extended_missingness_matrix  # Zero out weights for missing modalities
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize per sample
        # print("Modality weights shape:", modality_weights.shape)

        # Compute the final fused prediction (weighted sum over non-missing modalities)
        fused_preds = torch.sum(masked_preds * modality_weights, dim=1)  # Shape: [batch_size, 1]
        # print("Final fused preds shape:", fused_preds.shape)

        return {'msma': fused_preds, 'unified': fused_preds}

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

class AblationMSMAFusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(AblationMSMAFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities.split("-")  # Modalities we are considering
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        if self.ehr_model:
            for param in self.ehr_model.parameters():
                param.requires_grad = False
        if self.cxr_model:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
        if self.text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
        if self.args.ablation == 'no_miss':
            self.msma_joint_classifier = Classifier(args.output_dim, self.args)
        else:
            self.msma_joint_classifier = Classifier(args.output_dim + len(self.modalities), self.args)
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args)
        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)
        self.num_modalities = len(self.modalities)
        
        if self.args.ablation == 'no_joint':
            self.weights = nn.Parameter(torch.ones(self.num_modalities))
        else:
            self.weights = nn.Parameter(torch.ones(self.num_modalities + 1))
        
        if self.args.freeze:
            self.freeze_all()
    
    def freeze_all(self):
        """Freeze all encoder and classifier parameters to prevent training."""
        # Freeze encoders
        for param in self.ehr_model.parameters():
                param.requires_grad = False
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
        missingness_matrix = torch.ones(batch_size, len(self.modalities), device='cpu')  # Start with all ones (present)
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

        #print("Missingness Matrix Shape:", missingness_matrix.shape)
        return missingness_matrix

    def similarity_calculator(self, embeddings, missingness_matrix):
        # Get the device from the embeddings
        device = embeddings[0].device
        # Ensure the missingness_matrix is on the same device
        missingness_matrix = missingness_matrix.to(device)
        batch_size = missingness_matrix.size(0)
        embedding_dim = embeddings[0].size(1)  # Assuming all embeddings have the same dimension
        num_modalities = len(self.modalities)
    
        # Final tensor to hold fused representations
        final_fused_representation = torch.zeros(batch_size, embedding_dim, device=embeddings[0].device)
    
        # Helper to compute cosine similarity and inverse similarity
        def cosine_and_inverse(emb1, emb2):
            sim = F.cosine_similarity(emb1, emb2, dim=1)  # Shape: [num_valid_samples]
            inv_sim = 1 / (1 + sim)  # Inverse similarity for weighting
            if self.args.ablation == 'reverse_w':
                return inv_sim, sim
            else:
                return sim, inv_sim
    
        # Initialize masks to track handled samples
        handled = torch.zeros(batch_size, dtype=torch.bool, device=embeddings[0].device)
    
        # Step 1: Process samples with all four modalities
        if num_modalities >= 4:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    for k in range(j + 1, num_modalities):
                        for l in range(k + 1, num_modalities):
                            valid_indices = (
                                (missingness_matrix[:, i] == 1) &
                                (missingness_matrix[:, j] == 1) &
                                (missingness_matrix[:, k] == 1) &
                                (missingness_matrix[:, l] == 1)
                            )
                            if valid_indices.any():
                                emb_i = embeddings[i][valid_indices]
                                emb_j = embeddings[j][valid_indices]
                                emb_k = embeddings[k][valid_indices]
                                emb_l = embeddings[l][valid_indices]
    
                                # Pairwise cosine similarities
                                sim_ij, inv_sim_ij = cosine_and_inverse(emb_i, emb_j)
                                sim_ik, inv_sim_ik = cosine_and_inverse(emb_i, emb_k)
                                sim_il, inv_sim_il = cosine_and_inverse(emb_i, emb_l)
                                sim_jk, inv_sim_jk = cosine_and_inverse(emb_j, emb_k)
                                sim_jl, inv_sim_jl = cosine_and_inverse(emb_j, emb_l)
                                sim_kl, inv_sim_kl = cosine_and_inverse(emb_k, emb_l)
    
                                # Triplet fused representations and average cosine similarity
                                fused_ijk = (emb_i + emb_j + emb_k) / 3
                                fused_ijl = (emb_i + emb_j + emb_l) / 3
                                fused_ikl = (emb_i + emb_k + emb_l) / 3
                                fused_jkl = (emb_j + emb_k + emb_l) / 3
                                avg_sim_triplet_ijk = (sim_ij + sim_ik + sim_jk) / 3
                                avg_sim_triplet_ijl = (sim_ij + sim_il + sim_jl) / 3
                                avg_sim_triplet_ikl = (sim_ik + sim_il + sim_kl) / 3
                                avg_sim_triplet_jkl = (sim_jk + sim_jl + sim_kl) / 3
                                
                                if self.args.ablation == 'reverse_w':
                                    inv_sim_triplet_ijk = (inv_sim_ij + inv_sim_ik + inv_sim_jk) / 3
                                    inv_sim_triplet_ijl = (inv_sim_ij + inv_sim_il + inv_sim_jl) / 3
                                    inv_sim_triplet_ikl = (inv_sim_ik + inv_sim_il + inv_sim_kl) / 3
                                    inv_sim_triplet_jkl = (inv_sim_jk + inv_sim_jl + inv_sim_kl) / 3
                                else:
                                    inv_sim_triplet_ijk = 1 / (1 + avg_sim_triplet_ijk)
                                    inv_sim_triplet_ijl = 1 / (1 + avg_sim_triplet_ijl)
                                    inv_sim_triplet_ikl = 1 / (1 + avg_sim_triplet_ikl)
                                    inv_sim_triplet_jkl = 1 / (1 + avg_sim_triplet_jkl)
                                
                                avg_sim_quadruplet = (sim_ij + sim_ik + sim_il + sim_jk + sim_jl + sim_kl) / 6
                                
                                if self.args.ablation == 'reverse_w':
                                    inv_sim_quadruplet = (inv_sim_ij + inv_sim_ik + inv_sim_il + inv_sim_jk + inv_sim_jl + inv_sim_kl) / 6
                                else:
                                    inv_sim_quadruplet = 1 / (1 + avg_sim_quadruplet)
    
                                # Quadruplet fused representation
                                fused_quadruplet = (emb_i + emb_j + emb_k + emb_l) / 4
    
                                # Combine all fused representations using inverse similarities as weights
                                if self.args.ablation == 'no_triplets':
                                    total_weight = (
                                        inv_sim_ij + inv_sim_ik + inv_sim_il +
                                        inv_sim_jk + inv_sim_jl + inv_sim_kl
                                    )
                                    weighted_fused = (
                                        (emb_i + emb_j) / 2 * inv_sim_ij.unsqueeze(1) +
                                        (emb_i + emb_k) / 2 * inv_sim_ik.unsqueeze(1) +
                                        (emb_i + emb_l) / 2 * inv_sim_il.unsqueeze(1) +
                                        (emb_j + emb_k) / 2 * inv_sim_jk.unsqueeze(1) +
                                        (emb_j + emb_l) / 2 * inv_sim_jl.unsqueeze(1) +
                                        (emb_k + emb_l) / 2 * inv_sim_kl.unsqueeze(1)
                                    ) / total_weight.unsqueeze(1)
                                elif self.args.ablation == 'no_pairs':
                                    total_weight = (
                                        inv_sim_triplet_ijk + inv_sim_triplet_ijl + 
                                        inv_sim_triplet_ikl + inv_sim_triplet_jkl +
                                        inv_sim_quadruplet
                                    )
                                    weighted_fused = (
                                        fused_ijk * inv_sim_triplet_ijk.unsqueeze(1) +
                                        fused_ijl * inv_sim_triplet_ijl.unsqueeze(1) +
                                        fused_ikl * inv_sim_triplet_ikl.unsqueeze(1) +
                                        fused_jkl * inv_sim_triplet_jkl.unsqueeze(1) +
                                        fused_quadruplet * inv_sim_quadruplet.unsqueeze(1)
                                    ) / total_weight.unsqueeze(1)
                                else:
                                    total_weight = (
                                        inv_sim_ij + inv_sim_ik + inv_sim_il +
                                        inv_sim_jk + inv_sim_jl + inv_sim_kl +
                                        inv_sim_triplet_ijk + inv_sim_triplet_ijl + 
                                        inv_sim_triplet_ikl + inv_sim_triplet_jkl +
                                        inv_sim_quadruplet
                                    )
                                    weighted_fused = (
                                        (emb_i + emb_j) / 2 * inv_sim_ij.unsqueeze(1) +
                                        (emb_i + emb_k) / 2 * inv_sim_ik.unsqueeze(1) +
                                        (emb_i + emb_l) / 2 * inv_sim_il.unsqueeze(1) +
                                        (emb_j + emb_k) / 2 * inv_sim_jk.unsqueeze(1) +
                                        (emb_j + emb_l) / 2 * inv_sim_jl.unsqueeze(1) +
                                        (emb_k + emb_l) / 2 * inv_sim_kl.unsqueeze(1) +
                                        fused_ijk * inv_sim_triplet_ijk.unsqueeze(1) +
                                        fused_ijl * inv_sim_triplet_ijl.unsqueeze(1) +
                                        fused_ikl * inv_sim_triplet_ikl.unsqueeze(1) +
                                        fused_jkl * inv_sim_triplet_jkl.unsqueeze(1) +
                                        fused_quadruplet * inv_sim_quadruplet.unsqueeze(1)
                                    ) / total_weight.unsqueeze(1)
        
                                # Update the final representation
                                final_fused_representation[valid_indices] = weighted_fused
                                handled[valid_indices] = True
    
        # Step 2: Process samples with three modalities
        if num_modalities >= 3:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    for k in range(j + 1, num_modalities):
                        valid_indices = (
                            (missingness_matrix[:, i] == 1) &
                            (missingness_matrix[:, j] == 1) &
                            (missingness_matrix[:, k] == 1) &
                            ~handled
                        )
                        if valid_indices.any():
                            emb_i = embeddings[i][valid_indices]
                            emb_j = embeddings[j][valid_indices]
                            emb_k = embeddings[k][valid_indices]
                            sim_ij, inv_sim_ij = cosine_and_inverse(emb_i, emb_j)
                            sim_ik, inv_sim_ik = cosine_and_inverse(emb_i, emb_k)
                            sim_jk, inv_sim_jk = cosine_and_inverse(emb_j, emb_k)
                            avg_sim_triplet = (sim_ij + sim_ik + sim_jk) / 3
                            
                            if self.args.ablation == 'reverse_w':
                                inv_sim_triplet = (inv_sim_ij + inv_sim_ik + inv_sim_jk) / 3
                            else:
                                inv_sim_triplet = 1 / (1 + avg_sim_triplet)
                            fused_triplet = (emb_i + emb_j + emb_k) / 3
                            if self.args.ablation == 'no_triplets':
                                    total_weight = (
                                        inv_sim_ij + inv_sim_ik + inv_sim_jk
                                    )
                                    weighted_fused = (
                                        (emb_i + emb_j) / 2 * inv_sim_ij.unsqueeze(1) +
                                        (emb_i + emb_k) / 2 * inv_sim_ik.unsqueeze(1) +
                                        (emb_j + emb_k) / 2 * inv_sim_jk.unsqueeze(1)
                                    ) / total_weight.unsqueeze(1)
                            elif self.args.ablation == 'no_pairs':
                                total_weight = (
                                    inv_sim_triplet
                                )
                                weighted_fused = (
                                    fused_triplet * inv_sim_triplet.unsqueeze(1)
                                ) / total_weight.unsqueeze(1)
                            else:
                                total_weight = inv_sim_ij + inv_sim_ik + inv_sim_jk + inv_sim_triplet
                                weighted_fused = (
                                    (emb_i + emb_j) / 2 * inv_sim_ij.unsqueeze(1) +
                                    (emb_i + emb_k) / 2 * inv_sim_ik.unsqueeze(1) +
                                    (emb_j + emb_k) / 2 * inv_sim_jk.unsqueeze(1) +
                                    fused_triplet * inv_sim_triplet.unsqueeze(1)
                                ) / total_weight.unsqueeze(1)
                            final_fused_representation[valid_indices] = weighted_fused
                            handled[valid_indices] = True
    
        # Step 3: Process samples with two modalities
        if num_modalities >= 2:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    valid_indices = (
                        (missingness_matrix[:, i] == 1) &
                        (missingness_matrix[:, j] == 1) &
                        ~handled
                    )
                    if valid_indices.any():
                        emb_i = embeddings[i][valid_indices]
                        emb_j = embeddings[j][valid_indices]
                        _, inv_sim = cosine_and_inverse(emb_i, emb_j)
                        fused = (emb_i + emb_j) / 2
                        final_fused_representation[valid_indices] = fused
                        handled[valid_indices] = True
        
        # Step 4: Process samples with one modality
        for i in range(num_modalities):
            valid_indices = (missingness_matrix[:, i] == 1) & ~handled
            if valid_indices.any():
                emb_i = embeddings[i][valid_indices]
                final_fused_representation[valid_indices] = emb_i
                handled[valid_indices] = True

        #print('shape of final representation:', final_fused_representation.shape)
        return final_fused_representation


    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        batch_size = None
        features = []
        preds = []

        if 'EHR' in self.modalities:
            if x is not None:
                ehr_feats = self.ehr_model(x, seq_lengths)
                if len(ehr_feats.shape) > 2:  # Only apply pooling if there are more than 2 dimensions
                    ehr_feats = ehr_feats.mean(dim=1)
                features.append(ehr_feats)
                batch_size = ehr_feats.size(0)
                #print("EHR feature shape:", ehr_feats.shape)
                ehr_pred = self.ehr_classifier(ehr_feats)
                preds.append(ehr_pred)
            else:
                raise ValueError("EHR data is required.")

        missingness_matrix = self.detect_missingness_batch(batch_size, cxr=img, dn=dn, rr=rr)

        if 'CXR' in self.modalities and img is not None:
            cxr_feats = self.cxr_model(img)
            #print("CXR feature shape:", cxr_feats.shape)
            # cxr_feats = cxr_feats + self.cxr_positional_embedding
            # cxr_feats = cxr_feats.mean(dim=1)  # Apply mean pooling to get [16, 384]

            #print(f"Reduced CXR feature shape: {cxr_feats.shape}")
            cxr_feats = cxr_feats[:, 0, :]
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            preds.append(cxr_pred)

        if self.text_model:
            if 'DN' in self.modalities and dn is not None:
                dn_feats, _ = self.text_model(dn_notes=dn)
                #dn_feats = dn_feats + self.dn_positional_embedding
                dn_feats = dn_feats.mean(dim=1)
                features.append(dn_feats)
                #print("DN feature shape:", dn_feats.shape)
                dn_pred = self.dn_classifier(dn_feats)
                preds.append(dn_pred)

            if 'RR' in self.modalities and rr is not None:
                _, rr_feats = self.text_model(rr_notes=rr)
                #rr_feats = rr_feats + self.rr_positional_embedding
                rr_feats = rr_feats.mean(dim=1)
                features.append(rr_feats)
                #print("RR feature shape:", rr_feats.shape)
                rr_pred = self.rr_classifier(rr_feats)
                preds.append(rr_pred)

        final_fused_representation = self.similarity_calculator(features, missingness_matrix)

        missingness_matrix = missingness_matrix.to(final_fused_representation.device)
        if self.args.ablation == 'no_miss':
            fused_with_missingness = final_fused_representation
        else:
            fused_with_missingness = torch.cat((final_fused_representation, missingness_matrix), dim=1)
        #print("Final fused representation shape:", fused_with_missingness.shape)

        # for i, pred in enumerate(preds):
        #     print(f"Prediction {i} shape: {pred.shape}")
                # Convert preds (list of [batch_size, 1]) into a tensor [batch_size, num_modalities]
        preds = torch.cat(preds, dim=1)  # Shape: [batch_size, num_modalities]
        #print("Preds concatenated shape:", preds.shape)

        # Generate a missingness matrix for the joint output (always present)
        joint_output = self.msma_joint_classifier(fused_with_missingness)  # Shape: [batch_size, 1]
        #print("Joint output shape:", joint_output.shape)
        joint_output_missingness = torch.ones(batch_size, 1, device=preds.device)  # Always included

        # Concatenate joint output to preds and extend missingness matrix
        if self.args.ablation == 'no_joint':
            extended_missingness_matrix = missingness_matrix
        else:
            preds = torch.cat([preds, joint_output], dim=1)  # Shape: [batch_size, num_modalities + 1]
            extended_missingness_matrix = torch.cat([missingness_matrix, joint_output_missingness], dim=1)  # Shape: [batch_size, num_modalities + 1]
        # print("Extended preds shape:", preds.shape)
        # print("Extended missingness matrix shape:", extended_missingness_matrix.shape)

        # Mask predictions for missing modalities (joint output is always included)
        masked_preds = preds * extended_missingness_matrix  # Shape: [batch_size, num_modalities + 1]
        # print("Masked preds shape:", masked_preds.shape)

        # Adjust weights to consider only non-missing modalities
        normalized_weights = F.softmax(self.weights, dim=0)  # Shape: [num_modalities + 1]
        modality_weights = normalized_weights * extended_missingness_matrix  # Zero out weights for missing modalities
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize per sample
        # print("Modality weights shape:", modality_weights.shape)

        # Compute the final fused prediction (weighted sum over non-missing modalities)
        if self.args.ablation == 'no_late':
            fused_preds = joint_output
        else:
            fused_preds = torch.sum(masked_preds * modality_weights, dim=1)  # Shape: [batch_size, 1]
            # print("Final fused preds shape:", fused_preds.shape)

        return {'msma': fused_preds, 'unified': fused_preds}

class NewMSMAFusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(NewMSMAFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities.split("-")  # Modalities we are considering
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        if self.ehr_model:
            for param in self.ehr_model.parameters():
                param.requires_grad = False
        if self.cxr_model:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
        if self.text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

        self.msma_joint_classifier = Classifier(args.output_dim, self.args)  
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args) if 'EHR' in self.modalities else None
        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)

        self.num_modalities = len(self.modalities)
        self.weights = nn.Parameter(torch.ones(self.num_modalities + 1))
        
        if self.args.freeze:
            self.freeze_all()

    def freeze_all(self):
        for param in self.ehr_model.parameters():
            param.requires_grad = False
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

    def similarity_calculator(self, embeddings, missingness_matrix):
        """
        Returns a dictionary structure containing:
        {
          'pairs': [(emb_pair, weight), ...],
          'triplets': [(emb_triplet, weight), ...],
          'quadruplets': [(emb_quadruplet, weight), ...]
        }
        
        where emb_pair/triplet/quadruplet is the averaged embedding of that combination
        and weight is the inverse similarity weight for that combination.
        
        Only returns combinations where all involved modalities are present.
        """
        device = embeddings[0].device
        missingness_matrix = missingness_matrix.to(device)
        batch_size = missingness_matrix.size(0)
        embedding_dim = embeddings[0].size(1)
        num_modalities = len(self.modalities)

        results = {
            'pairs': [],
            'triplets': [],
            'quadruplets': []
        }

        def cosine_and_inverse(emb1, emb2):
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            inv_sim = 1 / (1 + sim)
            return sim, inv_sim

        # Generate index combinations
        indices = list(range(num_modalities))

        # Pairs
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                valid_indices = (missingness_matrix[:, i] == 1) & (missingness_matrix[:, j] == 1)
                if valid_indices.any():
                    emb_i = embeddings[i][valid_indices]
                    emb_j = embeddings[j][valid_indices]
                    sim, inv_sim = cosine_and_inverse(emb_i, emb_j)
                    avg_emb = (emb_i + emb_j) / 2
                    # Store (embedding, weight, valid_mask)
                    results['pairs'].append((avg_emb, inv_sim, valid_indices))

        # Triplets
        if num_modalities >= 3:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    for k in range(j + 1, num_modalities):
                        valid_indices = (missingness_matrix[:, i] == 1) & (missingness_matrix[:, j] == 1) & (missingness_matrix[:, k] == 1)
                        if valid_indices.any():
                            emb_i = embeddings[i][valid_indices]
                            emb_j = embeddings[j][valid_indices]
                            emb_k = embeddings[k][valid_indices]
                            sim_ij, _inv_ij = cosine_and_inverse(emb_i, emb_j)
                            sim_ik, _inv_ik = cosine_and_inverse(emb_i, emb_k)
                            sim_jk, _inv_jk = cosine_and_inverse(emb_j, emb_k)
                            avg_sim = (sim_ij + sim_ik + sim_jk) / 3
                            inv_sim_triplet = 1 / (1 + avg_sim)
                            fused_triplet = (emb_i + emb_j + emb_k) / 3
                            results['triplets'].append((fused_triplet, inv_sim_triplet, valid_indices))

        # Quadruplets (if 4 or more modalities)
        if num_modalities >= 4:
            # For simplicity, handle just quadruplets of exactly 4 modalities. 
            # Extend if needed for more modalities.
            # Example: If exactly 4 modalities: EHR, CXR, DN, RR
            comb = indices
            valid_indices = torch.ones(batch_size, dtype=torch.bool, device=device)
            for m_idx in comb:
                valid_indices = valid_indices & (missingness_matrix[:, m_idx] == 1)
            if valid_indices.any():
                # Compute combined embedding
                selected_embs = [embeddings[m_idx][valid_indices] for m_idx in comb]
                fused_quadruplet = sum(selected_embs) / 4.0
                # Compute all pairwise similarities for weighting
                sims = []
                for a in range(num_modalities):
                    for b in range(a+1, num_modalities):
                        s, _ = cosine_and_inverse(selected_embs[a], selected_embs[b])
                        sims.append(s)
                avg_sim_quadruplet = torch.stack(sims, dim=0).mean(dim=0)
                inv_sim_quadruplet = 1 / (1 + avg_sim_quadruplet)
                results['quadruplets'].append((fused_quadruplet, inv_sim_quadruplet, valid_indices))

        return results

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        batch_size = None
        features = []
        preds = []

        # Unimodal features and predictions
        if 'EHR' in self.modalities:
            if x is not None:
                ehr_feats = self.ehr_model(x, seq_lengths)
                if len(ehr_feats.shape) > 2:
                    ehr_feats = ehr_feats.mean(dim=1)
                features.append(ehr_feats)
                batch_size = ehr_feats.size(0)
                ehr_pred = self.ehr_classifier(ehr_feats)
                preds.append(ehr_pred)
            else:
                raise ValueError("EHR data is required.")
        
        missingness_matrix = self.detect_missingness_batch(batch_size, cxr=img, dn=dn, rr=rr)

        if 'CXR' in self.modalities and img is not None:
            cxr_feats = self.cxr_model(img)
            cxr_feats = cxr_feats[:, 0, :]
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            preds.append(cxr_pred)

        if 'DN' in self.modalities and dn is not None:
            dn_feats, _ = self.text_model(dn_notes=dn)
            dn_feats = dn_feats.mean(dim=1)
            features.append(dn_feats)
            dn_pred = self.dn_classifier(dn_feats)
            preds.append(dn_pred)

        if 'RR' in self.modalities and rr is not None:
            _, rr_feats = self.text_model(rr_notes=rr)
            rr_feats = rr_feats.mean(dim=1)
            features.append(rr_feats)
            rr_pred = self.rr_classifier(rr_feats)
            preds.append(rr_pred)

        # Calculate all averaged embeddings and similarity weights
        # returns structure: { 'pairs': [(emb, weight, mask), ...],
        #                      'triplets': [(emb, weight, mask), ...],
        #                      'quadruplets': [(emb, weight, mask), ...] }
        combo_results = self.similarity_calculator(features, missingness_matrix)

        # Concatenate all these averaged embeddings
        # We need to consider that different sets (pairs, triplets, quadruplets) 
        # might have different valid subsets of samples.
        # Strategy: We'll create a list of all embeddings that are valid for each sample.
        # Then stack them to get [batch_size, N, feats_dim], where N is the total number of combos used.
        
        # First, determine how many combo embeddings each sample actually has.
        # We'll fill missing combos with zeros if a sample doesn't have that combo.
        all_combos = combo_results['pairs'] + combo_results['triplets'] + combo_results['quadruplets']
        num_combos = len(all_combos)
        if num_combos == 0:
            # No combos (e.g., only one modality present)
            # In this case, we can produce a fallback joint prediction (e.g., just zero?)
            fused_preds = torch.zeros_like(preds[0])  # no combos to fuse
            # Fuse unimodal preds with fused_preds as before
            # Just follow original logic:
            preds = torch.cat(preds, dim=1)
            joint_output = fused_preds.unsqueeze(1)  # [batch_size, 1]
            preds = torch.cat([preds, joint_output], dim=1)  
            # Weighted final fusion as before
            extended_missingness_matrix = torch.cat([missingness_matrix.to(preds.device), torch.ones(batch_size, 1, device=preds.device)], dim=1)
            masked_preds = preds * extended_missingness_matrix
            normalized_weights = F.softmax(self.weights, dim=0)
            modality_weights = normalized_weights * extended_missingness_matrix
            modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)
            fused_preds_final = torch.sum(masked_preds * modality_weights, dim=1)
            return {'msma': fused_preds_final, 'unified': fused_preds_final}

        # Prepare a [batch_size, num_combos, feats_dim] tensor filled with zeros
        feats_dim = features[0].size(1)
        combo_embeddings = torch.zeros(batch_size, num_combos, feats_dim, device=features[0].device)
        combo_weights = torch.zeros(batch_size, num_combos, device=features[0].device)

        # Fill combo_embeddings and combo_weights
        for idx, (emb, inv_sim, valid_mask) in enumerate(all_combos):
            # emb shape: [num_valid, feats_dim]
            # inv_sim shape: [num_valid]
            # valid_mask shape: [batch_size]
            # We place emb and inv_sim in the corresponding positions for those samples
            combo_embeddings[valid_mask, idx, :] = emb
            combo_weights[valid_mask, idx] = inv_sim

        # Pass combo_embeddings through joint classifier
        # Expecting msma_joint_classifier to handle [batch_size, num_combos, feats_dim] 
        # and produce [batch_size, num_combos, output_dim].
        joint_preds = self.msma_joint_classifier(combo_embeddings)  
        # joint_preds: [batch_size, num_combos, output_dim]

        # Now fuse these predictions using combo_weights
        # Normalize weights per sample over all combos that are present (nonzero weights)
        # Add small epsilon to avoid division by zero
        sum_weights = (combo_weights.sum(dim=1, keepdim=True) + 1e-8)
        normalized_combo_weights = combo_weights / sum_weights  
        normalized_combo_weights = normalized_combo_weights.unsqueeze(-1)  # [batch_size, num_combos, 1]

        # Weighted sum of predictions
        # joint_preds * normalized_combo_weights -> [batch_size, num_combos, output_dim]
        final_joint_pred = (joint_preds * normalized_combo_weights).sum(dim=1)  # [batch_size, output_dim]

        # Now fuse this joint pred with unimodal preds as before
        preds = torch.cat(preds, dim=1)  # [batch_size, num_modalities]
        # Add the joint prediction as an additional modality
        joint_output = final_joint_pred  # [batch_size, output_dim], assuming output_dim=1 or num_classes=1
        # If output_dim is > 1 (multiclass), you might need to adjust weighting logic accordingly.
        # For this example, we assume output_dim=1 for simplicity. Otherwise, you'd repeat the weighting logic similarly.

        # If output_dim > 1, the unimodal preds and joint_output should match dimensions. 
        # If unimodal preds are [batch_size, num_modalities], 
        # and joint_output is [batch_size, output_dim], 
        # ensure they match. For simplicity, assume all produce a single scalar output_dim=1.

        # Check dimension consistency:
        # If output_dim != 1, you'd need a different fusion strategy. For now, assume output_dim=1.
        preds = torch.cat([preds, joint_output], dim=1)  # [batch_size, num_modalities+1]

        # Missingness matrix for joint output (always present)
        extended_missingness_matrix = torch.cat(
            [missingness_matrix.to(preds.device), torch.ones(batch_size, 1, device=preds.device)], dim=1
        )

        # Mask predictions for missing modalities
        masked_preds = preds * extended_missingness_matrix

        # Adjust weights for final fusion over unimodal + joint
        normalized_weights = F.softmax(self.weights, dim=0)
        modality_weights = normalized_weights * extended_missingness_matrix
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)

        fused_preds = torch.sum(masked_preds * modality_weights, dim=1)

        return {'msma': fused_preds, 'unified': fused_preds}


class NovelMSMAFusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(NovelMSMAFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities.split("-")  # Modalities we are considering
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        if self.ehr_model:
            for param in self.ehr_model.parameters():
                param.requires_grad = False
        if self.cxr_model:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
        if self.text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

        # One output dimension assumed for simplicity. If multiclass, adjust accordingly.
        self.msma_joint_classifier = Classifier(args.output_dim, self.args)  
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args) if 'EHR' in self.modalities else None
        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)

        # Missingness classifier: takes missingness vector [batch_size, num_modalities]
        # Adjust output_dim or architecture based on your needs.
        self.missingness_classifier = Classifier(len(self.modalities), self.args)

        self.num_modalities = len(self.modalities)
        # Now we have unimodal preds, plus joint pred, plus missingness pred = num_modalities + 2
        self.weights = nn.Parameter(torch.ones(self.num_modalities + 1, args.num_classes))  # Dynamic classes

        
        if self.args.freeze:
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

    def similarity_calculator(self, embeddings, missingness_matrix):
        device = embeddings[0].device
        missingness_matrix = missingness_matrix.to(device)
        batch_size = missingness_matrix.size(0)
        embedding_dim = embeddings[0].size(1)
        num_modalities = len(self.modalities)

        results = {
            'pairs': [],
            'triplets': [],
            'quadruplets': []
        }

        def cosine_and_inverse(emb1, emb2):
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            inv_sim = 1 / (1 + sim)
            return sim, inv_sim

        indices = list(range(num_modalities))

        # Pairs
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                valid_indices = (missingness_matrix[:, i] == 1) & (missingness_matrix[:, j] == 1)
                if valid_indices.any():
                    emb_i = embeddings[i][valid_indices]
                    emb_j = embeddings[j][valid_indices]
                    sim, inv_sim = cosine_and_inverse(emb_i, emb_j)
                    avg_emb = (emb_i + emb_j) / 2
                    results['pairs'].append((avg_emb, inv_sim, valid_indices))

        # Triplets
        if num_modalities >= 3:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    for k in range(j + 1, num_modalities):
                        valid_indices = (missingness_matrix[:, i] == 1) & (missingness_matrix[:, j] == 1) & (missingness_matrix[:, k] == 1)
                        if valid_indices.any():
                            emb_i = embeddings[i][valid_indices]
                            emb_j = embeddings[j][valid_indices]
                            emb_k = embeddings[k][valid_indices]
                            sim_ij, _inv_ij = cosine_and_inverse(emb_i, emb_j)
                            sim_ik, _inv_ik = cosine_and_inverse(emb_i, emb_k)
                            sim_jk, _inv_jk = cosine_and_inverse(emb_j, emb_k)
                            avg_sim = (sim_ij + sim_ik + sim_jk) / 3
                            inv_sim_triplet = 1 / (1 + avg_sim)
                            fused_triplet = (emb_i + emb_j + emb_k) / 3
                            results['triplets'].append((fused_triplet, inv_sim_triplet, valid_indices))

        # Quadruplets (if 4 modalities)
        if num_modalities == 4:
            valid_indices = torch.ones(batch_size, dtype=torch.bool, device=device)
            for m_idx in indices:
                valid_indices = valid_indices & (missingness_matrix[:, m_idx] == 1)
            if valid_indices.any():
                selected_embs = [embeddings[m_idx][valid_indices] for m_idx in indices]
                fused_quadruplet = sum(selected_embs) / 4.0
                sims = []
                for a in range(num_modalities):
                    for b in range(a+1, num_modalities):
                        s, _ = cosine_and_inverse(selected_embs[a], selected_embs[b])
                        sims.append(s)
                avg_sim_quadruplet = torch.stack(sims, dim=0).mean(dim=0)
                inv_sim_quadruplet = 1 / (1 + avg_sim_quadruplet)
                results['quadruplets'].append((fused_quadruplet, inv_sim_quadruplet, valid_indices))

        return results

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        batch_size = None
        features = []
        preds = []

        # Unimodal features and predictions
        if 'EHR' in self.modalities:
            if x is not None:
                ehr_feats = self.ehr_model(x, seq_lengths)
                if len(ehr_feats.shape) > 2:
                    ehr_feats = ehr_feats.mean(dim=1)
                features.append(ehr_feats)
                batch_size = ehr_feats.size(0)
                ehr_pred = self.ehr_classifier(ehr_feats)
                preds.append(ehr_pred)
            else:
                raise ValueError("EHR data is required.")
        
        missingness_matrix = self.detect_missingness_batch(batch_size, cxr=img, dn=dn, rr=rr)

        if 'CXR' in self.modalities and img is not None:
            cxr_feats = self.cxr_model(img)
            cxr_feats = cxr_feats[:, 0, :]
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            preds.append(cxr_pred)

        if 'DN' in self.modalities and dn is not None:
            dn_feats, _ = self.text_model(dn_notes=dn)
            dn_feats = dn_feats.mean(dim=1)
            features.append(dn_feats)
            dn_pred = self.dn_classifier(dn_feats)
            preds.append(dn_pred)

        if 'RR' in self.modalities and rr is not None:
            _, rr_feats = self.text_model(rr_notes=rr)
            rr_feats = rr_feats.mean(dim=1)
            features.append(rr_feats)
            rr_pred = self.rr_classifier(rr_feats)
            preds.append(rr_pred)

        combo_results = self.similarity_calculator(features, missingness_matrix)

        all_combos = combo_results['pairs'] + combo_results['triplets'] + combo_results['quadruplets']
        num_combos = len(all_combos)
        
        if num_combos == 0:
            # No combos (only unimodal)
            # Just produce a fallback joint pred = 0
            fused_preds = torch.zeros_like(preds[0])  # no combos to fuse
            # We'll also incorporate missingness classifier here.
            # missingness_classifier takes missingness_matrix [batch_size, num_modalities]
            missingness_matrix_device = missingness_matrix.to(preds[0].device)
            missingness_preds = self.missingness_classifier(missingness_matrix_device.float())

            preds = torch.cat(preds, dim=1)
            joint_output = fused_preds  # joint pred
            # Add missingness pred
            preds = torch.cat([preds, joint_output, missingness_preds], dim=1)  
            extended_missingness_matrix = torch.cat(
                [missingness_matrix_device, 
                 torch.ones(batch_size, 1, device=preds.device),
                 torch.ones(batch_size, 1, device=preds.device)], dim=1
            )
            extended_missingness_matrix = extended_missingness_matrix.unsqueeze(-1).repeat(1, 1, self.args.num_classes)
            extended_missingness_matrix = extended_missingness_matrix.view(batch_size, -1)
            masked_preds = preds * extended_missingness_matrix
            
            normalized_weights = F.softmax(self.weights, dim=0)
            
            # Expand weights to match batch size
            normalized_weights = normalized_weights.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, modalities + 2, num_classes]
            
            # Reshape weights to flatten class dimensions
            normalized_weights = normalized_weights.view(batch_size, -1)  # [batch_size, modalities * num_classes]
            
            modality_weights = normalized_weights * extended_missingness_matrix
            modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            masked_preds = masked_preds.view(batch_size, -1, self.args.num_classes)
            modality_weights = modality_weights.view(batch_size, -1, self.args.num_classes)
            fused_preds_final = torch.sum(masked_preds * modality_weights, dim=1)
            return {'msma': fused_preds_final, 'unified': fused_preds_final}

        # If we have combinations:
        feats_dim = features[0].size(1)
        combo_embeddings = torch.zeros(batch_size, num_combos, feats_dim, device=features[0].device)
        combo_weights = torch.zeros(batch_size, num_combos, device=features[0].device)

        for idx, (emb, inv_sim, valid_mask) in enumerate(all_combos):
            combo_embeddings[valid_mask, idx, :] = emb
            combo_weights[valid_mask, idx] = inv_sim

        # Joint predictions from combos
        joint_preds = self.msma_joint_classifier(combo_embeddings)  
        # joint_preds: [batch_size, num_combos, output_dim]

        sum_weights = combo_weights.sum(dim=1, keepdim=True) + 1e-8
        normalized_combo_weights = (combo_weights / sum_weights).unsqueeze(-1)  # [batch_size, num_combos, 1]

        final_joint_pred = (joint_preds * normalized_combo_weights).sum(dim=1)  # [batch_size, output_dim]

        # Also get missingness predictions
        missingness_matrix_device = missingness_matrix.to(features[0].device)
        missingness_preds = self.missingness_classifier(missingness_matrix_device.float())
        
        # Now fuse final_joint_pred, unimodal preds, and missingness pred
        preds = torch.cat(preds, dim=1)  # [batch_size, num_modalities]
        # Add joint prediction and missingness prediction
        preds = torch.cat([preds, final_joint_pred, missingness_preds], dim=1)
        # preds shape: [batch_size, num_modalities + 2]

        # Extended missingness matrix: unimodals already have it, joint and missingness always present
        extended_missingness_matrix = torch.cat(
            [missingness_matrix_device, 
             torch.ones(batch_size, 1, device=preds.device),  # joint pred always present
             torch.ones(batch_size, 1, device=preds.device)], # missingness pred always present
            dim=1
        )
        
        
        # Expand the mask to match class dimensions
        extended_missingness_matrix = extended_missingness_matrix.unsqueeze(-1).repeat(1, 1, self.args.num_classes)  # [batch_size, 4, 25]
        
        # Reshape mask to flatten the modalities into class dimensions
        extended_missingness_matrix = extended_missingness_matrix.view(batch_size, -1)  # [batch_size, 25]

        
        masked_preds = preds * extended_missingness_matrix
        # Normalize weights across modalities and classes
        normalized_weights = F.softmax(self.weights, dim=0)  # [modalities + 2, num_classes]
        
        # Expand weights to match batch size
        normalized_weights = normalized_weights.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, modalities + 2, num_classes]
        
        # Reshape weights to flatten class dimensions
        normalized_weights = normalized_weights.view(batch_size, -1)  # [batch_size, modalities * num_classes]

        
        
        modality_weights = normalized_weights * extended_missingness_matrix
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        masked_preds = masked_preds.view(batch_size, -1, self.args.num_classes)
        modality_weights = modality_weights.view(batch_size, -1, self.args.num_classes)

        fused_preds = torch.sum(masked_preds * modality_weights, dim=1)
        
        #print("Shape of fused_preds:", fused_preds.shape)

        return {'msma': fused_preds, 'unified': fused_preds}
        
class Hypernetwork(nn.Module):
    def __init__(self, args, final_weights_dim, combo_weights_dim, input_dim):
        super(Hypernetwork, self).__init__()
        # A small MLP that takes an input vector per sample and outputs per-sample weights
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, final_weights_dim + combo_weights_dim)
        )
        self.final_weights_dim = final_weights_dim
        self.combo_weights_dim = combo_weights_dim

    def forward(self, x):
        # x: [batch_size, input_dim]
        out = self.mlp(x)
        final_weights = out[:, :self.final_weights_dim]
        combo_weights = out[:, self.final_weights_dim:self.final_weights_dim+self.combo_weights_dim]
        return final_weights, combo_weights


class SimpleMSMAFusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(SimpleMSMAFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities.split("-")
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None

        # Freeze encoders
        if self.ehr_model:
            for param in self.ehr_model.parameters():
                param.requires_grad = False
        if self.cxr_model:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
        if self.text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

        # Classifiers
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args) if 'EHR' in self.modalities else None
        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)

        self.missingness_classifier = Classifier(len(self.modalities), self.args)

        self.num_modalities = len(self.modalities)

        # Precompute all possible combinations
        self.combo_indices = []
        for r in [2, 3, 4]:
            if r <= self.num_modalities:
                for c in combinations(range(self.num_modalities), r):
                    self.combo_indices.append(c)
        self.num_combos_total = len(self.combo_indices)

        # Compute the base_dim as before
        if 'EHR' in self.modalities:
            base_dim = self.ehr_model.feats_dim
        elif 'CXR' in self.modalities:
            base_dim = self.cxr_model.feats_dim
        elif 'DN' in self.modalities:
            base_dim = self.text_model.feats_dim_dn
        else:
            base_dim = self.text_model.feats_dim_rr
        
        # Determine max number of modalities in any combo (if no combos, default to 1)
        max_combo_modality_count = max((len(c) for c in self.combo_indices), default=1)
        
        # Calculate the input dimension for the msma_joint_classifier
        combo_dim = max_combo_modality_count * base_dim
        
        # Now initialize msma_joint_classifier using combo_dim as input dimension
        self.msma_joint_classifier = Classifier(combo_dim, self.args)

        input_dim = (self.num_modalities * base_dim) + self.num_modalities
        self.hypernetwork = Hypernetwork(args, self.num_modalities + 2, self.num_combos_total, input_dim)

        if self.args.freeze:
            self.freeze_all()

    def freeze_all(self):
        if self.ehr_model:
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

    def similarity_calculator(self, embeddings, missingness_matrix):
        device = embeddings[0].device
        missingness_matrix = missingness_matrix.to(device)
        batch_size = missingness_matrix.size(0)

        results = {
            'pairs': [],
            'triplets': [],
            'quadruplets': []
        }

        # Instead of averaging, we concatenate embeddings
        # Pairs
        for c in [ci for ci in self.combo_indices if len(ci) == 2]:
            i, j = c
            valid_indices = (missingness_matrix[:, i] == 1) & (missingness_matrix[:, j] == 1)
            if valid_indices.any():
                emb_i = embeddings[i][valid_indices]
                emb_j = embeddings[j][valid_indices]
                concat_emb = torch.cat([emb_i, emb_j], dim=1)
                results['pairs'].append((concat_emb, valid_indices))

        # Triplets
        for c in [ci for ci in self.combo_indices if len(ci) == 3]:
            i, j, k = c
            valid_indices = (missingness_matrix[:, i] == 1) & (missingness_matrix[:, j] == 1) & (missingness_matrix[:, k] == 1)
            if valid_indices.any():
                emb_i = embeddings[i][valid_indices]
                emb_j = embeddings[j][valid_indices]
                emb_k = embeddings[k][valid_indices]
                concat_emb = torch.cat([emb_i, emb_j, emb_k], dim=1)
                results['triplets'].append((concat_emb, valid_indices))

        # Quadruplets
        for c in [ci for ci in self.combo_indices if len(ci) == 4]:
            valid_indices = torch.ones(batch_size, dtype=torch.bool, device=device)
            emb_list = []
            for idx_mod in c:
                valid_indices = valid_indices & (missingness_matrix[:, idx_mod] == 1)
            if valid_indices.any():
                for idx_mod in c:
                    emb_list.append(embeddings[idx_mod][valid_indices])
                concat_emb = torch.cat(emb_list, dim=1)
                results['quadruplets'].append((concat_emb, valid_indices))

        return results

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        batch_size = None
        features = []
        preds = []

        # Compute unimodal features and preds
        if 'EHR' in self.modalities:
            if x is not None:
                ehr_feats = self.ehr_model(x, seq_lengths)
                if len(ehr_feats.shape) > 2:
                    ehr_feats = ehr_feats.mean(dim=1)
                features.append(ehr_feats)
                batch_size = ehr_feats.size(0)
                ehr_pred = self.ehr_classifier(ehr_feats)
                if ehr_pred.dim() == 1:
                    ehr_pred = ehr_pred.unsqueeze(-1)
                preds.append(ehr_pred)
            else:
                raise ValueError("EHR data is required.")

        missingness_matrix = self.detect_missingness_batch(batch_size, cxr=img, dn=dn, rr=rr)

        if 'CXR' in self.modalities and img is not None:
            cxr_feats = self.cxr_model(img)
            cxr_feats = cxr_feats[:, 0, :]
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            if cxr_pred.dim() == 1:
                cxr_pred = cxr_pred.unsqueeze(-1)
            preds.append(cxr_pred)

        if 'DN' in self.modalities and dn is not None:
            dn_feats, _ = self.text_model(dn_notes=dn)
            dn_feats = dn_feats.mean(dim=1)
            features.append(dn_feats)
            dn_pred = self.dn_classifier(dn_feats)
            if dn_pred.dim() == 1:
                dn_pred = dn_pred.unsqueeze(-1)
            preds.append(dn_pred)

        if 'RR' in self.modalities and rr is not None:
            _, rr_feats = self.text_model(rr_notes=rr)
            rr_feats = rr_feats.mean(dim=1)
            features.append(rr_feats)
            rr_pred = self.rr_classifier(rr_feats)
            if rr_pred.dim() == 1:
                rr_pred = rr_pred.unsqueeze(-1)
            preds.append(rr_pred)

        combo_results = self.similarity_calculator(features, missingness_matrix)

        # Create input for hypernetwork
        hyper_input = torch.cat(features, dim=1)  # [batch_size, num_modalities * feats_dim]
        unimodal_preds = torch.cat(preds, dim=1)
        missingness_matrix_device = missingness_matrix.to(unimodal_preds.device)
        missingness_preds = self.missingness_classifier(missingness_matrix_device.float())
        hyper_input = torch.cat([hyper_input, missingness_matrix_device.float()], dim=1)

        # Generate weights
        final_weights, combo_weights = self.hypernetwork(hyper_input)

        # Gather all combos
        all_combos = combo_results['pairs'] + combo_results['triplets'] + combo_results['quadruplets']
        num_combos = len(all_combos)

        if num_combos == 0:
            # No combos available, fallback
            final_joint_pred = unimodal_preds[:, [0]]
            all_preds = torch.cat([unimodal_preds, final_joint_pred, missingness_preds], dim=1)
            extended_missingness_matrix = torch.cat(
                [missingness_matrix_device, 
                 torch.ones(batch_size, 1, device=all_preds.device),
                 torch.ones(batch_size, 1, device=all_preds.device)], dim=1
            )

            masked_preds = all_preds * extended_missingness_matrix
            normalized_weights = F.softmax(final_weights, dim=1)
            modality_weights = normalized_weights * extended_missingness_matrix
            modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)
            fused_preds_final = (masked_preds * modality_weights).sum(dim=1)
            return {'msma': fused_preds_final, 'unified': fused_preds_final}

        # Determine the maximum number of modalities in any combo
        feats_dim = features[0].size(1)
        max_combo_modality_count = max(len(c) for c in self.combo_indices) if self.combo_indices else 1
        combo_dim = max_combo_modality_count * feats_dim

        combo_embeddings = torch.zeros(batch_size, num_combos, combo_dim, device=unimodal_preds.device)
        combo_valid_mask = torch.zeros(batch_size, num_combos, dtype=torch.bool, device=unimodal_preds.device)

        for idx, (emb, valid_mask) in enumerate(all_combos):
            # emb is already concatenated
            current_dim = emb.size(1)
            combo_embeddings[valid_mask, idx, :current_dim] = emb
            combo_valid_mask[valid_mask, idx] = True

        # Get predictions for every combo using a single joint classifier
        # Reshape: [batch_size*num_combos, combo_dim]
        combo_embeddings_flat = combo_embeddings.view(-1, combo_dim)
        joint_preds_flat = self.msma_joint_classifier(combo_embeddings_flat)
        joint_preds = joint_preds_flat.view(batch_size, num_combos, -1)

        # Weight combos
        combo_logits = combo_weights[:, :num_combos]
        combo_logits = combo_logits.masked_fill(~combo_valid_mask, float('-inf'))

        valid_counts = combo_valid_mask.sum(dim=1)
        no_valid_combo_mask = (valid_counts == 0)
        has_valid_combo_mask = ~no_valid_combo_mask

        final_joint_pred = torch.zeros(batch_size, joint_preds.size(-1), device=joint_preds.device)

        if has_valid_combo_mask.any():
            combo_logits_valid = combo_logits[has_valid_combo_mask]
            combo_weights_valid = F.softmax(combo_logits_valid, dim=1).unsqueeze(-1)
            joint_preds_valid = joint_preds[has_valid_combo_mask]
            final_joint_pred_valid = (joint_preds_valid * combo_weights_valid).sum(dim=1)
            final_joint_pred[has_valid_combo_mask] = final_joint_pred_valid

        final_joint_pred[no_valid_combo_mask] = unimodal_preds[no_valid_combo_mask, [0]].unsqueeze(1)

        all_preds = torch.cat([unimodal_preds, final_joint_pred, missingness_preds], dim=1)
        extended_missingness_matrix = torch.cat(
            [missingness_matrix_device, 
             torch.ones(batch_size, 1, device=all_preds.device),
             torch.ones(batch_size, 1, device=all_preds.device)], 
            dim=1
        )

        masked_preds = all_preds * extended_missingness_matrix
        normalized_final_weights = F.softmax(final_weights, dim=1)
        modality_weights = normalized_final_weights * extended_missingness_matrix
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)

        fused_preds = (masked_preds * modality_weights).sum(dim=1)
        
        # fused_preds = final_joint_pred
        
        return {'msma': fused_preds, 'unified': fused_preds}
        
class CMSMAFusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(CMSMAFusion, self).__init__()
        self.args = args
        self.modalities = args.modalities.split("-")  # Modalities we are considering
        self.ehr_model = ehr_model if 'EHR' in self.modalities else None
        self.cxr_model = cxr_model if 'CXR' in self.modalities else None
        self.text_model = text_model if any(m in self.modalities for m in ['DN', 'RR']) else None
        
        # One output dimension assumed for simplicity. If multiclass, adjust accordingly.
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args) if 'EHR' in self.modalities else None
        self.ehr_confidence_predictor = ConfidencePredictor(self.ehr_model.full_feats_dim)
        if 'CXR' in self.modalities:
            self.cxr_classifier = Classifier(self.cxr_model.feats_dim, self.args)
            self.cxr_confidence_predictor = ConfidencePredictor(self.cxr_model.full_feats_dim)
            d_in = self.cxr_model.cxr_encoder.projection_layer.in_features
            d_out = self.cxr_model.cxr_encoder.projection_layer.out_features
            
            self.cxr_high_proj = nn.Linear(d_in, d_out)
            self.cxr_low_proj  = nn.Linear(d_in, d_out)
            
            # Initialize weights/bias from the original single projection_layer
            with torch.no_grad():
                self.cxr_high_proj.weight.copy_(self.cxr_model.cxr_encoder.projection_layer.weight)
                self.cxr_high_proj.bias.copy_(self.cxr_model.cxr_encoder.projection_layer.bias)
                self.cxr_low_proj.weight.copy_(self.cxr_model.cxr_encoder.projection_layer.weight)
                self.cxr_low_proj.bias.copy_(self.cxr_model.cxr_encoder.projection_layer.bias)
                
        if 'RR' in self.modalities:
            self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)
            self.rr_confidence_predictor = ConfidencePredictor(self.text_model.full_feats_dim_rr)
            
            d_in = self.text_model.bert.config.hidden_size
            d_out = self.text_model.feats_dim_rr
            
            self.rr_high_proj = nn.Linear(d_in, d_out)
            self.rr_low_proj  = nn.Linear(d_in, d_out)

            with torch.no_grad():
                self.rr_high_proj.weight.copy_(self.text_model.fc_rr.weight)
                self.rr_high_proj.bias.copy_(self.text_model.fc_rr.bias)
                self.rr_low_proj.weight.copy_(self.text_model.fc_rr.weight)
                self.rr_low_proj.bias.copy_(self.text_model.fc_rr.bias)
            
        if 'DN' in self.modalities:
            self.dn_classifier = Classifier(self.text_model.feats_dim_dn, self.args)
            self.dn_confidence_predictor = ConfidencePredictor(self.text_model.full_feats_dim_dn)
            
            d_in = self.text_model.bert.config.hidden_size
            d_out = self.text_model.feats_dim_dn
            
            self.dn_high_proj = nn.Linear(d_in, d_out)
            self.dn_low_proj  = nn.Linear(d_in, d_out)

            with torch.no_grad():
                self.dn_high_proj.weight.copy_(self.text_model.fc_dn.weight)
                self.dn_high_proj.bias.copy_(self.text_model.fc_dn.bias)
                self.dn_low_proj.weight.copy_(self.text_model.fc_dn.weight)
                self.dn_low_proj.bias.copy_(self.text_model.fc_dn.bias)
        
        if self.ehr_model:
            for param in self.ehr_model.parameters():
                param.requires_grad = False
        if self.cxr_model:
            for param in self.cxr_model.parameters():
                param.requires_grad = False
        if self.text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False

        # Missingness classifier: takes missingness vector [batch_size, num_modalities]
        self.missingness_classifier = Classifier(len(self.modalities), self.args)
        self.high_conf_classifier = Classifier(args.patch_output_dim, self.args)
        self.low_conf_classifier = Classifier(args.patch_output_dim, self.args)

        self.num_modalities = len(self.modalities)
        # Now we have unimodal preds, plus joint pred, plus missingness pred = num_modalities + 2
        self.weights = nn.Parameter(torch.ones(self.num_modalities + 1, args.num_classes))  # Dynamic classes
        

        
        if self.args.freeze:
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
        
    def aggregate_with_mean(self, token_list):
        """
        Returns the mean vector of all token embeddings in token_list.
        If token_list is empty, returns a zero vector of shape [1, patch_output_dim].
        """
        if not token_list:
            return torch.zeros(
                1,
                self.args.patch_output_dim,
                device=next(self.parameters()).device
            )
    
        # Stack into [num_tokens, feat_dim]
        tokens = torch.stack(token_list, dim=0)
    
        # Mean-pool across the tokens -> [1, feat_dim]
        return tokens.mean(dim=0, keepdim=True)
    
    def forward(self, x=None, seq_lengths=None, img=None, pairs=None, rr=None, dn=None):
        print("[Forward] Starting forward pass")
        batch_size = None
        features = []
        preds = []
        high_conf_batch = []
        low_conf_batch = []
        

        # Process each modality
        if 'EHR' in self.modalities:
            ehr_feats, full_ehr_feats = self.ehr_model(x, seq_lengths)
            print("[EHR] ehr_feats shape:", ehr_feats.shape)
            print("[EHR] full_ehr_feats shape:", full_ehr_feats.shape)
            
            if len(ehr_feats.shape) > 2:  # Only apply pooling if there are more than 2 dimensions
                ehr_feats = ehr_feats.mean(dim=1)
            features.append(ehr_feats)
            batch_size = ehr_feats.size(0)
            missingness_matrix = self.detect_missingness_batch(batch_size, cxr=img, dn=dn, rr=rr)
            high_conf_tokens = [[] for _ in range(batch_size)]
            low_conf_tokens = [[] for _ in range(batch_size)]
            print("[EHR] batch_size determined to be:", batch_size)
            #print("EHR feature shape:", ehr_feats.shape)
            ehr_pred = self.ehr_classifier(ehr_feats)
            print("[EHR] ehr_pred shape:", ehr_pred.shape)
            preds.append(ehr_pred)
            
            ehr_confidences = self.ehr_confidence_predictor(full_ehr_feats)  # [batch_size, seq_len]
            print("[EHR] ehr_confidences shape:", ehr_confidences.shape)
            for sample_idx in range(batch_size):
                for token_idx in range(full_ehr_feats.size(1)):  # Iterate over seq_len
                    if ehr_confidences[sample_idx, token_idx] >= self.args.confidence_threshold:
                        high_conf_tokens[sample_idx].append(full_ehr_feats[sample_idx, token_idx])
                    else:
                        low_conf_tokens[sample_idx].append(full_ehr_feats[sample_idx, token_idx])

        if 'CXR' in self.modalities:
            cxr_feats, full_cxr_feats = self.cxr_model(img)
            cxr_feats = cxr_feats[:, 0, :]
            print("[CXR] After slicing [:, 0, :], cxr_feats shape:", cxr_feats.shape)
            print("[CXR] full_cxr_feats shape:", full_cxr_feats.shape)
            
            features.append(cxr_feats)
            cxr_pred = self.cxr_classifier(cxr_feats)
            print("[CXR] cxr_pred shape:", cxr_pred.shape)
            preds.append(cxr_pred)
            
            cxr_confidences = self.cxr_confidence_predictor(full_cxr_feats)  # [batch_size, num_patches]
            print("[CXR] cxr_confidences shape:", cxr_confidences.shape)
            for i in range(batch_size):
                cxr_idx = self.modalities.index('CXR')
                # if present, add tokens
                if missingness_matrix[i, cxr_idx] == 1:
                    num_patches = full_cxr_feats.size(1)
                    for token_idx in range(num_patches):
                        if cxr_confidences[i, token_idx] >= self.args.confidence_threshold:
                            high_conf_tokens[i].append(self.cxr_high_proj(full_cxr_feats[i, token_idx]))
                        else:
                            low_conf_tokens[i].append(self.cxr_low_proj(full_cxr_feats[i, token_idx]))

        if 'DN' in self.modalities and dn is not None:
            dn_feats, full_dn_feats, _, _ = self.text_model(dn_notes=dn)
            dn_feats = dn_feats.mean(dim=1)
            print("[DN] dn_feats shape:", dn_feats.shape)
            print("[DN] full_dn_feats shape:", full_dn_feats.shape)
            
            features.append(dn_feats)
            dn_pred = self.dn_classifier(dn_feats)
            print("[DN] dn_pred shape:", dn_pred.shape)
            preds.append(dn_pred)
            dn_confidences = self.dn_confidence_predictor(full_dn_feats)  # [batch_size, seq_len]
            print("[DN] dn_confidences shape:", dn_confidences.shape)
            for i in range(batch_size):
                dn_idx = self.modalities.index('DN')
                if missingness_matrix[i, dn_idx] == 1:
                    seq_len_dn = full_dn_feats.size(1)
                    for token_idx in range(seq_len_dn):
                        if dn_confidences[i, token_idx] >= self.args.confidence_threshold:
                            high_conf_tokens[i].append(self.dn_high_proj(full_dn_feats[i, token_idx]))
                        else:
                            low_conf_tokens[i].append(self.dn_low_proj(full_dn_feats[i, token_idx]))

        if 'RR' in self.modalities and rr is not None:
            _, _, rr_feats, full_rr_feats = self.text_model(rr_notes=rr)
            rr_feats = rr_feats.mean(dim=1)
            print("[RR] rr_feats shape:", rr_feats.shape)
            print("[RR] full_rr_feats shape:", full_rr_feats.shape)
            
            features.append(rr_feats)
            rr_pred = self.rr_classifier(rr_feats)
            print("[RR] rr_pred shape:", rr_pred.shape)
            preds.append(rr_pred)
            rr_confidences = self.rr_confidence_predictor(full_rr_feats)  # [batch_size, seq_len]
            print("[RR] rr_confidences shape:", rr_confidences.shape)

            for i in range(batch_size):
                rr_idx = self.modalities.index('RR')
                if missingness_matrix[i, rr_idx] == 1:
                    seq_len_rr = full_rr_feats.size(1)
                    for token_idx in range(seq_len_rr):
                        if rr_confidences[i, token_idx] >= self.args.confidence_threshold:
                            high_conf_tokens[i].append(self.rr_high_proj(full_rr_feats[i, token_idx]))
                        else:
                            low_conf_tokens[i].append(self.rr_low_proj(full_rr_feats[i, token_idx]))

        # Print the number of tokens for each sample
        print("High confidence tokens count per sample:")
        for idx, tokens in enumerate(high_conf_tokens):
            print(f"  Sample {idx}: {len(tokens)} tokens")
        
        print("Low confidence tokens count per sample:")
        for idx, tokens in enumerate(low_conf_tokens):
            print(f"  Sample {idx}: {len(tokens)} tokens")
        
        # # Print the shape of each token in the high confidence list
        # print("Shapes of high confidence tokens per sample:")
        # for idx, tokens in enumerate(high_conf_tokens):
        #     print(f"  Sample {idx}:")
        #     for token_idx, token in enumerate(tokens):
        #         print(f"    Token {token_idx} shape: {token.size()}")  # Each token is a tensor
        
        # # Print the shape of each token in the low confidence list
        # print("Shapes of low confidence tokens per sample:")
        # for idx, tokens in enumerate(low_conf_tokens):
        #     print(f"  Sample {idx}:")
        #     for token_idx, token in enumerate(tokens):
        #         print(f"    Token {token_idx} shape: {token.size()}")
        
        # Aggregate tokens for the current sample
        high_conf_vectors = []
        low_conf_vectors  = []
        for i in range(batch_size):
            high_vec = self.aggregate_with_mean(high_conf_tokens[i])  # [1, feat_dim]
            low_vec  = self.aggregate_with_mean(low_conf_tokens[i])   # [1, feat_dim]
            high_conf_vectors.append(high_vec)
            low_conf_vectors.append(low_vec)
            
            print(f"  High confidence vector shape: {high_vec.size()}")
            print(f"  Low confidence vector shape: {low_vec.size()}")

        high_conf_batch = torch.cat(high_conf_vectors, dim=0)  # (B, feat_dim)
        low_conf_batch  = torch.cat(low_conf_vectors,  dim=0)  # (B, feat_dim)
        
        print("[Forward] high_conf_batch shape:", high_conf_batch.shape)
        print("[Forward] low_conf_batch  shape:", low_conf_batch.shape)
        
        missingness_matrix_device = missingness_matrix.to(preds[0].device)
        missingness_preds = self.missingness_classifier(missingness_matrix_device.float())
        
        print("[Forward] missingness_preds shape:", missingness_preds.shape)
        
        high_conf_pred = self.high_conf_classifier(high_conf_batch)
        low_conf_pred =  self.low_conf_classifier(low_conf_batch)
        preds = torch.cat(preds, dim=1)  # [batch_size, num_modalities]
        # Add joint prediction and missingness prediction
        preds = torch.cat([preds, missingness_preds], dim=1)
        # preds shape: [batch_size, num_modalities + 1]
        
        extended_missingness_matrix = torch.cat(
            [missingness_matrix_device, 
             torch.ones(batch_size, 1, device=preds[0].device)], # missingness pred always present
            dim=1
        )
        print("[Forward] missingness matrix shape:", extended_missingness_matrix.shape)
        extended_missingness_matrix = extended_missingness_matrix.unsqueeze(-1).repeat(1, 1, self.args.num_classes)
        print("[Forward] missingness matrix shape:", extended_missingness_matrix.shape)
        extended_missingness_matrix = extended_missingness_matrix.view(batch_size, -1)
        print("[Forward] missingness matrix shape:", extended_missingness_matrix.shape)
        
        normalized_weights = F.softmax(self.weights, dim=0)
        
        masked_preds = preds * extended_missingness_matrix
        
        # Expand weights to match batch size
        normalized_weights = normalized_weights.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, modalities + 2, num_classes]
        
        # Reshape weights to flatten class dimensions
        normalized_weights = normalized_weights.view(batch_size, -1)  # [batch_size, modalities * num_classes]
        
        modality_weights = normalized_weights * extended_missingness_matrix
        modality_weights = modality_weights / (modality_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        masked_preds = masked_preds.view(batch_size, -1, self.args.num_classes)
        modality_weights = modality_weights.view(batch_size, -1, self.args.num_classes)
        fused_preds_final = torch.sum(masked_preds * modality_weights, dim=1)
        
        print("[Forward] high conf shape:", high_conf_pred.shape)
        print("[Forward] low conf shape:", low_conf_pred.shape)
        print("[Forward] late preds shape:", fused_preds_final.shape)
        
        return {'high_conf': high_conf_pred, 'low': low_conf_pred, 'late': fused_preds_final }



class MultiheadAttentionWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=False):
        super(MultiheadAttentionWrapper, self).__init__()
        self.batch_first = batch_first
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value, **kwargs):
        if self.batch_first:
            # Convert to (seq_len, batch_size, embed_dim)
            query, key, value = [x.permute(1, 0, 2) for x in (query, key, value)]
        output, weights = self.attention(query, key, value, **kwargs)
        if self.batch_first:
            # Convert back to (batch_size, seq_len, embed_dim)
            output = output.permute(1, 0, 2)
        return output, weights


