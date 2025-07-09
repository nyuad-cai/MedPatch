import torch
from torch import nn
from transformers import LongformerModel, LongformerTokenizer

class DischargeNotesEncoder(nn.Module):
    def __init__(self,device, pretrained_model_name='allenai/longformer-base-4096', output_dim=384):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(pretrained_model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.longformer.config.hidden_size, output_dim)
        self.output_dim = output_dim
        self.device=device

    def forward(self, notes):
        # Tokenize input notes
        encoded_input = self.tokenizer(notes, padding=True, truncation=True, max_length=4096, return_tensors='pt')
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)

        # Get Longformer embeddings
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output

        # Project to desired output dimension
        v_dn = self.fc(outputs.last_hidden_state)
        cls = self.fc(cls_output)
        return v_dn, cls

