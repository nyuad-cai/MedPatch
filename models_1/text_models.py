import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast

class Text_encoder(nn.Module):
    def __init__(self, args, device):
        """
        Initializes the Text_encoder.

        Args:
        - args (Namespace): Contains configurations like `bert_model_name`, `output_dim_dn`, `output_dim_rr`, and `modalities`.
        - device (torch.device): Device to run the model on.
        """
        super().__init__()
        self.device = device
        self.modalities = getattr(args, 'modalities', []).split('-')
        self.args=args

        # Get BERT model name and output dimensions from args
        pretrained_model_name = getattr(args, 'bert_model_name', 'emilyalsentzer/Bio_ClinicalBERT')
        output_dim_dn = getattr(args, 'output_dim_dn', 512)
        output_dim_rr = getattr(args, 'output_dim_rr', 512)

        # Pretrained BERT model and tokenizer
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

        self.feats_dim_dn = output_dim_dn
        self.feats_dim_rr = output_dim_rr
        
        self.full_feats_dim_dn = self.bert.config.hidden_size  # Size of full BERT embeddings
        self.full_feats_dim_rr = self.bert.config.hidden_size

        # Freeze the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False

        # Separate FC layers for DN and RR modalities
        
        if 'DN' in self.modalities:
            self.fc_dn = nn.Linear(self.bert.config.hidden_size, output_dim_dn)
        if 'RR' in self.modalities:
            self.fc_rr = nn.Linear(self.bert.config.hidden_size, output_dim_rr)

    def process_notes(self, notes, fc_layer, max_length=512, stride=0):
        """
        Processes notes by averaging chunk embeddings for each note.

        Args:
        - notes (list[str]): List of notes to process.
        - fc_layer (nn.Linear): Fully connected layer to project embeddings.
        - max_length (int): Maximum number of tokens per chunk (and final sequence length).
        - stride (int): Number of tokens to overlap between chunks.

        Returns:
        - torch.Tensor: Processed embeddings with shape [batch_size, seq_len=512, output_dim].
        """
        # Tokenize notes with automatic handling of overflowing tokens
        encoded_inputs = self.tokenizer(
            notes,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors='pt'
        )

        # Map each chunk back to its original note
        num_chunks = len(encoded_inputs['input_ids'])
        sample_mapping = encoded_inputs['overflow_to_sample_mapping']  # List of indices mapping chunks to original notes

        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)

        # Get BERT embeddings for all chunks
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            chunk_embeddings = outputs.last_hidden_state  # Shape: [num_chunks, 512, hidden_dim]

        # Initialize a list to hold averaged embeddings for each note
        batch_size = len(notes)
        note_embeddings = [[] for _ in range(batch_size)]

        # Aggregate chunk embeddings back to their respective notes
        for idx, mapping in enumerate(sample_mapping):
            note_embeddings[mapping].append(chunk_embeddings[idx])

        # For each note, calculate the average embedding across chunks
        aggregated_embeddings = []
        for embeddings in note_embeddings:
            if embeddings:
                # Stack and mean pool the embeddings across chunks
                embeddings_tensor = torch.stack(embeddings)  # Shape: [num_chunks, 512, hidden_dim]
                aggregated_embedding = embeddings_tensor.mean(dim=0)  # Shape: [512, hidden_dim]
            else:
                # Handle empty notes
                aggregated_embedding = torch.zeros((max_length, self.bert.config.hidden_size), device=self.device)
            aggregated_embeddings.append(aggregated_embedding)
        
        # Stack aggregated embeddings for the batch
        batch_embeddings = torch.stack(aggregated_embeddings)  # Shape: [batch_size, 512, hidden_dim]

        # Pass through the FC layer
        projected_embeddings = fc_layer(batch_embeddings)  # Shape: [batch_size, 512, output_dim]
        
        if 'c-' in self.args.fusion_type:
            return projected_embeddings, batch_embeddings  # Shape: [batch_size, 512, hidden_dim]
        elif 'healnet' in self.args.fusion_type:
            return projected_embeddings, batch_embeddings  # Shape: [batch_size, 512, hidden_dim]
        else:
            return projected_embeddings

    def forward(self, dn_notes=None, rr_notes=None):
        """
        Forward pass of the Text_encoder.

        Args:
        - dn_notes (list[str] or None): List of notes for DN modality.
        - rr_notes (list[str] or None): List of notes for RR modality.

        Returns:
        - dn_output (torch.Tensor or None): Output for DN modality.
        - rr_output (torch.Tensor or None): Output for RR modality.
        """
        dn_output = None
        rr_output = None
        dn_output_full = None
        rr_output_full = None

        if dn_notes is not None and 'DN' in self.modalities:
            if 'c-' in self.args.fusion_type:
                dn_output, dn_output_full = self.process_notes(dn_notes, self.fc_dn)
            elif 'healnet' in self.args.fusion_type:
                dn_output, dn_output_full = self.process_notes(dn_notes, self.fc_dn)
            else:
                dn_output = self.process_notes(dn_notes, self.fc_dn)

        if rr_notes is not None and 'RR' in self.modalities:
            if 'c-' in self.args.fusion_type:
                rr_output, rr_output_full = self.process_notes(rr_notes, self.fc_rr)
            elif 'healnet' in self.args.fusion_type:
                rr_output, rr_output_full = self.process_notes(rr_notes, self.fc_rr)
            else:
                rr_output = self.process_notes(rr_notes, self.fc_rr)

        if 'c-' in self.args.fusion_type:
            return dn_output, dn_output_full, rr_output, rr_output_full
        elif 'healnet' in self.args.fusion_type:
            return dn_output, dn_output_full, rr_output, rr_output_full
        else:
            return dn_output, rr_output
