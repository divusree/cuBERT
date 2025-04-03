import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass 
torch.set_printoptions(profile="full")
from seqeval.metrics import classification_report, f1_score

@dataclass
class ModelArgs:
    dim: int = 768
    n_layers: int = 12
    norm_eps: float = 1e-5
    vocab_size: int = 1000 # set on tokenizer load
    device: str = None
    n_heads: int = 12
    max_batch_size: int = 16
    max_seq_len: int = 512
    ffn_hidden_dim: int = 4 * dim
    dtype: torch.dtype = torch.float32
    dropout_rate: float = 0.25
def precompute_pos_encoding(args):
    pos_encoding = torch.zeros((args.max_seq_len, args.dim), dtype = args.dtype, device = args.device)
    even_idx = torch.arange(0, args.dim, 2, device = args.device)
    odd_idx = torch.arange(1, args.dim, 2, device = args.device)
    pos_idx = torch.arange(0, args.max_seq_len, device = args.device)[:,None]
    pos_encoding[:, ::2] = torch.sin(pos_idx/(1e4**(even_idx/args.dim)))
    pos_encoding[:, 1::2] = torch.cos(pos_idx/(1e4**((odd_idx-1)/args.dim)))

    return pos_encoding    
class SelfAttention(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()    
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.proj_dim = self.n_heads * self.head_dim 

        self.wq = nn.Linear(args.dim, self.proj_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.proj_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.proj_dim, bias = False)
        self.wo = nn.Linear(self.proj_dim, args.dim, bias = False)

        self.softmax = torch.nn.Softmax(dim = -1)
        self.qk_scale = 1 / math.sqrt(self.head_dim)

        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, h: torch.tensor, attn_mask = None):
        # h.shape = [batch_size, seq_len, dim]
        batch_size, seq_len, dim = h.shape
        # hq.shape = [batch_size, seq_len, n_heads * head_dim]
        hq = self.wq(h)
        hk = self.wk(h)
        hv = self.wv(h)

        # hproj.shape = [batch_size, n_heads, seq_len, head_dim]
        hq = hq.view(batch_size, self.n_heads, seq_len, self.head_dim)
        hk = hk.view(batch_size, self.n_heads, seq_len, self.head_dim)
        hv = hv.view(batch_size, self.n_heads, seq_len, self.head_dim)

        # hk.transpose.shape = [batch_size, n_heads, head_dim, seq_len]
        attn_temp = hq @ hk.transpose(-1, -2) # [batch_size, n_heads, seq_len, seq_len]
        attn_temp = attn_temp * self.qk_scale

        if attn_mask is not None:
            attn_temp = attn_temp.masked_fill(attn_mask == 0, float('-inf'))

        # matmul: [batch_size, n_heads, seq_len, seq_len] @  [batch_size, n_heads, seq_len, head_dim]
        # attn_scores = [batch_size, n_heads, seq_len, head_dim]
        attn_weights  = self.softmax(attn_temp.float()) @ hv 
        
        attn_weights = self.dropout(attn_weights)
        
        # attn_scores = [batch_size, seq_len, head_dim * n_heads = dim]
        attn_weights  = attn_weights.view(batch_size, seq_len, self.head_dim * self.n_heads)
        attn_weights  = attn_weights.contiguous() #.type_as(hq)

        # attn_scores.shape = [batch_size, seq_len, dim]
        attn_scores = self.wo(attn_weights)
        return attn_scores

class FeedForward(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.dim, args.ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.ffn_hidden_dim, args.dim),
            nn.Dropout(args.dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
class LayerNorm(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()    
        self.eps = args.norm_eps
        self.gamma = nn.Parameter(torch.ones(args.dim))
        self.beta = nn.Parameter(torch.zeros(args.dim))
    def forward(self, h: torch.tensor):
        mean = h.mean(axis = -1,keepdim = True)
        var = h.var(dim=-1, unbiased=False, keepdim=True)
        rsqrt = torch.rsqrt(var + self.eps)
        norm = self.gamma * (h - mean) * rsqrt + self.beta
        return norm.type_as(h) 
    
class EncoderBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()    
    
        self.attention = SelfAttention(args)
        self.attn_norm = LayerNorm(args)
        self.feed_forward = FeedForward(args)
        self.ffn_norm = LayerNorm(args)
        self.dropout = nn.Dropout(args.dropout_rate) 
    def forward(self, h: torch.tensor, attn_mask : torch.tensor):
        # h.shape = [batch_size, seq_len, dim]
        # pre layer norm
        h = h + self.dropout(self.attention(self.attn_norm(h), attn_mask))
        h = h + self.dropout(self.feed_forward(self.ffn_norm(h)))
        return h


class BERTBase(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert args.dim % args.n_heads == 0

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim, device = args.device, padding_idx=0)
        self.pos_encoding = precompute_pos_encoding(args)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.dropout = nn.Dropout(args.dropout_rate)


    def forward(self, tokens: torch.tensor, attn_mask : torch.tensor):
        batch_size, seq_len = tokens.shape
        # get embeddings - [batch_size, seq_len, dim]
        h = self.tok_embeddings(tokens)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        attn_mask = attn_mask.expand(-1, self.args.n_heads, -1, seq_len)  
        # if attn_mask is not None:
        #     h = h * attn_mask.unsqueeze(-1)  # Zero out padding embeddings
        
        # pos encoding shape - [seq_len, dim]
        pos_encodings = self.pos_encoding[:seq_len] 

        h = h + pos_encodings.unsqueeze(0) # figure this out
        # perform forward through all layers
        for layer in self.layers:
            h = layer(h, attn_mask )
            # if attn_mask is not None:
            #     h = h * attn_mask.unsqueeze(-1)  # Re-zero padding after each layer
                    
        h = self.dropout(h)

        return h # i will apply softmax outside
    
class BERTForNER(BERTBase):  # Inherit from your BERTBase
    def __init__(self, args: ModelArgs, num_ner_labels: int):
        super().__init__(args)
        self.ner_head = nn.Linear(args.dim, num_ner_labels) 
        
    def forward(self, tokens: torch.Tensor, attn_mask : torch.Tensor):
      
        h = super().forward(tokens, attn_mask = attn_mask )  # Shape: [batch_size, seq_len, dim]
        
        logits = self.ner_head(h)  # Shape: [batch_size, seq_len, num_ner_labels]
        # if attn_mask is not None:
        #     logits = logits * attn_mask.unsqueeze(-1)  # Force zero outputs for padding
                    
        return logits    
 
    
def compute_loss(criterion, outputs, labels, mask):
    # Reshape and mask
    logits = outputs.view(-1, outputs.shape[-1])
    labels = labels.view(-1)
    mask = mask.view(-1).bool()
    
    # Apply mask
    logits = logits[mask]
    labels = labels[mask]
    
    return criterion(logits, labels)    


def evaluate(model, validation_loader, criterion, label_list, device, writer, global_step):
    """
    Validate the NER model and return metrics
    
    Args:
        model: Your NER model
        validation_loader: DataLoader for validation set
        criterion: Loss function (should ignore_index=-100)
        label_list: List of label names (including 'O' for non-entity)
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        Dictionary containing:
        - avg_loss: Average validation loss
        - report: Classification report (precision/recall/F1)
        - padding_check: Whether padding outputs are zero
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    padding_checks = []
    
    with torch.no_grad():
        for batch in validation_loader:
            # Move data to device
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(inputs, attn_mask=masks)
            
            # Calculate loss (only on non-padding tokens)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            total_loss += loss.item()
            
            # Check padding outputs
            padding_mask = (masks == 0)
            padding_outputs = outputs[padding_mask]
            padding_checks.append(
                torch.allclose(padding_outputs, 
                             torch.zeros_like(padding_outputs), 
                             atol=1e-6))
            
            # Convert to predictions (excluding padding)
            preds = torch.argmax(outputs, dim=-1)
            active_mask = (labels != -100)  # Ignore padding
            
            # Collect predictions and labels for seqeval
            for i in range(labels.shape[0]):
                active_preds = preds[i][active_mask[i]].cpu().numpy()
                active_labels = labels[i][active_mask[i]].cpu().numpy()
                
                # Convert to label names
                pred_labels = [label_list[p] for p in active_preds]
                true_labels = [label_list[l] for l in active_labels]
                
                all_predictions.append(pred_labels)
                all_labels.append(true_labels)
    
    # Calculate metrics
    avg_loss = total_loss / len(validation_loader)
    report_dict = classification_report(all_labels, all_predictions, output_dict = True)
    padding_valid = all(padding_checks)
    
    for key in report_dict:
        if isinstance(report_dict[key], dict):  # Per-class metrics (e.g., 'PER', 'LOC')
            writer.add_scalar(f"F1/{key}", report_dict[key]["f1-score"], global_step)
    
    writer.add_scalar("F1/micro_avg", report_dict["micro avg"]["f1-score"], global_step)
    writer.add_scalar("F1/macro_avg", report_dict["macro avg"]["f1-score"], global_step)
    writer.add_scalar("Loss/val", avg_loss, global_step=global_step)
    print(f"Validation Loss: {avg_loss:.4f}")

    return {
        'avg_loss': avg_loss,
        'report': report_dict,
        'padding_check': padding_valid,
        'predictions': all_predictions,
        'labels': all_labels,
        "f1-score":  f1_score(all_labels, all_predictions)
    }