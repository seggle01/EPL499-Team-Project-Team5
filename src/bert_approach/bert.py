from encoder import Encoder
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Positional Encoding to add position information to embeddings.
        
        Args:
            d_model: Dimension of embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float = 0.1):
        """
        BERT Embeddings: Token Embeddings + Positional Encoding
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.d_model = d_model
    
    def forward(self, input_ids: torch.Tensor):
        """
        Convert token indices to embeddings with positional encoding.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, d_model]
        """
        # Get token embeddings and scale by sqrt(d_model)
        embeddings = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        return embeddings

class BERT(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 max_len: int,
                 d_model: int,
                 d_ff: int, 
                 num_heads: int = 1, 
                 N: int = 1, 
                 lr: float | int = 0.001,
                 dropout: float | int = 0.1,
                 num_classes: int = 2):
        """
        BERT Model for Binary Classification
        
        Args:
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            d_model: Dimension of model
            d_ff: Dimension of feedforward network
            num_heads: Number of attention heads
            N: Number of encoder layers
            lr: Learning rate (for compatibility)
            dropout: Dropout probability
            num_classes: Number of output classes (2 for binary)
        """
        super().__init__()
        
        # Embedding layer
        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_len, dropout)
        
        # Encoder layers
        self.encoder = Encoder(
            classes=d_model,  # Encoder outputs d_model dimensions
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            N=N,
            lr=lr,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self.d_model = d_model
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass through BERT model.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Class predictions [batch_size, num_classes]
        """
        # Convert indices to embeddings
        embeddings = self.embeddings(input_ids)  # [batch_size, seq_len, d_model]
        
        # Convert attention mask to boolean if provided
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = None
        
        # Pass through encoder
        encoder_output = self.encoder(embeddings, mask)  # [batch_size, seq_len, d_model]
        
        # Use [CLS] token (first token) for classification
        # Or use mean pooling over sequence
        cls_output = encoder_output[:, 0, :]  # [batch_size, d_model]
        
        # Classification head
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        
        return logits