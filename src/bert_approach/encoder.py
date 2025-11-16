import torch
import torch.nn as nn
from math import sqrt

class Layer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 num_heads: int = 1, 
                 lr: float | int = 0.001,
                 dropout: float | int = 0.1):
        """Constructor"""
        super().__init__()
        assert isinstance(d_model, int) and d_model % 2 == 0, "d_model must be a positive even number"
        assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert isinstance(d_ff, int) and d_ff % 2 == 0, "d_ff must be a positive even number"
        assert isinstance(lr, (float, int)) and 0 <= lr <= 1, "lr must be between 0 and 1"
        assert isinstance(dropout, (float, int)) and 0 <= dropout <= 1, "dropout must be between 0 and 1"
        
        # Define components of a layer
        self.attention = MultiHeadAttention(d_model, num_heads, lr)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # or nn.ReLU()
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm_1 = nn.LayerNorm(d_model) 
        self.layer_norm_2 = nn.LayerNorm(d_model)  
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
        assert isinstance(input, torch.Tensor) and input.dtype == torch.float32, "input must be a torch.Tensor object and the datatype should be torch.float32"
        
        # Apply layer norm
        norm_out = self.layer_norm_1(input)
        # Apply Multi Head Attention
        attention_out = self.attention((norm_out, norm_out, norm_out), mask)
        attention_out = self.dropout(attention_out)
        # Residual connection
        res_sum = attention_out + input

        # Apply second layer norm
        norm_out = self.layer_norm_2(res_sum)
        # Apply MLP
        mlp_out = self.mlp(norm_out)
        mlp_out = self.dropout(mlp_out)
        # Residual connection
        output = mlp_out + res_sum

        return output
    
class Encoder(nn.Module):
    def __init__(self, 
                 classes : int,
                 d_model: int,
                 d_ff: int, 
                 num_heads: int = 1, 
                 N: int = 1, 
                 lr: float | int = 0.001,
                 dropout: float | int = 0.1):
        """Constructor"""
        super().__init__()
        assert isinstance(classes, int) and classes > 0, "classes must be a non-zero positive integer"
        assert isinstance(d_model, int) and d_model % 2 == 0, "d_model must be a positive even number"
        assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert isinstance(d_ff, int) and d_ff % 2 == 0, "d_ff must be a positive even number"
        assert isinstance(N, int) and N > 0, "N must be a positive integer"
        assert isinstance(lr, (float, int)) and 0 <= lr <= 1, "lr must be between 0 and 1"
        assert isinstance(dropout, (float, int)) and 0 <= dropout <= 1, "dropout must be between 0 and 1"
        
        self.dropout = nn.Dropout(dropout)
        # Define encoder layers
        self.layers = nn.ModuleList([
            Layer(d_model, d_ff, num_heads, lr, dropout) for _ in range(N)
        ])

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
        """
        Encoder process: Iterative flow through each layer
        """
        assert isinstance(input, torch.Tensor) and input.dtype == torch.float32, "input must be a torch.Tensor object and the datatype should be torch.float32"
    
        layer_in = self.dropout(input)

        for layer in self.layers:
            layer_out = layer(layer_in, mask)
            layer_in = layer_out
        
        output = layer_out
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model : int, 
                 num_heads : int,
                 lr: float | int = 0.001):
        """Constructor"""
        super().__init__()
        assert isinstance(d_model, int) and d_model % 2 == 0, "d_model must be a positive even number"
        assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert isinstance(lr, (float, int)) and 0 <= lr <= 1, "lr must be between 0 and 1"
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sqrt_d_k = sqrt(self.d_k)

        # Linear projections for Q, K, V 
        self.w_q = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False)
        self.w_k = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False)
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False)
    
    def scaled_dot_product_attention(self, Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, mask : torch.Tensor = None):
        """
        Perform Scaled Dot-Product Attention
        Q, K, V shape: [batch_size, num_heads, seq_len, d_k]
        
        Output Shape: [batch_size, num_heads, seq_len, d_k]
        """
        assert isinstance(Q, torch.Tensor) and Q.dtype == torch.float32, "Q must be a torch.Tensor object and the datatype should be torch.float32"
        assert isinstance(K, torch.Tensor) and K.dtype == torch.float32, "K must be a torch.Tensor object and the datatype should be torch.float32"
        assert isinstance(V, torch.Tensor) and V.dtype == torch.float32, "V must be a torch.Tensor object and the datatype should be torch.float32"
        assert ((isinstance(mask, torch.Tensor) and mask.dtype == torch.bool) or mask == None), "mask must be a torch.Tensor object and the datatype should be torch.bool"
        
        self.V = V
        self.K = K
        self.Q = Q
        
        # Calculate Attention Scores
        # Step 1 : Q x K.T x d_k ^ -1/2
        scores = (torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d_k)
        # Step 2 : Apply Mask(Optional)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, N]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Step 3 : Softmax activation function
        # For each query row apply softmax for the row to sum to 1
        attention_weights = torch.softmax(scores, dim=-1)
        # Step 4 : Multiply with V matrix
        attention_out = torch.matmul(attention_weights, V)
        
        return attention_out
        
    def forward(self, QKV : tuple, mask=None):
        """Perform Multi-Head Attention"""
        assert isinstance(QKV[0], torch.Tensor) and QKV[0].dtype == torch.float32, "query must be a torch.Tensor object and the datatype should be torch.float32"
        assert isinstance(QKV[1], torch.Tensor) and QKV[1].dtype == torch.float32, "key must be a torch.Tensor object and the datatype should be torch.float32"
        assert isinstance(QKV[2], torch.Tensor) and QKV[2].dtype == torch.float32, "value must be a torch.Tensor object and the datatype should be torch.float32"
        assert ((isinstance(mask, torch.Tensor) and mask.dtype == torch.bool) or mask == None), "mask must be a torch.Tensor object and the datatype should be torch.bool"
        self.query = QKV[0]
        self.key = QKV[1]
        self.value = QKV[2]
        
        batch_size, tgt_len, d_model = self.query.shape
        src_len = self.key.shape[1]
        
        # 1. Linear projections to get Q, K, V
        Q = self.w_q(self.query)  # [batch_size, tgt_len, d_model]
        K = self.w_k(self.key)    # [batch_size, src_len, d_model] 
        V = self.w_v(self.value)  # [batch_size, src_len, d_model]

        # 2. Reshape for multi-head attention
        # Split d_model into num_heads * d_k
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: [num_heads, seq_len, d_k]
      
        # 3. Apply scaled dot-product attention
        attention_out = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Concatenate heads and project back to d_model
        attention_out = attention_out.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, d_model
        )
        
        # 5. Final linear projection
        output = self.w_o(attention_out)
        
        return output