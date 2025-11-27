"""
Vision Transformer for Protein Secondary Structure Prediction

This module implements a Vision Transformer architecture for predicting
protein secondary structures from ProtBERT embeddings. The model uses
overlapping patch extraction, learnable positional embeddings, and
multi-head self-attention to capture spatial relationships in protein
embedding images.

Author: Stefanos Englezou  
Last Modified: 18/10/2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log


class Layer(nn.Module):
    """
    Transformer encoder layer with self-attention and MLP.
    
    This class implements a single transformer encoder layer consisting
    of multi-head self-attention followed by a feed-forward MLP network.
    Layer normalization is applied before each sub-layer (pre-norm architecture),
    and residual connections are used after each sub-layer. Dropout is applied
    after both attention and MLP operations for regularization.
    
    The architecture follows: 
    x -> LayerNorm -> MultiHeadAttention -> Dropout -> Add(x) -> 
    LayerNorm -> MLP -> Dropout -> Add -> output
    
    Attributes
    ----------
    attention : MultiHeadAttention
        Multi-head self-attention mechanism.
    mlp : nn.Sequential
        Feed-forward network with two linear layers and ReLU activation.
    layer_norm_1 : nn.LayerNorm
        Layer normalization applied before attention.
    layer_norm_2 : nn.LayerNorm
        Layer normalization applied before MLP.
    dropout : nn.Dropout
        Dropout layer for regularization.
    """
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 num_heads: int = 1,
                 dropout: float | int = 0.1):
        """
        Initialize transformer encoder layer.
        
        This constructor creates a transformer encoder layer with
        the specified model dimension, feed-forward dimension, number
        of attention heads, and dropout rate. All components are
        initialized and ready for forward pass.
        
        Parameters
        ----------
        d_model : int
            Dimension of the model embeddings. Must be positive, even,
            and divisible by num_heads.
        d_ff : int
            Dimension of the feed-forward network hidden layer. Must be
            a positive even number.
        num_heads : int, default=1
            Number of attention heads in multi-head attention. Must be
            a positive integer and d_model must be divisible by num_heads.
        dropout : float or int, default=0.1
            Dropout probability for regularization. Must be between 0 and 1.
        """
        super().__init__()
        assert isinstance(d_model, int) and d_model > 0 and d_model % 2 == 0, "d_model must be a positive even number"
        assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert isinstance(d_ff, int) and d_model > 0 and d_ff % 2 == 0, "d_ff must be a positive even number"
        assert isinstance(dropout, (float, int)) and 0 <= dropout <= 1, "dropout must be between 0 and 1"
        
        # Define components of a layer
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.mlp =  self.mlp_head = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
        self.layer_norm_1 = nn.LayerNorm(d_model) 
        self.layer_norm_2 = nn.LayerNorm(d_model)  
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass through transformer encoder layer.
        
        This method applies the full transformer encoder layer operations:
        layer normalization, multi-head attention with residual connection,
        followed by layer normalization, MLP with residual connection.
        Dropout is applied after both attention and MLP operations.
        
        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model) with
            dtype torch.float32.
        mask : torch.Tensor, optional
            Attention mask of shape (batch_size, seq_len) or 
            (batch_size, 1, 1, seq_len) with dtype torch.bool.
            True/1 indicates positions to attend, False/0 to mask.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model)
            after applying attention and MLP with residual connections.
        """
        assert isinstance(input, torch.Tensor) and input.dtype == torch.float32, "input must be a torch.Tensor object and the datatype should be torch.float32"
        assert ((isinstance(mask, torch.Tensor) and mask.dtype == torch.bool) or mask == None), "mask must be a torch.Tensor object and the datatype should be torch.bool"
        
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
    
class VisionTransformer(nn.Module):
    def __init__(self,
                 width: int = 32,
                 height: int = 32,
                 patch_size: int = 8,
                 patch_stride: int = 8,
                 embed_dim: int = 256,
                 d_ff: int = 1024, 
                 num_heads: int = 8,
                 layers: int = 1,
                 num_classes: int = 3,
                 dropout: float | int = 0.1,
                 pooling_patches: list = None,
                 mode: str = "pre-training"): 
        super().__init__()
        
        # Calculate number of patches
        hor_patches = ((width - patch_size) // patch_stride) + 1 
        ver_patches = ((height - patch_size) // patch_stride) + 1 
        
        self.patches = ver_patches * hor_patches
        self.pooling_patches = pooling_patches
        
        # Initialize CNN with configurable in_channels
        self.patch_embedding = OverlappingCNN(
            patch_size=patch_size, 
            stride=patch_stride,
            embed_dim=embed_dim
        )
        
        # Add learnable Positional Embeddings
        self.pos_e = PositionalEncoding(embed_dim, self.patches, dropout)

        # Define encoder layers
        self.layers = nn.ModuleList([
            Layer(embed_dim, d_ff, num_heads, dropout) for _ in range(layers)
        ])

        # Define MLP head based on the mode
        if mode == "pre-training":
            # Two layers MLP
            self.mlp_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_classes)
            )
        elif mode == "fine-tuning":
            # One layer MLP
            self.mlp_head = nn.Linear(embed_dim, num_classes)
        else:
            raise Exception("Unknown mode choosen for the model.")
        
    def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass through Vision Transformer.
        
        This method implements the complete forward pass: patch extraction,
        positional embedding addition, transformer encoder layers, patch
        selection/pooling, and final classification. The model processes
        input images and outputs class logits for each sample in the batch.
        
        The processing flow is:
        1. Extract and project overlapping patches
        2. Add learnable positional embeddings
        3. Process through transformer encoder layers
        4. Select specified patches or perform average pooling
        5. Apply MLP head for final classification
        
        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
            with dtype torch.float32. Typically channels=1 for grayscale
            protein embedding images.
        mask : torch.Tensor, optional
            Attention mask for transformer layers. Shape and dtype depend
            on the attention implementation.
        
        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes) representing
            predicted class scores for each sample.
        """
        assert isinstance(input, torch.Tensor) and input.dtype == torch.float32, "input must be a torch.Tensor object and the datatype should be torch.float32"
        assert input.dim() == 4, "input tensor must be of shape: [batch_size, channels, width, height]"
        assert ((isinstance(mask, torch.Tensor) and mask.dtype == torch.bool) or mask == None), "mask must be a torch.Tensor object and the datatype should be torch.bool"

        # Create patches and apply projections
        patched_embed = self.patch_embedding(input)

        # Apply positional encoding
        x = self.pos_e(patched_embed)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final MLP head 
        if self.pooling_patches is not None:
            center_patches = x[:,self.pooling_patches,:].mean(dim=1)
        else:
            # Avg Pooling
            center_patches = x.mean(dim=1)
            
        output = self.mlp_head(center_patches)
        
        return output

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This class implements the multi-head attention mechanism from
    "Attention is All You Need" (Vaswani et al., 2017). The input
    is projected into query, key, and value representations, which
    are then split into multiple heads. Scaled dot-product attention
    is computed in parallel for each head, and the results are
    concatenated and projected to produce the final output.
    
    The attention mechanism allows the model to focus on different
    positions and representation subspaces simultaneously, capturing
    complex relationships in the input data.
    
    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    d_k : int
        Dimension of each attention head (d_model / num_heads).
    sqrt_d_k : float
        Square root of d_k used for scaling attention scores.
    w_q : nn.Linear
        Linear projection for query.
    w_k : nn.Linear
        Linear projection for key.
    w_v : nn.Linear
        Linear projection for value.
    w_o : nn.Linear
        Output projection after concatenating heads.
    """
    def __init__(self, 
                 d_model : int, 
                 num_heads : int):
        """
        Initialize multi-head attention module.
        
        This constructor creates the linear projection layers for
        query, key, value, and output. The model dimension is split
        across multiple heads, with each head having dimension d_k.
        All projections are initialized without bias terms.
        
        Parameters
        ----------
        d_model : int
            Dimension of the model embeddings. Must be positive, even,
            and divisible by num_heads.
        num_heads : int
            Number of attention heads. Must be a positive integer and
            d_model must be divisible by num_heads to ensure each head
            has integer dimension.
        """
        super().__init__()
        assert isinstance(d_model, int) and d_model % 2 == 0, "d_model must be a positive even number"
        assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sqrt_d_k = sqrt(self.d_k)
        
        # self.temperature = nn.Parameter(torch.ones(1) * 10.0)  # Learnable temperature

        # Linear projections for Q, K, V 
        self.w_q = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False)
        self.w_k = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False)
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, dtype=torch.float32, bias=False)
    
    def scaled_dot_product_attention(self, Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, mask : torch.Tensor = None):
        """
        Perform scaled dot-product attention.
        
        This method computes attention weights by taking the dot product
        of queries and keys, scaling by the square root of the key dimension,
        applying optional masking, computing softmax, and finally multiplying
        by values. The scaling prevents the dot products from growing too
        large in magnitude, which would push the softmax into regions with
        extremely small gradients.
        
        The attention formula is:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        
        Parameters
        ----------
        Q : torch.Tensor
            Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            with dtype torch.float32.
        K : torch.Tensor
            Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            with dtype torch.float32.
        V : torch.Tensor
            Value tensor of shape (batch_size, num_heads, seq_len, d_k)
            with dtype torch.float32.
        mask : torch.Tensor, optional
            Attention mask with dtype torch.bool. Can have shape
            (batch_size, seq_len) which will be broadcast to
            (batch_size, 1, 1, seq_len), or already in the broadcast shape.
            True/1 indicates positions to attend, False/0 to mask.
        
        Returns
        -------
        torch.Tensor
            Attention output of shape (batch_size, num_heads, seq_len, d_k)
            representing the weighted combination of values based on
            attention weights.
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
        """
        Forward pass through multi-head attention.
        
        This method applies multi-head attention by:
        1. Projecting inputs to query, key, value representations
        2. Splitting projections into multiple heads
        3. Computing scaled dot-product attention for each head
        4. Concatenating head outputs
        5. Applying final output projection
        
        For self-attention, all three inputs (query, key, value) are
        typically the same tensor.
        
        Parameters
        ----------
        QKV : tuple
            Tuple of three tensors (query, key, value), each of shape
            (batch_size, seq_len, d_model) with dtype torch.float32.
            For self-attention, all three are identical.
        mask : torch.Tensor, optional
            Attention mask with dtype torch.bool. See scaled_dot_product_attention
            for details on shape and usage.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model) after
            applying multi-head attention and output projection.
        """
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
        # Shape: [batch_size, num_heads, seq_len, d_k]
      
        # 3. Apply scaled dot-product attention
        attention_out = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Concatenate heads and project back to d_model
        attention_out = attention_out.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, d_model
        )
        
        # 5. Final linear projection
        output = self.w_o(attention_out)
        
        return output
    
class OverlappingCNN(nn.Module):
    """
    Convolutional patch extraction and projection layer.
    """
    def __init__(self, patch_size=16, stride=8, embed_dim=256, in_channels=1):
        """
        Initialize overlapping CNN patch extractor.
        
        Parameters
        ----------
        patch_size : int, default=16
            Size of square patches to extract.
        stride : int, default=8
            Stride for convolution.
        embed_dim : int, default=256
            Output dimension for each patch embedding.
        in_channels : int, default=1
            Number of input channels (1 for grayscale, 64 for CNN features).
        """
        super().__init__()
        self.conv_proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride
        )
    
    def forward(self, x):
        """Extract and project patches from input images."""
        x = self.conv_proj(x)        # (B, embed_dim, H_out, W_out)
        x = x.flatten(2)             # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)        # (B, num_patches, embed_dim)
        return x
    
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        
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
    
class VisionTransformerWithLearnableAux(nn.Module):
    """
    Vision Transformer with learnable auxiliary embeddings.
    
    Takes GloVe 200d vector as patch 0, concatenates learnable auxiliary
    patches, and processes through transformer. Only uses patch 0 for
    final classification.
    """
    def __init__(self,
                 glove_dim: int = 200,
                 embed_dim: int = 192,
                 d_ff: int = 1024, 
                 num_heads: int = 8,
                 layers: int = 4,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 num_auxiliary_patches: int = 3,
                 mode: str = "fine-tuning"):
        super().__init__()
        
        # Total patches = 1 GloVe + num_auxiliary
        self.num_auxiliary = num_auxiliary_patches
        self.total_patches = 1 + num_auxiliary_patches
        
        # Project GloVe 200d → embed_dim
        self.glove_projection = nn.Linear(glove_dim, embed_dim)
        
        # Learnable auxiliary patches (randomly initialized)
        # Shape: (1, num_auxiliary, embed_dim)
        self.auxiliary_patches = nn.Parameter(
            torch.randn(1, num_auxiliary_patches, embed_dim) * 0.02
        )
        
        # Positional embeddings for all patches
        self.pos_e = PositionalEncoding(embed_dim, self.total_patches, dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            Layer(embed_dim, d_ff, num_heads, dropout) for _ in range(layers)
        ])

        # Classification head
        if mode == "pre-training":
            self.mlp_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_classes)
            )
        elif mode == "fine-tuning":
            self.mlp_head = nn.Linear(embed_dim, num_classes)
        else:
            raise Exception("Unknown mode chosen for the model.")
        
    def forward(self, glove_vectors: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass with GloVe + learnable auxiliary patches.
        
        Parameters
        ----------
        glove_vectors : torch.Tensor
            GloVe embeddings of shape (batch_size, 1, 20, 10) or (batch_size, 200).
            Will be flattened to (batch_size, 200).
        mask : torch.Tensor, optional
            Attention mask for transformer layers.
        
        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        batch_size = glove_vectors.shape[0]
        
        # Flatten GloVe if needed: (batch, 1, 20, 10) → (batch, 200)
        if glove_vectors.dim() == 4:
            glove_vectors = glove_vectors.view(batch_size, -1)  # (batch, 200)
        
        # 1. Project GloVe to embed_dim: (batch, 200) → (batch, embed_dim)
        glove_patch = self.glove_projection(glove_vectors)  # (batch, embed_dim)
        # glove_patch = glove_vectors
        glove_patch = glove_patch.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # 2. Expand auxiliary patches for batch
        aux_patches = self.auxiliary_patches.expand(batch_size, -1, -1)  # (batch, num_aux, embed_dim)
        
        # 3. Concatenate: [GloVe patch | Auxiliary patches]
        all_patches = torch.cat([glove_patch, aux_patches], dim=1)  # (batch, 1+num_aux, embed_dim)
        
        # 4. Add positional encoding
        x = self.pos_e(all_patches)
        
        # 5. Pass through transformer layers
        # Auxiliary patches interact with GloVe through attention
        for layer in self.layers:
            x = layer(x, mask)
        
        # 6. Use ONLY patch 0 (GloVe) for classification
        avg_representation = x.mean(dim=1)  # (batch, embed_dim)
        
        # 7. Classification head
        output = self.mlp_head(avg_representation)
        
        return output