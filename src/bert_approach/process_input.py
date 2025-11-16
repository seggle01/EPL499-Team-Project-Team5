import torch
import numpy as np
from word_normalization import *

def text_to_indices(text, vocab, max_len):
    """
    Convert text to token indices with BERT-style special tokens.
    
    Args:
        text: Raw text string
        vocab: Vocabulary dictionary
        max_len: Maximum sequence length
        
    Returns:
        indices: List of token indices
    """
    # Preprocess text to tokens
    tokens = preprocessing_text(text)
    
    # Add BERT-style special tokens: [CLS] + tokens + [SEP]
    tokens = ['<cls>'] + tokens + ['<sep>']
    
    # Convert tokens to indices (use <unk> for unknown tokens)
    indices = []
    for token in tokens:
        if token in vocab:
            indices.append(vocab[token])
        # else:
        #     indices.append(vocab['<unk>'])
    
    # Truncate if too long (keep [CLS] and [SEP])
    if len(indices) > max_len:
        indices = indices[:max_len-1] + [vocab['<sep>']]
    
    # Pad if too short
    while len(indices) < max_len:
        indices.append(vocab['<pad>'])
    
    return indices

def create_attention_mask(indices, pad_idx):
    """
    Create attention mask (1 for real tokens, 0 for padding).
    
    Args:
        indices: List of token indices
        pad_idx: Index of padding token
        
    Returns:
        mask: Binary mask
    """
    return [1 if idx != pad_idx else 0 for idx in indices]

def process_batch(batch_texts, batch_labels, vocab, max_len, device):
    """
    Process a batch of texts and convert to tensors.
    """
    batch_indices = []
    batch_masks = []
    
    # Process each text in batch
    for text in batch_texts:
        # Convert text to indices
        indices = text_to_indices(text, vocab, max_len)
        
        # Create attention mask
        mask = create_attention_mask(indices, vocab['<pad>'])
        
        batch_indices.append(indices)
        batch_masks.append(mask)
    
    # Convert to PyTorch tensors
    input_ids = torch.LongTensor(batch_indices).to(device)
    attention_masks = torch.LongTensor(batch_masks).to(device)
    
    # FIX: Ensure labels are 1D, not 2D
    labels = torch.LongTensor(batch_labels.values).to(device)  # Shape: [batch_size]
    
    return input_ids, attention_masks, labels
