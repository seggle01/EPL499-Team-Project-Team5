import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

def load_glove(path, EMBED_DIM, use_cache=True):
    """
    Load GloVe word embeddings from file into a dictionary with caching.
    
    Parameters
    ----------
    path : str
        File path to the GloVe embeddings file.
    EMBED_DIM : int
        Expected dimensionality of the embedding vectors.
    use_cache : bool, optional
        If True, use cached pickle file if available (default: True).
    
    Returns
    -------
    dict
        Dictionary mapping words (str) to their embedding vectors (np.array).
        Only includes vectors that match the specified EMBED_DIM.
    """
    # Create cache filename based on original file
    cache_path = Path('./glove/glove').with_suffix('.pkl')
    
    # Try to load from cache first
    if use_cache and cache_path.exists():
        print(f"Loading GloVe from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            glove = pickle.load(f)
        print(f"Loaded {len(glove)} word vectors from cache")
        return glove
    
    # Load from original text file
    print(f"Loading GloVe from text file: {path}")
    glove = {}
    
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            
            if vec.shape[0] != EMBED_DIM:
                continue
            
            glove[word] = vec
    
    # Save to cache for next time
    print(f"Saving cache to: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(glove, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Loaded {len(glove)} word vectors")
    return glove

def tweet_to_glove_vector(text, glove, EMBED_DIM):
    """
    Convert preprocessed tweet text to a single averaged GloVe embedding vector.
    
    Uses space-separated tokens from preprocessed text (lemmatized/lowercased).
    Words not found in the GloVe vocabulary are ignored. If no valid tokens
    are found, returns a zero vector.
    
    Parameters
    ----------
    text : str
        Space-joined string of preprocessed tokens from 'clean_text_*' column.
    glove : dict
        Dictionary mapping words to GloVe embedding vectors.
    EMBED_DIM : int
        Dimensionality of the embedding vectors.
    
    Returns
    -------
    np.array
        Averaged embedding vector of shape (EMBED_DIM,). Returns zero vector
        if no tokens are found in the GloVe vocabulary.
    """
    # Split text into individual tokens
    tokens = text.split()
    
    # Look up each token in GloVe dictionary, collecting valid vectors
    vecs = [glove[t] for t in tokens if t in glove]
    
    # Return zero vector if no tokens found in vocabulary
    if len(vecs) == 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    
    # Compute mean of all token vectors along axis 0 (average pooling)
    return np.mean(vecs, axis=0)
