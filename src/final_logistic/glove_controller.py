import numpy as np
from tqdm import tqdm

def load_glove(path, EMBED_DIM):
    """
    Load GloVe word embeddings from file into a dictionary.
    
    Parameters
    ----------
    path : str
        File path to the GloVe embeddings file.
    EMBED_DIM : int
        Expected dimensionality of the embedding vectors.
    
    Returns
    -------
    dict
        Dictionary mapping words (str) to their embedding vectors (np.array).
        Only includes vectors that match the specified EMBED_DIM.
    """
    glove = {}
    
    # Open GloVe file with UTF-8 encoding to handle special characters
    with open(path, 'r', encoding='utf8') as f:
        # Iterate through each line in the file with progress bar
        for line in tqdm(f, desc="Loading GloVe"):
            # Split line into word and vector components
            parts = line.rstrip().split(' ')
            word = parts[0]  # First element is the word
            
            # Convert remaining elements to float32 numpy array
            vec = np.asarray(parts[1:], dtype=np.float32)
            
            # Skip vectors that don't match expected dimension
            if vec.shape[0] != EMBED_DIM:
                continue
            
            # Store word-vector pair in dictionary
            glove[word] = vec
    
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
