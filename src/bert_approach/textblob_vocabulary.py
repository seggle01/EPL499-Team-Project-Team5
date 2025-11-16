# textblob_vocabulary.py

import pandas as pd
import numpy as np
from collections import Counter
from textblob import TextBlob
from word_normalization import preprocessing_text
import emoji
import json
from tqdm import tqdm

def calculate_textblob_scores(df_train, df_test, min_freq=3):
    """
    Calculate polarity and subjectivity for each word using TextBlob.
    
    Returns:
        dict: {word: {'polarity': float, 'subjectivity': float, 'frequency': int}}
    """
    print("Calculating TextBlob scores for all tokens...")
    
    # Collect all unique words with frequency
    word_freq = Counter()
    
    for text in pd.concat([df_train['text'], df_test['text']]):
        tokens = preprocessing_text(text)
        word_freq.update(tokens)
    
    # Calculate polarity and subjectivity for each word
    word_scores = {}
    
    for word, freq in tqdm(word_freq.items(), desc="Analyzing words"):
        if freq < min_freq:
            continue
        
        # Skip emojis (handle separately)
        if emoji.is_emoji(word):
            continue
        
        # Get TextBlob sentiment
        blob = TextBlob(word)
        polarity = blob.sentiment.polarity  # -1 (neg) to +1 (pos)
        subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        
        word_scores[word] = {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'frequency': freq
        }
    
    return word_scores

def build_textblob_vocabulary(df_train, df_test, max_vocab=5000, 
                              polarity_weight=0.5, 
                              subjectivity_weight=0.3,
                              frequency_weight=0.2):
    """
    Build vocabulary using TextBlob polarity and subjectivity scores.
    
    Selection criteria:
    - High polarity (strong positive or negative sentiment)
    - High subjectivity (opinionated words)
    - Reasonable frequency (used in multiple tweets)
    
    Args:
        max_vocab: Target vocabulary size
        polarity_weight: Weight for polarity strength (default 0.5)
        subjectivity_weight: Weight for subjectivity (default 0.3)
        frequency_weight: Weight for word frequency (default 0.2)
    """
    print("="*70)
    print("BUILDING TEXTBLOB-BASED VOCABULARY")
    print("="*70)
    
    # Step 1: Calculate TextBlob scores
    print("\n1. Calculating TextBlob sentiment scores...")
    word_scores = calculate_textblob_scores(df_train, df_test, min_freq=3)
    print(f"   Analyzed {len(word_scores)} words")
    
    # Step 2: Rank words by combined score
    print("\n2. Ranking words by sentiment strength...")
    
    word_rankings = {}
    for word, scores in word_scores.items():
        # Polarity strength (absolute value - we want strong opinions)
        polarity_strength = abs(scores['polarity'])
        
        # Subjectivity (higher = more opinionated)
        subjectivity_score = scores['subjectivity']
        
        # Frequency importance (log scale)
        freq_score = np.log1p(scores['frequency']) / 10.0  # Normalize
        
        # Combined score
        combined_score = (
            polarity_weight * polarity_strength +
            subjectivity_weight * subjectivity_score +
            frequency_weight * freq_score
        )
        
        word_rankings[word] = {
            'combined_score': combined_score,
            'polarity': scores['polarity'],
            'polarity_strength': polarity_strength,
            'subjectivity': scores['subjectivity'],
            'frequency': scores['frequency']
        }
    
    # Sort by combined score
    ranked_words = sorted(word_rankings.items(), 
                         key=lambda x: x[1]['combined_score'], 
                         reverse=True)
    
    # Step 3: Show top sentiment words
    print(f"\n3. Top 30 words by TextBlob sentiment:")
    print(f"   {'Word':<15} {'Polarity':<10} {'Subject.':<10} {'Freq':<8} {'Score':<8}")
    print(f"   {'-'*60}")
    
    for word, data in ranked_words[:30]:
        polarity_label = "POS" if data['polarity'] > 0 else "NEG" if data['polarity'] < 0 else "NEU"
        print(f"   {word:<15} {data['polarity']:>6.3f} {polarity_label:<3} "
              f"{data['subjectivity']:>6.3f}     {data['frequency']:<8} {data['combined_score']:.3f}")
    
    # Step 4: Build vocabulary
    print(f"\n4. Building vocabulary (target: {max_vocab} words)...")
    
    vocab = {
        '<pad>': 0,
        '<cls>': 1,
        '<sep>': 2
    }
    
    index = 3
    
    # Add critical sentiment words (guaranteed inclusion)
    critical_words = {
        # Strong positive
        'love', 'amazing', 'awesome', 'best', 'great', 'excellent',
        'perfect', 'wonderful', 'fantastic', 'beautiful', 'brilliant',
        # Strong negative
        'hate', 'worst', 'terrible', 'awful', 'horrible', 'bad',
        'disgusting', 'pathetic', 'useless', 'disappointing', 'waste',
        # Negations (CRITICAL!)
        'not', 'no', 'never', 'nor', 'neither', 'nobody', 'nothing',
        'nowhere', 'hardly', 'barely', 'scarcely', 'cannot',
        # Intensifiers
        'very', 'really', 'so', 'too', 'extremely', 'absolutely',
        'completely', 'totally', 'highly'
    }
    
    critical_added = 0
    for word in critical_words:
        if word not in vocab:
            vocab[word] = index
            index += 1
            critical_added += 1
    
    print(f"   Added {critical_added} critical sentiment words")
    
    # Add top-ranked words from TextBlob
    added_from_textblob = 0
    for word, data in ranked_words:
        if index >= max_vocab - 200:  # Leave room for emojis
            break
        if word not in vocab:
            vocab[word] = index
            index += 1
            added_from_textblob += 1
    
    print(f"   Added {added_from_textblob} TextBlob-ranked words")
    
    # Step 5: Add ALL emojis
    print(f"\n5. Adding emojis...")
    all_emojis = set()
    
    for text in pd.concat([df_train['text'], df_test['text']]):
        tokens = preprocessing_text(text)
        emojis = [t for t in tokens if emoji.is_emoji(t)]
        all_emojis.update(emojis)
    
    for emoji_char in sorted(all_emojis):
        vocab[emoji_char] = index
        index += 1
    
    print(f"   Added {len(all_emojis)} emojis")
    
    # Step 6: Calculate statistics
    print(f"\n6. Vocabulary statistics...")
    
    # Count by sentiment
    strong_positive = sum(1 for w in vocab if w in word_rankings and word_rankings[w]['polarity'] > 0.3)
    strong_negative = sum(1 for w in vocab if w in word_rankings and word_rankings[w]['polarity'] < -0.3)
    neutral = len(vocab) - strong_positive - strong_negative - len(all_emojis) - 3
    
    # Calculate coverage
    total_tokens = 0
    covered_tokens = 0
    
    for text in df_train['text']:
        tokens = preprocessing_text(text)
        total_tokens += len(tokens)
        covered_tokens += sum(1 for t in tokens if t in vocab)
    
    coverage = 100 * covered_tokens / total_tokens
    
    print(f"\n{'='*70}")
    print(f"✓ TEXTBLOB VOCABULARY COMPLETE")
    print(f"{'='*70}")
    print(f"  Total vocabulary: {len(vocab)}")
    print(f"  - Special tokens: 3")
    print(f"  - Strong positive words: {strong_positive} (polarity > 0.3)")
    print(f"  - Strong negative words: {strong_negative} (polarity < -0.3)")
    print(f"  - Neutral/weak sentiment: {neutral}")
    print(f"  - Emojis: {len(all_emojis)}")
    print(f"  Coverage: {coverage:.2f}% of training corpus")
    print(f"{'='*70}\n")
    
    return vocab, word_rankings

# Main execution
if __name__ == "__main__":
    import os
    
    # Load data
    DATA_PATH = '../../data/'
    df_train = pd.read_csv(os.path.join(DATA_PATH, 'twitter_sentiment_train.csv'))
    df_test = pd.read_csv(os.path.join(DATA_PATH, 'twitter_sentiment_test.csv'))
    
    # Build vocabulary with TextBlob
    vocab, word_rankings = build_textblob_vocabulary(
        df_train,
        df_test,
        max_vocab=5000,
        polarity_weight=0.5,      # 50% weight on polarity strength
        subjectivity_weight=0.3,  # 30% weight on subjectivity
        frequency_weight=0.2      # 20% weight on frequency
    )
    
    # Save vocabulary
    vocab_path = './vocab_textblob.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Vocabulary saved to {vocab_path}")
    
    # Save word rankings for analysis
    rankings_path = './word_rankings_textblob.json'
    with open(rankings_path, 'w') as f:
        json.dump(word_rankings, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Word rankings saved to {rankings_path}")
