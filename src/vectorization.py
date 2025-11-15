"""
Text vectorization utilities.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(training_data, test_data, ngram_range=(1, 2), max_features=3000):
    """
    Create TF-IDF features from text data.
    
    Args:
        training_data: List or Series of training texts
        test_data: List or Series of test texts
        ngram_range: Tuple of (min_n, max_n) for n-grams
        max_features: Maximum number of features to keep
        
    Returns:
        tfidf_train: DataFrame of TF-IDF features for training data
        tfidf_test: DataFrame of TF-IDF features for test data
        vectorizer: Fitted TfidfVectorizer object
    """
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=False,
        min_df=10,
        max_df=0.90
    )
    
    tfidf_train = pd.DataFrame(
        tfidf.fit_transform(training_data).toarray(),
        columns=tfidf.get_feature_names_out()
    )
    
    tfidf_test = pd.DataFrame(
        tfidf.transform(test_data).toarray(),
        columns=tfidf.get_feature_names_out()
    )
    
    return tfidf_train, tfidf_test, tfidf