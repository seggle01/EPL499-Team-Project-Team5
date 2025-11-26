import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from sarcasm_feature_extraction import *

def extract_features(text):
    """
    Extracts custom features for sarcasm classification from a single tweet.

    Combines multiple feature functions to generate a dictionary representation
    for each tweet, suitable for processing by DictVectorizer.

    Parameters
    ----------
    text : str
        Original tweet text for feature extraction.

    Returns
    -------
    dict
        Dictionary mapping feature names to their computed values.
    """
    features = {}
    # List of feature-extraction functions applied to each tweet
    functions = [
        count_punctuation,           # Counts punctuation marks
        count_all_capital_tokens,    # Counts fully capitalized tokens
        count_emoticons,             # Counts emoticons/emojis
        get_sentiment_and_subjectivity, # Sentiment analysis
        get_hashtag_polarities,      # Hashtag sentiment polarity
        sentiment_contrast_features, # Features for sarcasm/contrast
        exaggeration_features,       # Detects exaggeration cues
        sarcasm_hashtag_feature      # Detects sarcasm hashtags like #sarcasm
    ]
    # Update features dictionary with outputs from each function
    for func in functions:
        features.update(func(text))
    return features

def main():
    """
    Main entry point for sarcasm classification pipeline.

    Loads data, preprocesses tweets, extracts features, vectorizes them,
    trains logistic regression, saves model and vectorizer, and evaluates.

    Steps:
    - Load and preprocess CSV datasets
    - Map target labels to binary format
    - Extract feature dictionaries for all tweets
    - Transform dictionaries to feature arrays using DictVectorizer
    - Train a LogisticRegression model on features
    - Save model and vectorizer to disk
    - Predict on the test set and evaluate performance

    Returns
    -------
    None
        Prints performance metrics and saves model artifacts.
    """
    # Load sarcasm datasets
    df_train = pd.read_csv("../data/sarcasm_training.csv")
    df_test  = pd.read_csv("../data/sarcasm_test.csv")

    # Convert sarcasm labels from 'yes'/'no' to 1/0
    df_train["Sarcasm (yes/no)"] = df_train["Sarcasm (yes/no)"].map({"yes": 1, "no": 0})
    df_test["Sarcasm (yes/no)"]  = df_test["Sarcasm (yes/no)"].map({"yes": 1, "no": 0})

    # Get raw tweet texts and target labels for train and test splits
    X_train_texts = df_train["Tweet"].astype(str).tolist()
    y_train = df_train["Sarcasm (yes/no)"].tolist()
    X_test_texts = df_test["Tweet"].astype(str).tolist()
    y_test = df_test["Sarcasm (yes/no)"].tolist()

    # ---------------------------------------------
    # Feature Extraction
    # ---------------------------------------------

    # Extract feature dictionaries from tweet texts
    X_train_features = [extract_features(text) for text in X_train_texts]
    X_test_features  = [extract_features(text) for text in X_test_texts]

    # Vectorize feature dicts to numeric feature arrays for model input
    vec = DictVectorizer(sparse=False)
    # Learn feature mappings and convert training features
    X_train = vec.fit_transform(X_train_features)
    # Convert test features using same mapping
    X_test  = vec.transform(X_test_features)

    # ---------------------------------------------
    # Train Logistic Regression Model
    # ---------------------------------------------

    # Create and train logistic regression classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    import pickle
    # Save trained classifier to disk (.pkl file)
    with open("../data/sarcasm_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=5)

    # Save DictVectorizer to disk (.pkl file) to ensure reproducibility
    with open("../data/dict_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f, protocol=5)

    # Predict sarcasm labels on test set
    y_pred = model.predict(X_test)

    # Print model evaluation metrics
    print("\n=== Sarcasm Model Performance ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["non-sarcastic", "sarcastic"]))
