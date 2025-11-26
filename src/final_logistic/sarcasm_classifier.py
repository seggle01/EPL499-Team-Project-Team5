
# # Twitter Irony / Sarcasm Classification
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from sarcasm_feature_extraction import *

def extract_features(text):
    features = {}
    functions = [
        count_punctuation,
        count_all_capital_tokens,
        count_emoticons,
        get_sentiment_and_subjectivity,
        get_hashtag_polarities,
        sentiment_contrast_features,
        exaggeration_features,
        sarcasm_hashtag_feature
    ]
    for func in functions:
        features.update(func(text))
    return features

def main():
    df_train = pd.read_csv("../data/sarcasm_training.csv")
    df_test  = pd.read_csv("../data/sarcasm_test.csv")

    df_train["Sarcasm (yes/no)"] = df_train["Sarcasm (yes/no)"].map({"yes": 1, "no": 0})
    df_test["Sarcasm (yes/no)"]  = df_test["Sarcasm (yes/no)"].map({"yes": 1, "no": 0})

    X_train_texts = df_train["Tweet"].astype(str).tolist()
    y_train = df_train["Sarcasm (yes/no)"].tolist()

    X_test_texts = df_test["Tweet"].astype(str).tolist()
    y_test = df_test["Sarcasm (yes/no)"].tolist()

    # # ------------------------------------------------------------
    # # Feature Extraction
    # # ------------------------------------------------------------

    X_train_features = [extract_features(text) for text in X_train_texts]
    X_test_features  = [extract_features(text) for text in X_test_texts]

    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train_features)
    X_test  = vec.transform(X_test_features)

    # # ------------------------------------------------------------
    # # Train Logistic Regression Model
    # # ------------------------------------------------------------

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    import pickle

    # Save both the trained model and the vectorizer
    with open("../data/sarcasm_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=5)

    with open("../data/dict_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f, protocol=5)

    y_pred = model.predict(X_test)

