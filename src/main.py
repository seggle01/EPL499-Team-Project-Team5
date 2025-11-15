"""
Main pipeline for Twitter Sentiment Classification.
"""
import pandas as pd
import numpy as np
from joblib import load
from tqdm import tqdm

# Import custom modules
from text_cleaning import preprocessing_text
from vectorization import tfidf_features
from feature_extraction import (
    load_resources,
    count_specified_punctuations,
    count_profanity_words,
    count_slang_words,
    count_all_capital_tokens,
    count_not,
    count_sad_emoticons,
    count_happy_emoticons,
    count_elongated_words,
    count_positive_words,
    count_negative_words,
    uppercase_ratio,
    count_pos_tags,
    count_named_entities,
    get_sentiment_and_subjectivity,
    get_hashtag_polarities,
    predict_sarcasm,
    tweet_word_count
)
from model_training import optimize_sgd_classifier, train_final_model
from evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    analyze_misclassifications,
    get_feature_importance,
    plot_feature_importance
)


def main():
    # Configuration
    RANDOM_STATE = 123
    
    # Load data
    print("Loading data...")
    df_train = pd.read_csv('../data/twitter_sentiment_train.csv')
    df_test = pd.read_csv('../data/twitter_sentiment_test.csv')
    
    # Shuffle train set
    df_train = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Load sarcasm model/vectorizer
    print("Loading sarcasm model and vectorizer...")
    sarcasm_model = load("../data/sarcasm_model.pkl")
    vec = load("../data/dict_vectorizer.pkl")
    
    # Load resources
    print("Loading resources...")
    profanity_words, slang_words, combined_sentiment, punct_list = load_resources()
    
    # Text preprocessing
    print("Preprocessing text...")
    clean_tokens_train = [preprocessing_text(t) for t in df_train['text']]
    clean_tokens_test = [preprocessing_text(t) for t in df_test['text']]
    clean_text_train = [' '.join(tokens) for tokens in clean_tokens_train]
    clean_text_test = [' '.join(tokens) for tokens in clean_tokens_test]
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf_train, tfidf_test, vectorizer = tfidf_features(
        clean_text_train,
        clean_text_test,
        ngram_range=(1, 2),
        max_features=3000
    )
    
    # Define feature functions
    feature_functions = [
        lambda text: count_specified_punctuations(text, punct_list),
        lambda text: count_profanity_words(text, profanity_words),
        lambda text: count_slang_words(text, slang_words),
        count_all_capital_tokens,
        count_not,
        count_sad_emoticons,
        count_happy_emoticons,
        count_elongated_words,
        count_positive_words,
        count_negative_words,
        uppercase_ratio,
        count_pos_tags,
        count_named_entities,
        lambda text: get_sentiment_and_subjectivity(text, combined_sentiment),
        get_hashtag_polarities,
        lambda text: predict_sarcasm(text, sarcasm_model, vec),
        tweet_word_count
    ]
    
    # Extract custom features
    print("Extracting custom features...")
    X_train = df_train[['text']].copy()
    X_test = df_test[['text']].copy()
    
    for func in tqdm(feature_functions):
        results = X_train['text'].apply(lambda x: func(str(x))).tolist()
        X_train = pd.concat([X_train, pd.DataFrame(results)], axis=1)
        
        results = X_test['text'].apply(lambda x: func(str(x))).tolist()
        X_test = pd.concat([X_test, pd.DataFrame(results)], axis=1)
    
    X_train_custom = X_train.drop(columns=['text'])
    X_test_custom = X_test.drop(columns=['text'])
    
    # Combine TF-IDF + custom features
    print("Combining features...")
    X_train_final = pd.concat(
        [tfidf_train.reset_index(drop=True), X_train_custom.reset_index(drop=True)],
        axis=1
    )
    X_test_final = pd.concat(
        [tfidf_test.reset_index(drop=True), X_test_custom.reset_index(drop=True)],
        axis=1
    )
    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)
    
    y_train = df_train['label']
    y_test = df_test['label']
    
    # Hyperparameter optimization
    print("\nOptimizing hyperparameters...")
    best_params, study = optimize_sgd_classifier(
        X_train_final,
        y_train,
        n_trials=100,
        random_state=RANDOM_STATE
    )
    
    # Train final model
    print("\nTraining final model...")
    final_model = train_final_model(X_train_final, y_train, best_params, RANDOM_STATE)
    
    # Make predictions
    y_pred_final = final_model.predict(X_test_final)
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    feature_importances = get_feature_importance(X_train_final, y_train, top_n=25)
    plot_feature_importance(feature_importances, top_n=25)
    
    # Evaluate model
    evaluate_model(y_test, y_pred_final)
    
    # Analyze misclassifications
    misclassified_indices = analyze_misclassifications(y_test, y_pred_final)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_final)
    
    print("\nPipeline complete!")
    
    return final_model, vectorizer, feature_importances


if __name__ == "__main__":
    model, vectorizer, importances = main()