

 # Twitter Sentiment Classification: Positive vs. Negative

A comprehensive sentiment analysis system for classifying Twitter posts as positive or negative using advanced NLP techniques, custom feature engineering, and machine learning.

---

## 👥 Team Members

- Costas Frantzides
- Vasilis Kynigaris
- Stefanos Englezou

---

## 📋 Project Overview

This project implements a robust sentiment classification pipeline for Twitter data. The system combines traditional machine learning approaches with modern NLP techniques to achieve high accuracy in distinguishing positive from negative sentiments in tweets.

### Key Components:
- **Modular Architecture**: Organized into reusable modules for feature extraction, text cleaning, vectorization, model training, and evaluation
- **Custom Feature Engineering**: 40+ hand-crafted features capturing linguistic, emotional, and stylistic patterns
- **Hyperparameter Optimization**: Automated tuning using Optuna for optimal model performance
- **Comprehensive Evaluation**: Detailed analysis including feature importance, confusion matrices, and misclassification reports

---

## 🔧 Feature Types Used

The system extracts **four main categories** of features from Twitter text:

### 1. **TF-IDF Features (3,000 features)**
- Unigrams and bigrams (1-2 word combinations)
- Captures important words and phrases
- Min document frequency: 10
- Max document frequency: 90%

### 2. **Text Pattern Features**
- **Punctuation**: Counts of `!`, `#`, `@`, `?`
- **Capitalization**: All-caps words, uppercase ratio
- **Emoticons**: Happy emoticons (`:)`, `:D`, `<3`) and sad emoticons (`:(`, `:/`, `T_T`)
- **Text Patterns**: Elongated words (e.g., "hellooo"), negation words ("not", "don't")
- **Word Counts**: Total words, positive/negative word lists
- **Profanity & Slang**: Custom dictionary-based counts

### 3. **Linguistic Features**
- **Part-of-Speech (POS) Tags**: 
  - Nouns, verbs, adjectives, adverbs, pronouns
  - Extracted using spaCy
- **Named Entity Recognition (NER)**:
  - PERSON, ORGANIZATION, LOCATION, DATE, MONEY entities
  - Miscellaneous entity category
- **Hashtags**: Count of hashtags in tweets

### 4. **Sentiment Features**
- **TextBlob Sentiment**:
  - Positive sentiment score
  - Negative sentiment score
  - Subjectivity score
- **Emoji/Emoticon Sentiment**:
  - Combined polarity from emoji and emoticon dictionaries
  - Weighted fusion with TextBlob scores (50/50 weight)
- **Sarcasm Detection**:
  - Pre-trained sarcasm classifier probability score

---

## 🤖 Models Implemented

### Primary Model: **SGDClassifier (Stochastic Gradient Descent)**

#### Model Configuration:
- **Loss Function**: Log loss (logistic regression)
- **Optimization**: Optuna hyperparameter tuning (100 trials)
- **Cross-Validation**: 3-fold CV with F1-macro scoring

#### Optimized Hyperparameters:
```python
{
    'alpha': 1.329464377881655e-05,      # L2 regularization
    'eta0': 0.027423046794063346,        # Learning rate
    'learning_rate': 'adaptive',          # Learning schedule
    'max_iter': 5000,                     # Maximum iterations
    'tol': 1e-3                           # Convergence tolerance
}
```

### Supporting Model: **Random Forest Classifier**
- Used for feature importance analysis
- 100 estimators
- Helps identify most influential features

---

## 📊 Key Results and Findings

### Model Performance

| Metric | Score |
|--------|-------|
| **Cross-Validation F1-Macro** | 0.8187 |
| **Test Accuracy** | *[Add from your results]* |
| **Precision (Negative)** | *[Add from your results]* |
| **Precision (Positive)** | *[Add from your results]* |
| **Recall (Negative)** | *[Add from your results]* |
| **Recall (Positive)** | *[Add from your results]* |

### Top 25 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | positive_sentiment | 0.0519 |
| 2 | negative_sentiment | 0.0448 |
| 3 | sarcasm_score | 0.0230 |
| 4 | ! (exclamation) | 0.0227 |
| 5 | uppercase_ratio | 0.0221 |
| 6 | subjectivity | 0.0150 |
| 7 | word_count | 0.0135 |
| 8 | num_verbs | 0.0117 |
| 9 | profanity_word_count | 0.0116 |
| 10 | positive_word_count | 0.0104 |
| 11 | num_MISC | 0.0102 |
| 12 | not_count | 0.0096 |
| 13 | "may" (TF-IDF) | 0.0091 |
| 14 | num_nouns | 0.0082 |
| 15 | num_pronouns | 0.0075 |
| 16 | num_adverbs | 0.0071 |
| 17 | "user" (TF-IDF) | 0.0063 |
| 18 | num_adjectives | 0.0063 |
| 19 | "tomorrow" (TF-IDF) | 0.0062 |
| 20 | happy_emoticon | 0.0061 |
| 21 | "see" (TF-IDF) | 0.0051 |
| 22 | "best" (TF-IDF) | 0.0049 |
| 23 | slang_word_count | 0.0048 |
| 24 | "love" (TF-IDF) | 0.0048 |
| 25 | all_capital_token_count | 0.0046 |

### Key Insights

1. **Sentiment Scores Dominate**: The top 2 features are positive and negative sentiment scores, highlighting the importance of combining TextBlob with emoji/emoticon sentiment analysis.

2. **Sarcasm Detection is Critical**: Sarcasm score ranks 3rd, indicating that detecting sarcasm significantly improves classification accuracy.

3. **Stylistic Features Matter**: Punctuation (especially `!`), capitalization patterns, and emoticons provide strong signals for sentiment.

4. **Linguistic Structure Helps**: POS tags and named entities contribute to understanding tweet context and sentiment nuances.

5. **Hybrid Approach Works Best**: The combination of TF-IDF (semantic content) with custom engineered features (emotional and stylistic patterns) yields superior performance compared to either approach alone.

6. **Negation is Important**: The presence of negation words (not, don't) ranks highly, crucial for reversing sentiment polarity.

---

## 🗂️ Project Structure

```
twitter-sentiment-classification/
│
├── data/
│   ├── twitter_sentiment_train.csv
│   ├── twitter_sentiment_test.csv
│   ├── sarcasm_model.pkl
│   ├── dict_vectorizer.pkl
│   ├── profanity.txt
│   ├── slang.txt
│   ├── emoji_polarity.json
│   └── emoticon_polarity.json
│
├── src/
│   ├── main.py                    # Main pipeline orchestrator
│   ├── feature_extraction.py      # Feature extraction functions
│   ├── text_cleaning.py           # Text preprocessing utilities
│   ├── vectorization.py           # TF-IDF vectorization
│   ├── model_training.py          # Model training & optimization
│   ├── evaluation.py              # Evaluation & visualization
│   ├── pre_processing.py          # Basic preprocessing functions
│   └── sarcasm_classifier.py      # Sarcasm detection module
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn spacy textblob emoji joblib optuna tqdm matplotlib seaborn
python -m spacy download en_core_web_sm
```

### Running the Pipeline

```bash
cd src
python main.py
```

### Using Individual Modules

```python
from feature_extraction import count_positive_words, get_sentiment_and_subjectivity
from text_cleaning import preprocessing_text
from evaluation import evaluate_model

# Example: Extract features from text
text = "I love this amazing product! 😊"
sentiment = get_sentiment_and_subjectivity(text, combined_sentiment_dict)
print(sentiment)
```

---

## 📈 Future Improvements

- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add ensemble methods combining multiple classifiers
- [ ] Expand feature set with word embeddings (Word2Vec, GloVe)
- [ ] Real-time sentiment analysis API
- [ ] Multi-class sentiment classification (positive, negative, neutral)
- [ ] Cross-domain sentiment analysis (different social media platforms)
- [ ] Improved sarcasm and irony detection

---

## 📝 License

*[Add your license information]*

---

## 🙏 Acknowledgments

- spaCy for NLP processing
- TextBlob for sentiment analysis
- Optuna for hyperparameter optimization
- scikit-learn for machine learning tools