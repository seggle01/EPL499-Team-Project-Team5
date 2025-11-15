# %% [markdown]
# # Twitter Irony / Sarcasm Classification

# %%
import re
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------
# 1️⃣  Feature Extraction Functions
# ------------------------------------------------------------

def count_punctuation(text):
    return {
        "exclamation_marks": text.count("!"),
        "question_marks": text.count("?"),
        "dots": text.count(".")
    }

def count_all_capital_tokens(text):
    tokens = re.findall(r'\b[A-Z]{2,}\b', text)
    return {"all_caps": len(tokens)}

def count_emoticons(text):
    happy = len(re.findall(r'(:\)|:-\)|:D|=\)|😊|😁|😃)', text))
    sad = len(re.findall(r'(:\(|:-\(|😞|😢|☹️)', text))
    return {"happy_emoticons": happy, "sad_emoticons": sad}

def get_sentiment_and_subjectivity(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    return {
        "positive_sentiment": pol if pol > 0 else 0,
        "negative_sentiment": abs(pol) if pol < 0 else 0,
        "subjectivity": subj
    }

def get_hashtag_polarities(text):
    hashtags = re.findall(r"#(\w+)", text)
    scores = [TextBlob(tag).sentiment.polarity for tag in hashtags]
    return {
        "hashtag_count": len(hashtags),
        "avg_hashtag_polarity": sum(scores)/len(scores) if scores else 0
    }

def sentiment_contrast_features(text):
    blob = TextBlob(text)
    words = blob.words
    pos_words = [w for w in words if TextBlob(w).sentiment.polarity > 0.3]
    neg_words = [w for w in words if TextBlob(w).sentiment.polarity < -0.3]
    return {
        "pos_word_count": len(pos_words),
        "neg_word_count": len(neg_words),
        "sentiment_contrast": 1 if (len(pos_words) > 0 and len(neg_words) > 0) else 0
    }

def exaggeration_features(text):
    elongated = len(re.findall(r'(.)\1{2,}', text))
    multi_punct = len(re.findall(r'([!?])\1{1,}', text))
    return {"elongated_words": elongated, "multi_punct": multi_punct}

def sarcasm_hashtag_feature(text):
    sarcasm_tags = re.findall(r"#sarcasm|#irony|#sarcastic", text.lower())
    return {"has_sarcasm_tag": 1 if sarcasm_tags else 0}

# ------------------------------------------------------------
# 2️⃣  Combine All Features
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 3️⃣  Load Training and Test Datasets
# ------------------------------------------------------------

df_train = pd.read_csv("sarcasm_training.csv")  # your training CSV
df_test  = pd.read_csv("sarcasm_test.csv")      # your test CSV

# Convert 'yes'/'no' to 1/0
df_train["Sarcasm (yes/no)"] = df_train["Sarcasm (yes/no)"].map({"yes": 1, "no": 0})
df_test["Sarcasm (yes/no)"]  = df_test["Sarcasm (yes/no)"].map({"yes": 1, "no": 0})

X_train_texts = df_train["Tweet"].astype(str).tolist()
y_train = df_train["Sarcasm (yes/no)"].tolist()

X_test_texts = df_test["Tweet"].astype(str).tolist()
y_test = df_test["Sarcasm (yes/no)"].tolist()

# ------------------------------------------------------------
# 4️⃣  Feature Extraction
# ------------------------------------------------------------

X_train_features = [extract_features(text) for text in X_train_texts]
X_test_features  = [extract_features(text) for text in X_test_texts]

vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train_features)
X_test  = vec.transform(X_test_features)

# ------------------------------------------------------------
# 5️⃣  Train Logistic Regression Model
# ------------------------------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# 6️⃣  Evaluation
# ------------------------------------------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
