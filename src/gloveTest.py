# glove_integrated.py
# Your original pipeline with TF-IDF replaced by GloVe Twitter 200d embeddings.
import os
import pandas as pd
import numpy as np        # must be imported BEFORE joblib load
from joblib import load
import re
import string
import json
import emoji
import spacy
from tqdm import tqdm
from textblob import TextBlob
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.feature_extraction.text import TfidfVectorizer  # no longer used
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pre_processing import *   # your preprocessing helpers (word_tokenize_sentence, to_lower, etc.)
import optuna

# ---------------------------
# 1) Load data (unchanged)
# ---------------------------
df_train = pd.read_csv('../data/twitter_sentiment_train.csv')
df_test  = pd.read_csv('../data/twitter_sentiment_test.csv')

# Shuffle train set
RANDOM_STATE = 123
df_train = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

#  Load sarcasm model/vectorizer
print("Loading model and vectorizer...")
sarcasm_model = load("../data/sarcasm_model.pkl")
vec = load("../data/dict_vectorizer.pkl")

#  Initialize NLP
nlp = spacy.load("en_core_web_sm")
int_to_label = {1: 'Positive', 0: 'Negative'}

#  Load profanity list
with open('../data/profanity.txt', 'r') as f:
    profanity_words = [s.strip() for s in f.readlines()]

#  Load slang list
with open('../data/slang.txt', 'r') as f:
    slang_words = [s.strip() for s in f.readlines()]

punct_list = ['!','#','@','?']

with open("../data/emoji_polarity.json", "r", encoding="utf-8") as f:
    emoji_json = json.load(f)
with open("../data/emoticon_polarity.json", "r", encoding="utf-8") as f:
    emoticon_json = json.load(f)
combined_sentiment = {**emoji_json, **emoticon_json}

# ---------------------------
# 2) All  feature funcs 
# ---------------------------
def count_all_capital_tokens(text: str) -> dict:
    matches = re.findall(r'\b[A-Z][A-Z]+\b', text)
    return {'all_capital_token_count': len(matches)}
def count_specified_punctuations(text: str, punct_list: list) -> dict:
    punct_occur = {char: text.count(char) for char in punct_list}
    return punct_occur
def count_profanity_words(text: str, profanity_list: list) -> dict:
    count = 0
    profanity_list = [s.lower() for s in profanity_list]
    tokenized_sent = run_pipeline(text, [word_tokenize_sentence, to_lower])
    for sent in tokenized_sent:
        for token in sent:
            if token in profanity_list:
                count += 1
    return {'profanity_word_count': count}
def count_slang_words(text: str, slang_list: list) -> dict:
    count = 0
    slang_list = [s.lower() for s in slang_list]
    tokenized_sent = run_pipeline(text, [word_tokenize_sentence, to_lower])
    for sent in tokenized_sent:
        for token in sent:
            if token in slang_list:
                count += 1
    return {'slang_word_count': count}
def count_sad_emoticons(text: str): return {"sad_emoticon": len(re.findall(r':\(|:\||:\/|:\\|:\'\(|>:\(|D:|:<|:c|;\(|T_T|T\.T', text))}
def count_happy_emoticons(text: str): return {"happy_emoticon": len(re.findall(r':\)|:D|;D|=\)|;-\)|:\}\)|:>|=\]|8\)|;-D|XD|xD|x-D|X-D|<3|:\*|;-\*|;\)|=D', text))}
def count_not(text: str): return {'not_count': len(re.findall(r'dnt|ont|not', text))}
def count_elongated_words(text): return {'elongated_word_count': len(re.findall(r'\b\w*(\w)\1{2,}\w*\b', text))}
def count_positive_words(text):
    positive_words = ['good', 'happy', 'love', 'great', 'excellent']
    tokens = str(text).lower().split()
    return {'positive_word_count': sum(1 for t in tokens if t in positive_words)}
def count_negative_words(text):
    negative_words = ['bad', 'sad', 'hate', 'terrible', 'awful']
    tokens = str(text).lower().split()
    return {'negative_word_count': sum(1 for t in tokens if t in negative_words)}
def uppercase_ratio(text):
    total_letters = sum(1 for c in text if c.isalpha())
    return {'uppercase_ratio': sum(1 for c in text if c.isupper()) / total_letters} if total_letters else {'uppercase_ratio': 0}
def count_pos_tags(text):
    doc = nlp(text)
    pos_counts = {
        "num_nouns": 0,
        "num_verbs": 0,
        "num_adjectives": 0,
        "num_adverbs": 0,
        "num_pronouns": 0,
    }
    for token in doc:
        if token.pos_ == "NOUN":
            pos_counts["num_nouns"] += 1
        elif token.pos_ == "VERB":
            pos_counts["num_verbs"] += 1
        elif token.pos_ == "ADJ":
            pos_counts["num_adjectives"] += 1
        elif token.pos_ == "ADV":
            pos_counts["num_adverbs"] += 1
        elif token.pos_ == "PRON":
            pos_counts["num_pronouns"] += 1
    return pos_counts
def count_named_entities(text):
    doc = nlp(text)
    ner_counts = {
        "num_PERSON": 0,
        "num_ORG": 0,
        "num_LOC": 0,
        "num_DATE": 0,
        "num_MONEY": 0,
        "num_MISC": 0,
    }
    for ent in doc.ents:
        if ent.label_ in ner_counts:
            ner_counts[f"num_{ent.label_}"] += 1
        else:
            ner_counts["num_MISC"] += 1
    return ner_counts
def get_hashtag_polarities(text: str) -> dict:
    hashtags = re.findall(r"#\w+", text)
    polarities = [TextBlob(tag[1:]).sentiment.polarity for tag in hashtags]
    if polarities:
        return {"hashtag_count": len(polarities)}
    else:
        return {"hashtag_count": 0}
def get_sentiment_and_subjectivity(text: str) -> dict:
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    tb_pos = pol if pol > 0 else 0
    tb_neg = abs(pol) if pol < 0 else 0
    items_in_text = [ch for ch in text if ch in emoji.EMOJI_DATA]
    for em in combined_sentiment:
        if em in text and em not in items_in_text: items_in_text.append(em)
    if items_in_text:
        pos_list = [combined_sentiment[i]["positivity"] for i in items_in_text if i in combined_sentiment]
        neg_list = [combined_sentiment[i]["negativity"] for i in items_in_text if i in combined_sentiment]
        if pos_list:
            avg_pos = sum(pos_list)/len(pos_list)
            avg_neg = sum(neg_list)/len(neg_list)
            w_pos_emoji, w_neg_emoji = 0.5, 0.5
            final_pos = tb_pos*(1-w_pos_emoji) + w_pos_emoji*avg_pos
            final_neg = tb_neg*(1-w_neg_emoji) + w_neg_emoji*avg_neg
        else:
            final_pos, final_neg = tb_pos, tb_neg
    else:
        final_pos, final_neg = tb_pos, tb_neg
    return {"positive_sentiment": final_pos, "negative_sentiment": final_neg, "subjectivity": subj}
def tweet_word_count(text):
    words = text.split()
    return {'word_count': len(words)}
def predict_sarcasm(tweet):
    from sarcasm_classifier import extract_features
    features = extract_features(tweet)
    X = vec.transform([features])
    prediction = sarcasm_model.predict(X)[0]
    probability = sarcasm_model.predict_proba(X)[0][0]
    return {"sarcasm_score": float(probability)}

def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ii]t)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
    return text

def convert_urls_emails(text):
    url_regex_1 = r'^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$'
    url_regex_2 = r'^[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$'
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(url_regex_1, 'URL', text)
    text = re.sub(url_regex_2, 'URL', text)
    text = re.sub(email_regex, 'EMAIL', text)
    return text

def clean_unicode(text: str):
    text = re.sub(r'\\u2019', "'", text)
    text = re.sub(r'\\u201c', '"', text)
    text = re.sub(r'\\u201d', '"', text)
    text = re.sub(r'\\u002c', ',', text)
    return text

def remove_numbers(text: str): return re.sub(r'[0-9]','',text)

def preprocessing_text(text):
    text = clean_unicode(text)
    text = remove_numbers(text)
    text = uncontract(text)
    text = convert_urls_emails(text)
    text = word_tokenize_sentence(text)
    text = to_lower(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text

# ---------------------------
# 3) Your cleaning/tokenized variables (unchanged)
# ---------------------------
clean_tokens_train = [preprocessing_text(t) for t in df_train['text']]
clean_tokens_test = [preprocessing_text(t) for t in df_test['text']]
clean_text_train = [' '.join(tokens) for tokens in clean_tokens_train]
clean_text_test = [' '.join(tokens) for tokens in clean_tokens_test]

# ---------------------------
# 4) use GloVe 
# ---------------------------

# Put your exact GloVe path here (you already provided it)
glove_path = r"C:\Users\35797\git\EPL499-Team-Project-Team5\data\glove.twitter.27B.200d.txt"
assert os.path.exists(glove_path), f"GloVe file not found at: {glove_path}"

EMBED_DIM = 200  # matches the 200d file

def load_glove(path):
    """Load GloVe file into dict: word -> np.array(vector)."""
    glove = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] != EMBED_DIM:
                continue
            glove[word] = vec
    return glove

# load (this may take ~30-60s depending on machine)
glove_vectors = load_glove(glove_path)
print(f"Loaded {len(glove_vectors)} glove vectors (dim={EMBED_DIM})")

def tweet_to_glove_vector(text, glove, dim=EMBED_DIM):
    """
    Convert preprocessed tweet (space-joined tokens) to averaged GloVe vector.
    Uses tokens from 'clean_text_*' (so tokens are lemmatized/lowercased as in your pipeline).
    """
    tokens = text.split()
    vecs = [glove[t] for t in tokens if t in glove]
    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

# create glove feature matrices (use tqdm for progress)
glove_train = np.vstack([tweet_to_glove_vector(t, glove_vectors) for t in tqdm(clean_text_train, desc="Embedding train")])
glove_test  = np.vstack([tweet_to_glove_vector(t, glove_vectors) for t in tqdm(clean_text_test, desc="Embedding test")])

glove_train = pd.DataFrame(glove_train, columns=[f"glv_{i}" for i in range(EMBED_DIM)])
glove_test  = pd.DataFrame(glove_test,  columns=[f"glv_{i}" for i in range(EMBED_DIM)])

# ---------------------------
# 5) Compute all your custom features (unchanged)
# ---------------------------
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
    get_sentiment_and_subjectivity,
    get_hashtag_polarities,
    predict_sarcasm,
    tweet_word_count
]

X_train = df_train[['text']].copy()
X_test  = df_test[['text']].copy()

for func in tqdm(feature_functions, desc="Computing custom features"):
    results = X_train['text'].apply(lambda x: func(str(x))).tolist()
    X_train = pd.concat([X_train, pd.DataFrame(results)], axis=1)
    results = X_test['text'].apply(lambda x: func(str(x))).tolist()
    X_test = pd.concat([X_test, pd.DataFrame(results)], axis=1)

X_train_custom = X_train.drop(columns=['text'])
X_test_custom  = X_test.drop(columns=['text'])

# ---------------------------
# 6) FINAL feature matrices: (GloVe) + (custom features)
# ---------------------------
X_train_final = pd.concat([glove_train.reset_index(drop=True), X_train_custom.reset_index(drop=True)], axis=1)
X_test_final  = pd.concat([glove_test.reset_index(drop=True),  X_test_custom.reset_index(drop=True)], axis=1)

# ensure column names are strings (helps downstream)
X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns  = X_test_final.columns.astype(str)

y_train = df_train['label']
y_test  = df_test['label']

print("Final feature shapes:", X_train_final.shape, X_test_final.shape)

# ---------------------------
# 7) Optuna objective & optimization for SGD (unchanged)
# ---------------------------
def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    eta0 = trial.suggest_float('eta0', 1e-3, 0.5, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
    model = SGDClassifier(
        loss='log_loss',
        alpha=alpha,
        eta0=eta0,
        learning_rate=learning_rate,
        max_iter=5000,
        random_state=RANDOM_STATE,
        tol=1e-3
    )
    score = cross_val_score(model, X_train_final, y_train, cv=3, scoring='f1_macro', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
print("F1_macro:", study.best_trial.value)
print("Params:", study.best_trial.params)

best_params = study.best_trial.params
final_model = SGDClassifier(
    loss='log_loss',
    alpha=best_params['alpha'],
    eta0=best_params['eta0'],
    learning_rate=best_params['learning_rate'],
    max_iter=5000,
    random_state=RANDOM_STATE,
    tol=1e-3
)

final_model.fit(X_train_final, y_train)
y_pred_final = final_model.predict(X_test_final)

# Train a Random Forest on your features (unchanged)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_final, y_train)
feature_importances = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 25 Most Important Features:")
print(feature_importances.head(25))

# Plot top 25 features
plt.figure(figsize=(10, 8))
top_25 = feature_importances.head(25)
plt.barh(range(len(top_25)), top_25['importance'].values, color='skyblue')
plt.yticks(range(len(top_25)), top_25['feature'].values)
plt.xlabel('Importance')
plt.title('Top 25 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n=== Final SGDClassifier (Optuna tuned) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final, target_names=['negative', 'positive']))

misclassified_indices = y_test.index[y_test != y_pred_final].tolist()
print(f"\n{'='*80}")
print(f"Total Misclassified: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.2f}%)")
print(f"{'='*80}\n")

cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SGD Classifier')
plt.show()
