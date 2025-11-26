
# # Twitter Sentiment Classification: Positive vs. Negative


import pandas as pd

df_train = pd.read_csv('../data/twitter_sentiment_train.csv')
df_test  = pd.read_csv('../data/twitter_sentiment_test.csv')

# Shuffle train set
RANDOM_STATE = 123
df_train = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


import numpy as np        # must be imported BEFORE joblib load
from joblib import load

print("Loading model and vectorizer...")

# load the model saved as joblib
sarcasm_model = load("../data/sarcasm_model.joblib")

# load the vectorizer saved as joblib
vec = load("../data/dict_vectorizer.joblib")





int_to_label = {1: 'Positive', 0: 'Negative'}


df_train.head(5)


# ### Import libraries


import re
import string
import json
import emoji
import spacy
from tqdm import tqdm
from textblob import TextBlob
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pre_processing import *
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
import nltk

nlp = spacy.load("en_core_web_sm")


with open('../data/profanity.txt', 'r') as f: 
    profanity_words = f.readlines()
profanity_words = [s.strip() for s in profanity_words]
#  Load slang list
with open('../data/slang.txt', 'r') as f:
    slang_words = [s.strip() for s in f.readlines()]

nltk.download('opinion_lexicon')
nltk.download('punkt')

positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Load emoji JSON
with open("../data/emoji_polarity.json", "r", encoding="utf-8") as f:
    emoji_json = json.load(f)

# Load emoticon JSON
with open("../data/emoticon_polarity.json", "r", encoding="utf-8") as f:
    emoticon_json = json.load(f)

# Merge both dictionaries
combined_sentiment = {**emoji_json, **emoticon_json}



punct_list = ['!', '#', '@', '?', '"']



# ### Feature Extraction Checklist
# 
# 1. Profanity words count
# 2. Sentiment and Subjectivity 
# 3. Emoji Sentiment + Emoticon e.g :), 😂, :((
# 3. Fully Capitalized
# 4. Punctuations


def count_all_capital_tokens(text: str) -> dict:
    """
    Counts the number of fully capitalized tokens (all letters uppercase) in a given text.
    Returns: {'all_capital_token_count': count}
    """
    matches = re.findall(r'\b[A-Z][A-Z]+\b', text)
    return {'all_capital_token_count': len(matches)}

def count_specified_punctuations(text: str, punct_list: list) -> dict:

    """
    Counts the occurrences of each punctuation mark in a given text.
    Returns: {'punctuation_char1': count1, 'punctuation_char2': count2, ...}
    """
    punct_occur = {}
    for char in punct_list:
        punct_occur[char] = 0
    for char in text:
        if char in punct_list:
            punct_occur[char] += 1
    return punct_occur
    """
    Counts sequences of repeated punctuation marks and specific patterns.
    Returns counts for repeated '!', '?', and mixed '!?' or '?!'.
    """
    exclam = len(re.findall(r'!{2,}', text))
    question = len(re.findall(r'\?{2,}', text))
    mixed = len(re.findall(r'(\!\?|\?\!)', text))
    
    # Optional: total repeated punctuation (all types)
    total_repeated = len(re.findall(r'([!?.,])\1{1,}', text))
    
    return {
        'repeat_exclam': exclam,
        'repeat_question': question,
        'repeat_mixed': mixed,
        'total_repeated_punct': total_repeated
    }


def count_profanity_words(text: str, profanity_list: list) -> dict:
    """
    Counts the number of profanity words in a given text using a predefined list.
    Returns: {'profanity_word_count': count}
    """
    count = 0
    # Normalize both input and the word list using the to_lower() function
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

def count_sad_emoticons(text: str):
    """
    Returns the occurrences of sad emoticons.
    Returns: {
      "sad_emoticon": count
    }
    """
    # Sad, crying, angry, and negative emoticons
    matches = re.findall(r':\(|:\||:\/|:\\|:\'\(|>:\(|D:|:<|:c|;\(|T_T|T\.T', text)
    return {"sad_emoticon": len(matches)}

def count_happy_emoticons(text: str):
    """
    Returns the occurrences of happy emoticons.
    Returns: {
      "happy_emoticon": count
    }
    """
    # Happy, excited, laughing, and positive emoticons
    matches = re.findall(r':\)|:D|;D|=\)|;-\)|:\}\)|:>|=\]|8\)|;-D|XD|xD|x-D|X-D|<3|:\*|;-\*|;\)|=D', text)
    return {"happy_emoticon": len(matches)}

def count_not(text: str):
    negation_pattern = r"\b(?:not|no|never|n't|cannot|cant|dont|doesnt|didnt|won't|wouldnt|shouldnt|couldnt|isnt|aren't|ain't)\b"
    matches = re.findall(negation_pattern, text.lower())
    return {'not_count': len(matches)}

def count_elongated_words(text):
    matches = re.findall(r'\b\w*(\w)\1{2,}\w*\b', text)
    return {'elongated_word_count': len(matches)}

def count_positive_words(text):
    tokens = word_tokenize(text.lower())
    return {'positive_word_count': sum(1 for t in tokens if t in positive_words)}

def count_negative_words(text):
    tokens = word_tokenize(text.lower())
    return {'negative_word_count': sum(1 for t in tokens if t in negative_words)}


def uppercase_ratio(text):
    total_letters = sum(1 for c in text if c.isalpha())
    return {'uppercase_ratio': sum(1 for c in text if c.isupper()) / total_letters} if total_letters else {'uppercase_ratio': 0}

def get_sentiment_and_subjectivity(text: str) -> dict:
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    
    tb_pos = pol if pol > 0 else 0
    tb_neg = abs(pol) if pol < 0 else 0

    # Find all emojis and emoticons in text
    items_in_text = [ch for ch in text if ch in emoji.EMOJI_DATA]  # emojis
    # emoticons (like :) ;D) — check by splitting text
    for em in combined_sentiment:
        if em in text and em not in items_in_text:
            items_in_text.append(em)

    if items_in_text:
        pos_list = [combined_sentiment[i]["positivity"] for i in items_in_text if i in combined_sentiment]
        neg_list = [combined_sentiment[i]["negativity"] for i in items_in_text if i in combined_sentiment]

        if pos_list:
            avg_pos = sum(pos_list) / len(pos_list)
            avg_neg = sum(neg_list) / len(neg_list)
            w_pos_emoji = 0.5
            final_pos = (tb_pos * (1-w_pos_emoji) + w_pos_emoji * avg_pos) 
            w_neg_emoji = 0.5
            final_neg = (tb_neg * (1-w_neg_emoji) + w_neg_emoji * avg_neg)  
        else:
            final_pos, final_neg = tb_pos, tb_neg
    else:
        final_pos, final_neg = tb_pos, tb_neg

    return {
        "positive_sentiment": final_pos,
        "negative_sentiment": final_neg,
        "subjectivity": subj
    }

def count_pos_tags(text):
    """
    Count detailed POS tags and ratios in a tweet.
    Returns a dict with counts and ratios for useful POS categories.
    """
    doc = nlp(text)
    total_tokens = len(doc) if len(doc) > 0 else 1

    pos_counts = {
        "num_nouns": 0,
        "num_verbs": 0,
        "num_adjectives": 0,
        "num_adverbs": 0,
        "num_pronouns": 0,
        "num_interjections": 0,
        "num_aux_verbs": 0,
        "num_determiners": 0,
        "num_particles": 0,
        "num_negations": 0
    }

    for token in doc:
        pos = token.pos_
        if pos == "NOUN":
            pos_counts["num_nouns"] += 1
        elif pos == "VERB":
            pos_counts["num_verbs"] += 1
        elif pos == "ADJ":
            pos_counts["num_adjectives"] += 1
        elif pos == "ADV":
            pos_counts["num_adverbs"] += 1
        elif pos == "PRON":
            pos_counts["num_pronouns"] += 1
        elif pos == "INTJ":
            pos_counts["num_interjections"] += 1
        elif pos == "AUX":
            pos_counts["num_aux_verbs"] += 1
        elif pos == "DET":
            pos_counts["num_determiners"] += 1
        elif pos == "PART":
            pos_counts["num_particles"] += 1

        # Count negations
        if token.lower_ in ["not", "n't", "no", "never", "cannot", "cant", "doesnt", "dont", "didnt", "wouldnt", "shouldnt"]:
            pos_counts["num_negations"] += 1

    # Add ratios
    for key in list(pos_counts.keys()):
        pos_counts[key + "_ratio"] = pos_counts[key] / total_tokens

    return pos_counts

def count_named_entities(text):
    """
    Count named entities by type in a tweet and include social media cues.
    Returns dict with counts, ratios, and co-occurrence features.
    """
    doc = nlp(text)
    total_tokens = len(doc) if len(doc) > 0 else 1

    ner_counts = {
        "num_PERSON": 0,
        "num_ORG": 0,
        "num_LOC": 0,
        "num_GPE": 0,
        "num_DATE": 0,
        "num_MONEY": 0,
        "num_PRODUCT": 0,
        "num_EVENT": 0,
        "num_WORK_OF_ART": 0,
        "num_LANGUAGE": 0,
        "num_MISC": 0,
        "num_mentions": text.count('@'),
        "num_hashtags": text.count('#')
    }

    for ent in doc.ents:
        label = ent.label_
        if label in ner_counts:
            ner_counts[label] += 1
        else:
            ner_counts["num_MISC"] += 1

    # Ratios
    for key in list(ner_counts.keys()):
        ner_counts[key + "_ratio"] = ner_counts[key] / total_tokens

    # Multiple entity co-occurrence feature
    ner_counts["multiple_entities"] = 1 if sum([ner_counts[k] for k in ner_counts if k.startswith("num_") and "_ratio" not in k]) > 1 else 0

    return ner_counts
    """
    Count the number of named entities by type in a tweet.
    Returns a dict with counts for common entity types.
    """
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
        return {
            # "hashtag_avg_polarity": sum(polarities)/len(polarities),
            # "hashtag_max_polarity": max(polarities),
            # "hashtag_min_polarity": min(polarities),
            "hashtag_count": len(polarities)
        }
    else:
        return {
            # "hashtag_avg_polarity": 0,
            # "hashtag_max_polarity": 0,
            # "hashtag_min_polarity": 0,
            "hashtag_count": 0
        }
    
def predict_sarcasm(tweet):
    """
    Predict if a tweet is sarcastic or not.
    
    Args:
        tweet (str): The tweet text to classify
        
    Returns:
        tuple: (prediction, probability) where prediction is 'Sarcastic' or 'Not Sarcastic'
    """
    from sarcasm_classifier import extract_features
    # Extract features
    features = extract_features(tweet)
    
    # Transform to model input format
    X = vec.transform([features])
    
    # Get prediction and probability
    prediction = sarcasm_model.predict(X)[0]
    probability = sarcasm_model.predict_proba(X)[0][0]

    # print(probability)
    
    return {
        "sarcasm_score" : float(probability)
    }
def tweet_word_count(text):
    """
    Returns the number of words in the tweet.
    """
    words = text.split()
    return {'word_count': len(words)}

def tweet_avg_word_length(text):
    """
    Returns the average word length in the tweet.
    """
    words = text.split()
    if words:
        return {'avg_word_length': sum(len(w) for w in words)/len(words)}
    else:
        return {'avg_word_length': 0}
    """
    Detects sentiment contrast within a text, often indicating sarcasm.
    
    Features:
    - sentiment_contrast: 1 if at least one sentence has both positive and negative words
    - polarity_flip_ratio: fraction of sentences with opposite polarity vs total sentences
    """
    # Split text into sentences (simple split by punctuation)
    sentences = re.split(r'[.!?…]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    contrast_count = 0
    
    for sentence in sentences:
        blob = TextBlob(sentence)
        words = blob.words
        
        pos_words = [w for w in words if TextBlob(w).sentiment.polarity > 0.3]
        neg_words = [w for w in words if TextBlob(w).sentiment.polarity < -0.3]
        
        if pos_words and neg_words:
            contrast_count += 1
    
    sentiment_contrast = 1 if contrast_count > 0 else 0
    polarity_flip_ratio = contrast_count / len(sentences) if sentences else 0
    
    return {
        "sentiment_contrast": sentiment_contrast,
        "polarity_flip_ratio": polarity_flip_ratio
    }



def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
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
    """
    Replaces common unicode characters with ASCII equivalents.
    Useful for tweet preprocessing.
    """
    text = re.sub(r'\\u2019', "'", text)
    text = re.sub(r'\\u201c', '"', text)
    text = re.sub(r'\\u201d', '"', text)
    text = re.sub(r'\\u002c', ',', text)
        
    return text

def remove_numbers(text: str):
    return re.sub(r'[0-9]','',text)

def preprocessing_text(text):
    # Word normalization
    text = clean_unicode(text)
    text = remove_numbers(text)
    text = uncontract(text)
    text = convert_urls_emails(text)
    # Classical preprocessing steps
    text = word_tokenize_sentence(text)
    text = to_lower(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text

# Apply preprocessing to the datasets
clean_tokens_train = [preprocessing_text(t) for t in df_train['text']]
clean_tokens_test = [preprocessing_text(t) for t in df_test['text']]
clean_text_train = [' '.join(tokens) for tokens in clean_tokens_train]
clean_text_test = [' '.join(tokens) for tokens in clean_tokens_test]

def tfidf_features(training_data, test_data, ngram_range, max_features):

    tfidf = TfidfVectorizer(
        ngram_range  = ngram_range,
        max_features = max_features,
        lowercase    = False,
        tokenizer    = None,
        preprocessor = None,
        stop_words   = None,
        min_df       = 10,
        max_df       = 0.90
        )

    tfidf_train = tfidf.fit_transform(training_data)

    tfidf_train = tfidf_train.toarray()
    tfidf_train = pd.DataFrame(tfidf_train)
    tfidf_train.columns = tfidf.get_feature_names_out()

    tfidf_test = tfidf.transform(test_data)

    tfidf_test = tfidf_test.toarray()
    tfidf_test = pd.DataFrame(tfidf_test)
    tfidf_test.columns = tfidf.get_feature_names_out()

    return tfidf_train, tfidf_test, tfidf

tfidf_train, tfidf_test, vectorizer = tfidf_features(clean_text_train, clean_text_test, (1,2), 3100)


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
    tweet_word_count,
    tweet_avg_word_length,
]

X_train = df_train[['text']].copy()
X_test = df_test[['text']].copy()

for func in tqdm(feature_functions):
    results = X_train['text'].apply(lambda x: func(str(x))).tolist()
    temp_df = pd.DataFrame(results)
    
    temp_df.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_train = pd.concat([X_train, temp_df], axis=1)
    
    results = X_test['text'].apply(lambda x: func(str(x))).tolist()
    temp_df = pd.DataFrame(results)
    
    temp_df.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    X_test = pd.concat([X_test, temp_df], axis=1)

# Drop the text column from custom features
X_train_custom = X_train.drop(columns=['text'])
X_test_custom = X_test.drop(columns=['text'])

# Concatenate TF-IDF + Custom Features
tfidf_train.reset_index(drop=True, inplace=True)
tfidf_test.reset_index(drop=True, inplace=True)
X_train_custom.reset_index(drop=True, inplace=True)
X_test_custom.reset_index(drop=True, inplace=True)

X_train_combined = pd.concat([tfidf_train, X_train_custom], axis=1)
X_test_combined = pd.concat([tfidf_test, X_test_custom], axis=1)

# Convert column names to strings before scaling
X_train_combined.columns = X_train_combined.columns.astype(str)
X_test_combined.columns = X_test_combined.columns.astype(str)

X_train_final = X_train_combined
X_test_final = X_test_combined

X_train_final.head(5)


y_train = df_train['label']
y_test = df_test['label']

model = SGDClassifier(
    loss='log_loss',
    learning_rate='adaptive',
    max_iter=5000,
    alpha=1.577782177e-05,
    eta0=0.016506,
    random_state=RANDOM_STATE
)

model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))


# Find misclassified examples
misclassified_indices = y_test.index[y_test != y_pred].tolist()

print(f"\n{'='*80}")
print(f"Total Misclassified: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.2f}%)")
print(f"{'='*80}\n")

# Display misclassified tweets
int_to_label = {1: 'Positive', 0: 'Negative'}

for i, idx in enumerate(misclassified_indices[:50], 1):  # Show first 50
    true_label = y_test[idx]
    pred_label = y_pred[y_test.index.get_loc(idx)]
    text = df_test.loc[idx, 'text']
    
    print(f"Example {i}:")
    print(f"  Text: {text}")
    print(f"  True Label: {int_to_label[true_label]}")
    print(f"  Predicted: {int_to_label[pred_label]}")
    print(f"  {'-'*76}\n")


# Train a Random Forest on your features
model_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model_rf.fit(X_train_final, y_train)
# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model_rf.feature_importances_
})

# Sort by importance
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Display top 25 features
print("\nTop 25 Most Important Features:")
print(feature_importances.head(25))

# Plot top 25 features
plt.figure(figsize=(10, 8))
top_25 = feature_importances.head(25)
plt.barh(range(len(top_25)), top_25['importance'].values,color='skyblue')
plt.yticks(range(len(top_25)), top_25['feature'].values)
plt.xlabel('Importance')
plt.title('Top 25 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


