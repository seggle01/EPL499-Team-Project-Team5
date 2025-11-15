"""
Feature extraction functions for Twitter sentiment analysis.
"""
import re
import json
import emoji
import spacy
from textblob import TextBlob
from pre_processing import *

# Initialize NLP
nlp = spacy.load("en_core_web_sm")

# Load resources
def load_resources():
    """Load all necessary resources for feature extraction."""
    with open('../data/profanity.txt', 'r') as f: 
        profanity_words = [s.strip() for s in f.readlines()]
    
    with open('../data/slang.txt', 'r') as f: 
        slang_words = [s.strip() for s in f.readlines()]
    
    with open("../data/emoji_polarity.json", "r", encoding="utf-8") as f:
        emoji_json = json.load(f)
    
    with open("../data/emoticon_polarity.json", "r", encoding="utf-8") as f:
        emoticon_json = json.load(f)
    
    combined_sentiment = {**emoji_json, **emoticon_json}
    punct_list = ['!', '#', '@', '?']
    
    return profanity_words, slang_words, combined_sentiment, punct_list


# Text pattern features
def count_all_capital_tokens(text: str) -> dict:
    """Count words in all capitals."""
    matches = re.findall(r'\b[A-Z][A-Z]+\b', text)
    return {'all_capital_token_count': len(matches)}


def count_specified_punctuations(text: str, punct_list: list) -> dict:
    """Count specific punctuation marks."""
    punct_occur = {char: text.count(char) for char in punct_list}
    return punct_occur


def count_profanity_words(text: str, profanity_list: list) -> dict:
    """Count profanity words in text."""
    count = 0
    profanity_list = [s.lower() for s in profanity_list]
    tokenized_sent = run_pipeline(text, [word_tokenize_sentence, to_lower])
    for sent in tokenized_sent:
        for token in sent:
            if token in profanity_list:
                count += 1
    return {'profanity_word_count': count}


def count_slang_words(text: str, slang_list: list) -> dict:
    """Count slang words in text."""
    count = 0
    slang_list = [s.lower() for s in slang_list]
    tokenized_sent = run_pipeline(text, [word_tokenize_sentence, to_lower])
    for sent in tokenized_sent:
        for token in sent:
            if token in slang_list:
                count += 1
    return {'slang_word_count': count}


def count_sad_emoticons(text: str) -> dict:
    """Count sad emoticons."""
    pattern = r':\(|:\||:\/|:\\|:\'\(|>:\(|D:|:<|:c|;\(|T_T|T\.T'
    return {"sad_emoticon": len(re.findall(pattern, text))}


def count_happy_emoticons(text: str) -> dict:
    """Count happy emoticons."""
    pattern = r':\)|:D|;D|=\)|;-\)|:\}\)|:>|=\]|8\)|;-D|XD|xD|x-D|X-D|<3|:\*|;-\*|;\)|=D'
    return {"happy_emoticon": len(re.findall(pattern, text))}


def count_not(text: str) -> dict:
    """Count negation words."""
    return {'not_count': len(re.findall(r'dnt|ont|not', text))}


def count_elongated_words(text: str) -> dict:
    """Count words with elongated characters (e.g., 'hellooo')."""
    return {'elongated_word_count': len(re.findall(r'\b\w*(\w)\1{2,}\w*\b', text))}


def count_positive_words(text: str) -> dict:
    """Count positive words from a predefined list."""
    positive_words = ['good', 'happy', 'love', 'great', 'excellent']
    tokens = str(text).lower().split()
    return {'positive_word_count': sum(1 for t in tokens if t in positive_words)}


def count_negative_words(text: str) -> dict:
    """Count negative words from a predefined list."""
    negative_words = ['bad', 'sad', 'hate', 'terrible', 'awful']
    tokens = str(text).lower().split()
    return {'negative_word_count': sum(1 for t in tokens if t in negative_words)}


def uppercase_ratio(text: str) -> dict:
    """Calculate ratio of uppercase letters."""
    total_letters = sum(1 for c in text if c.isalpha())
    ratio = sum(1 for c in text if c.isupper()) / total_letters if total_letters else 0
    return {'uppercase_ratio': ratio}


def tweet_word_count(text: str) -> dict:
    """Count words in tweet."""
    words = text.split()
    return {'word_count': len(words)}


# Linguistic features
def count_pos_tags(text: str) -> dict:
    """Count different POS tags in a tweet."""
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


def count_named_entities(text: str) -> dict:
    """Count named entities by type."""
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
        if f"num_{ent.label_}" in ner_counts:
            ner_counts[f"num_{ent.label_}"] += 1
        else:
            ner_counts["num_MISC"] += 1
    return ner_counts


# Sentiment features
def get_hashtag_polarities(text: str) -> dict:
    """Extract hashtag count."""
    hashtags = re.findall(r"#\w+", text)
    return {"hashtag_count": len(hashtags)}


def get_sentiment_and_subjectivity(text: str, combined_sentiment: dict) -> dict:
    """Calculate sentiment scores combining TextBlob and emoji/emoticon sentiment."""
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    
    tb_pos = pol if pol > 0 else 0
    tb_neg = abs(pol) if pol < 0 else 0
    
    items_in_text = [ch for ch in text if ch in emoji.EMOJI_DATA]
    for em in combined_sentiment:
        if em in text and em not in items_in_text:
            items_in_text.append(em)
    
    if items_in_text:
        pos_list = [combined_sentiment[i]["positivity"] for i in items_in_text if i in combined_sentiment]
        neg_list = [combined_sentiment[i]["negativity"] for i in items_in_text if i in combined_sentiment]
        if pos_list:
            avg_pos = sum(pos_list) / len(pos_list)
            avg_neg = sum(neg_list) / len(neg_list)
            w_pos_emoji, w_neg_emoji = 0.5, 0.5
            final_pos = tb_pos * (1 - w_pos_emoji) + w_pos_emoji * avg_pos
            final_neg = tb_neg * (1 - w_neg_emoji) + w_neg_emoji * avg_neg
        else:
            final_pos, final_neg = tb_pos, tb_neg
    else:
        final_pos, final_neg = tb_pos, tb_neg
    
    return {
        "positive_sentiment": final_pos,
        "negative_sentiment": final_neg,
        "subjectivity": subj
    }


def predict_sarcasm(tweet: str, sarcasm_model, vec) -> dict:
    """Predict sarcasm probability."""
    from sarcasm_classifier import extract_features
    features = extract_features(tweet)
    X = vec.transform([features])
    probability = sarcasm_model.predict_proba(X)[0][0]
    return {"sarcasm_score": float(probability)}