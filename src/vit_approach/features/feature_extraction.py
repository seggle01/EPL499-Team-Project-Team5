import re
import spacy
import emoji
from textblob import TextBlob
import pandas as pd
from text_processing.pre_processing import *

from spacy.cli import download

# Download the small English model
# download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


def count_all_capital_tokens(text: str):
    """
    Method to count all the fully capitalized tokens in a given text.

    Parameters
    ----------
    text : str
        Text containing tokens.
    
    Returns
    -------
    dict : Dictionary of the form {'all_capital_token_count': count}
    """
    matches = re.findall(r'\b[A-Z][A-Z]+\b', text)
    return {'all_capital_token_count': len(matches)}

def count_specified_punctuations(text: str, punct_list: list):
    """
    Method to count the occurrences of each punctuation mark in a given text.

    Parameters
    ----------
    text : str
        Text to count punctuations.
    punct_list : list
        List of punctuation marks to count.
    
    Returns
    -------
    dict : Dictionary of the form {'punctuation_char1': count1, 'punctuation_char2': count2, ...}
    """
    punct_occur = {}
    for char in punct_list:
        punct_occur[char] = 0
    for char in text:
        if char in punct_list:
            punct_occur[char] += 1
    return punct_occur

def count_profanity_words(text: str, profanity_list: list):
    """
    Method to count the number of profanity words in a given text using a predefined list.

    Parameters
    ----------
    text : str
        Text to search profanity words.
    profanity_list : list
        List of profanity words.
    
    Returns
    -------
    dict : Dictionary of the form {'profanity_word_count': count}
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

def count_slang_words(text: str, slang_list: list):
    """
    Method to count the number of slang words in a given text using a predefined list.

    Parameters
    ----------
    text : str
        Text to search slang words.
    slang_list : list
        List of slang words.
    
    Returns
    -------
    dict : Dictionary of the form {'slang_word_count': count}
    """
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
    Method to count the occurrences of sad emoticons.

    Parameters
    ----------
    text : str
        Text to search sad emoticons.
    
    Returns
    -------
    dict : Dictionary of the form {"sad_emoticon": count}
    """
    # Sad, crying, angry, and negative emoticons
    matches = re.findall(r':\(|:\||:\/|:\\|:\'\(|>:\(|D:|:<|:c|;\(|T_T|T\.T', text)
    return {"sad_emoticon": len(matches)}

def count_happy_emoticons(text: str):
    """
    Method to count the occurrences of happy emoticons.

    Parameters
    ----------
    text : str
        Text to search happy emoticons.
    
    Returns
    -------
    dict : Dictionary of the form {"happy_emoticon": count}
    """
    # Happy, excited, laughing, and positive emoticons
    matches = re.findall(r':\)|:D|;D|=\)|;-\)|:\}\)|:>|=\]|8\)|;-D|XD|xD|x-D|X-D|<3|:\*|;-\*|;\)|=D', text)
    return {"happy_emoticon": len(matches)}

def count_not(text: str):
    """
    Method to count negation words like 'not', 'dont' etc.

    Parameters
    ----------
    text : str
        Text to search negation words.
    
    Returns
    -------
    dict : Dictionary of the form {"not_count": count}
    """
    negation_pattern = r"\b(?:not|no|never|n't|cannot|cant|dont|doesnt|didnt|won't|wouldnt|shouldnt|couldnt|isnt|aren't|ain't)\b"
    matches = re.findall(negation_pattern, text.lower())
    return {'not_count': len(matches)}

def count_elongated_words(text: str):
    """
    Method to count elongated words like 'wowwwwww'.

    Parameters
    ----------
    text : str
        Text to search elongated words.
    
    Returns
    -------
    dict : Dictionary of the form {"elongated_word_count": count}
    """
    matches = re.findall(r'\b\w*(\w)\1{2,}\w*\b', text)
    return {'elongated_word_count': len(matches)}

def count_positive_words(text: str, positive_words: List):
    """
    Method to count positive words from a predefined list.

    Parameters
    ----------
    text : str
        Text to search positive words.
    positive_words : List
        List of positive words.
    
    Returns
    -------
    dict : Dictionary of the form {"positive_word_count": count}
    """
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text.lower())
    return {'positive_word_count': sum(1 for t in tokens if t in positive_words)}

def count_negative_words(text: str, negative_words: List):
    """
    Method to count negative words from a predefined list.

    Parameters
    ----------
    text : str
        Text to search negative words.
    negative_words : List
        List of negative words.
    
    Returns
    -------
    dict : Dictionary of the form {"negative_word_count": count}
    """
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text.lower())
    return {'negative_word_count': sum(1 for t in tokens if t in negative_words)}

def uppercase_ratio(text: str):
    """
    Method to calculate the ratio of uppercase letters to total alphabetic characters.

    Parameters
    ----------
    text : str
        Text to analyze uppercase ratio.
    
    Returns
    -------
    dict : Dictionary of the form {'uppercase_ratio': ratio}
    """
    total_letters = sum(1 for c in text if c.isalpha())
    return {'uppercase_ratio': sum(1 for c in text if c.isupper()) / total_letters} if total_letters else {'uppercase_ratio': 0}

def get_sentiment_and_subjectivity(text: str, combined_sentiment: dict):
    """
    Method to compute sentiment polarity and subjectivity scores combining TextBlob analysis with emoji/emoticon sentiment.

    Parameters
    ----------
    text : str
        Text to analyze for sentiment and subjectivity.
    combined_sentiment : dict
        Dictionary containing emoji and emoticons along their sentiment.
    
    Returns
    -------
    dict : Dictionary containing:
        - 'positive_sentiment': Positive sentiment score (0.0 to 1.0)
        - 'negative_sentiment': Negative sentiment score (0.0 to 1.0)
        - 'subjectivity': Subjectivity score (0.0 to 1.0)
    """
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

def count_pos_tags(text: str):
    """
    Method to count detailed part-of-speech (POS) tags and their ratios in text.

    Parameters
    ----------
    text : str
        Text to analyze for POS tags.
    
    Returns
    -------
    dict : Dictionary containing counts and ratios for each POS category including:
        - num_nouns, num_verbs, num_adjectives, num_adverbs, num_pronouns
        - num_interjections, num_aux_verbs, num_determiners, num_particles, num_negations
        - Corresponding ratio values for each count (e.g., num_nouns_ratio)
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

def count_named_entities(text: str):
    """
    Method to count named entities by type and social media cues (mentions, hashtags).

    Parameters
    ----------
    text : str
        Text to extract and count named entities.
    
    Returns
    -------
    dict : Dictionary containing counts and ratios for entity types including:
        - num_PERSON, num_ORG, num_LOC, num_GPE, num_DATE, num_MONEY
        - num_PRODUCT, num_EVENT, num_WORK_OF_ART, num_LANGUAGE, num_MISC
        - num_mentions, num_hashtags
        - Corresponding ratio values for each count
        - multiple_entities: Binary flag indicating if more than one entity type present
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

def predict_sarcasm(tweet: str, vec, model):
    """
    Method to predict if a tweet is sarcastic using a pre-trained classifier.

    Parameters
    ----------
    tweet : str
        The tweet text to classify for sarcasm.
    vec : DictVectorizer
        Pre-loaded DictVectorizer for feature transformation.
    model : sklearn classifier
        Pre-loaded trained classifier model.
    
    Returns
    -------
    dict : Dictionary of the form {'sarcasm_score': probability}
           where probability is the sarcasm likelihood score (0.0 to 1.0)
    """
    from features.sarcasm_classifier import extract_features
    
    # Extract features from tweet
    features = extract_features(tweet)
    
    # Transform to model input format (vec is already loaded)
    X = vec.transform([features])
    
    # Get prediction and probability
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][0]
    
    return {
        "sarcasm_score": float(probability)
    }

def tweet_word_count(text: str):
    """
    Method to count the number of words in a tweet.

    Parameters
    ----------
    text : str
        Tweet text to count words.
    
    Returns
    -------
    dict : Dictionary of the form {'word_count': count}
    """
    words = text.split()
    return {'word_count': len(words)}

def tweet_avg_word_length(text: str):
    """
    Method to calculate the average word length in a tweet.

    Parameters
    ----------
    text : str
        Tweet text to compute average word length.
    
    Returns
    -------
    dict : Dictionary of the form {'avg_word_length': average_length}
    """
    words = text.split()
    if words:
        return {'avg_word_length': sum(len(w) for w in words) / len(words)}
    else:
        return {'avg_word_length': 0}

def tfidf_features(training_data, test_data, ngram_range, max_features):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
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
