import re
from textblob import TextBlob



def count_punctuation(text):
    """
    Counts the number of exclamation marks, question marks, and dots in the input text.

    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    ----------
    dict : A dictionary with counts of 'exclamation_marks', 'question_marks', and 'dots'.

    """
    return {
        "exclamation_marks": text.count("!"),
        "question_marks": text.count("?"),
        "dots": text.count(".")
    }

def count_all_capital_tokens(text):
    """
    Counts the number of tokens in the input text that are in all capital letters.

    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    ----------
    dict : A dictionary with the count of all-capital tokens under the key 'all_caps'.

    """
    tokens = re.findall(r'\b[A-Z]{2,}\b', text)
    return {"all_caps": len(tokens)}

def count_emoticons(text):
    """
    Counts happy and sad emoticons in the input text.

    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    ----------
    dict : A dictionary with counts of 'happy_emoticons' and 'sad_emoticons'.

    """
    happy = len(re.findall(r'(:\)|:-\)|:D|=\)|😊|😁|😃)', text))
    sad = len(re.findall(r'(:\(|:-\(|😞|😢|☹️)', text))
    return {"happy_emoticons": happy, "sad_emoticons": sad}

def get_sentiment_and_subjectivity(text):
    """
    Analyzes the sentiment polarity and subjectivity of the input text using TextBlob.

    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    ----------
    dict : A dictionary with counts of 'positive_sentiment', 'negative_sentiment', and 'subjectivity'.


    """
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    return {
        "positive_sentiment": pol if pol > 0 else 0,
        "negative_sentiment": abs(pol) if pol < 0 else 0,
        "subjectivity": subj
    }

def get_hashtag_polarities(text):
    """
    Analyzes the sentiment polarity of hashtags in the input text using TextBlob.

    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    ----------
    dict : A dictionary with 'hashtag_count' and 'avg_hashtag_polarity'.

    """
    hashtags = re.findall(r"#(\w+)", text)
    scores = [TextBlob(tag).sentiment.polarity for tag in hashtags]
    return {
        "hashtag_count": len(hashtags),
        "avg_hashtag_polarity": sum(scores)/len(scores) if scores else 0
    }

def sentiment_contrast_features(text):
    """
    Extracts sentiment contrast features from the input text.

    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    ----------
    dict : A dictionary with 'pos_word_count', 'neg_word_count', and 'sentiment_contrast'.

    """
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
    """
    Counts elongated words and multiple punctuation marks in the input text.

    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    ----------
    dict : A dictionary with 'elongated_words' and 'multi_punct'.

    """
    elongated = len(re.findall(r'(.)\1{2,}', text))
    multi_punct = len(re.findall(r'([!?])\1{1,}', text))
    return {"elongated_words": elongated, "multi_punct": multi_punct}

def sarcasm_hashtag_feature(text):
    """
    Checks for the presence of sarcasm-related hashtags in the input text.

    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    ----------
    dict : A dictionary with 'has_sarcasm_tag' indicating presence of sarcasm hashtags.

    """
    sarcasm_tags = re.findall(r"#sarcasm|#irony|#sarcastic", text.lower())
    return {"has_sarcasm_tag": 1 if sarcasm_tags else 0}