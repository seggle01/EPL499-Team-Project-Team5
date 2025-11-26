
import nltk, random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from typing import List, Callable

random.seed(11)

import html
from twokenize import twokenize
# Monkey patch the broken function
def normalizeTextForTagger(text):
    text = text.replace("&amp;", "&")
    text = html.unescape(text)  # Use html.unescape instead
    return text

twokenize.normalizeTextForTagger = normalizeTextForTagger

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

_lem  = WordNetLemmatizer()
_stem = PorterStemmer()

def word_tokenize_sentence(sent: str) -> List[str]:
    return twokenize.tokenizeRawTweetText(sent)

def to_lower(tokens: List[str]) -> List[str]:
    return [token.lower() for token in tokens]

def remove_punct_digits(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token.isalpha()]

def remove_stopwords(tokens: List[str]) -> List[str]:
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize(tokens: List[str]) -> List[str]:
    return [_lem.lemmatize(token) for token in tokens]

def stem(tokens: List[str]) -> List[str]:
    return [_stem.stem(token) for token in tokens]

def apply_steps_to_sentence(sent: str, steps: List[Callable]) -> List[str]:
    # Apply the provided list of functions in order
    tokens = sent
    for step in steps:
        tokens = step(tokens)
    return tokens

def run_pipeline(text: str, steps: List[Callable]) -> List[List[str]]:
    """
    Processes a single text string by first tokenizing it into sentences,
    then applying the specified pipeline steps to each sentence.
    """
    sentences              = nltk.sent_tokenize(text)
    preprocessed_sentences = []

    for sent in sentences:
        tokenized_sentence = apply_steps_to_sentence(sent, steps)
        preprocessed_sentences.append(tokenized_sentence)

    return preprocessed_sentences
