
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Callable
from twokenize import twokenize
from word_normalization import *


# Monkey patch the broken function
def normalizeTextForTagger(text):
    import html
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

def word_tokenize_sentence(sent: str):
    """
    Tokenizes a sentence into words using the twokenize library.
    
    Parameters
    ----------
    sent : str
        The input sentence to be tokenized.

    Returns
    ----------
    List[str] : A list of tokens extracted from the sentence.
        
    """
    
    return twokenize.tokenizeRawTweetText(sent)

def to_lower(tokens: List[str]):
    """
    Converts all tokens in the list to lowercase.

    Parameters
    ----------
    tokens : List[str]
        A list of string tokens.

    Returns
    ----------  
    List[str] : A list of tokens converted to lowercase.    

    """
    return [token.lower() for token in tokens]

def remove_punct_digits(tokens: List[str]):
    """
    Removes tokens that contain non-alphabetic characters.

    Parameters
    ----------
    tokens : List[str]
        A list of string tokens.

    Returns
    ----------
    List[str] : A list of tokens with only alphabetic characters.    

    """
    return [token for token in tokens if token.isalpha()]

def remove_stopwords(tokens: List[str]):
    """
    Removes common English stopwords from the list of tokens.
    
    Parameters
    ----------
    tokens : List[str]
        A list of string tokens.

    Returns
    ----------
    List[str] : A list of tokens with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize(tokens: List[str]):
    """
    Lemmatizes each token in the list using WordNetLemmatizer.

    Parameters
    ----------
    tokens : List[str]
        A list of string tokens.
    
    Returns
    ----------
    List[str] : A list of lemmatized tokens.

    """
    return [_lem.lemmatize(token) for token in tokens]

def stem(tokens: List[str]):
    """
    Stems each token in the list using PorterStemmer.

    Parameters
    ----------
    tokens : List[str]
        A list of string tokens.
   
    Returns
    ----------
    List[str] : A list of stemmed tokens.

    """
    return [_stem.stem(token) for token in tokens]

def apply_steps_to_sentence(sent: str, steps: List[Callable]):
    """
    Applies a series of preprocessing steps to a single sentence.

    Parameters
    ----------
    sent : str
        The input sentence to be processed.
    steps : List[Callable]
        A list of functions representing the preprocessing steps to be applied. 

    Returns
    ----------
    List[str] : The processed list of tokens after applying all steps.
    
    """
    tokens = sent
    for step in steps:
        tokens = step(tokens)
    return tokens

def run_pipeline(text: str, steps: List[Callable]):
    """
    Runs a preprocessing pipeline on the input text.

    Parameters
    ----------
    text : str
        The input text to be processed. 
    steps : List[Callable]
        A list of functions representing the preprocessing steps to be applied.

    Returns
    ----------
    List[List[str]] : A list of lists, where each inner list contains the processed tokens of a sentence.

    """
    sentences              = nltk.sent_tokenize(text)
    preprocessed_sentences = []

    for sent in sentences:
        tokenized_sentence = apply_steps_to_sentence(sent, steps)
        preprocessed_sentences.append(tokenized_sentence)

    return preprocessed_sentences

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