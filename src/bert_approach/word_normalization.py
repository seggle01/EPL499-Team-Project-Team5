import re
import nltk, random
import emoji 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List, Callable
import html
import string
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

def remove_chars(tokens):
    return [token for token in tokens if len(token)>1]

def preprocessing_text(text):
    # Word normalization
    text = clean_unicode(text)
    text = uncontract(text)
    text = convert_urls_emails(text)
    # Classical preprocessing steps
    tokens = word_tokenize_sentence(text)
    tokens = to_lower(tokens)
    tokens = remove_punct_digits(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    # tokens = lemmatize(tokens)
    # tokens = remove_chars(tokens)
    return tokens