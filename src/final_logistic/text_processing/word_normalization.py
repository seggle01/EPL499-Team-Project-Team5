import re

def uncontract(text):
    """
    Expands common English contractions in the input text.

    Parameters
    ----------
    text : str
        The input text containing contractions.

    Returns
    ----------
    str : The text with contractions expanded.

    """
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
    """
    Replaces URLs and email addresses in the input text with placeholder tokens.

    Parameters
    ----------
    text : str
        The input text containing URLs and email addresses.

    Returns
    ----------
    str : The text with URLs replaced by 'URL' and email addresses replaced by 'EMAIL'.

    """
    url_regex_1 = r'^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$'
    url_regex_2 = r'^[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$'
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    text = re.sub(url_regex_1, 'URL', text)
    text = re.sub(url_regex_2, 'URL', text)
    text = re.sub(email_regex, 'EMAIL', text)
    return text

def clean_unicode(text: str):
    """
    Replaces specific unicode escape sequences in the input text with their corresponding characters.

    Parameters
    ----------
    text : str
        The input text containing unicode escape sequences.

    Returns
    ----------
    str : The text with unicode escape sequences replaced by their corresponding characters.

    """
    text = re.sub(r'\\u2019', "'", text)
    text = re.sub(r'\\u201c', '"', text)
    text = re.sub(r'\\u201d', '"', text)
    text = re.sub(r'\\u002c', ',', text)
        
    return text

def remove_numbers(text: str):
    """
    Removes all numeric digits from the input text.

    Parameters
    ----------
    text : str
        The input text containing numeric digits.

    Returns
    ----------
    str : The text with numeric digits removed.
    
    """
    return re.sub(r'[0-9]','',text)
