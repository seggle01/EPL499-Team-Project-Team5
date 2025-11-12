
import json
from textblob import TextBlob
import emoji

# Load emoji JSON
with open("../data/emoji_polarity.json", "r", encoding="utf-8") as f:
    emoji_json = json.load(f)

# Load emoticon JSON
with open("../data/unicode_polarity.json", "r", encoding="utf-8") as f:
    emoticon_json = json.load(f)

# Merge both dictionaries
combined_sentiment = {**emoji_json, **emoticon_json}

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
            final_pos = (tb_pos + avg_pos) / 2
            final_neg = (tb_neg + avg_neg) / 2
        else:
            final_pos, final_neg = tb_pos, tb_neg
    else:
        final_pos, final_neg = tb_pos, tb_neg

    return {
        "positive_sentiment": final_pos,
        "negative_sentiment": final_neg,
        "subjectivity": subj
    }
