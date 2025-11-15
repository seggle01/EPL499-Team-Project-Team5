def get_hashtag_polarities(text: str) -> dict:
    hashtags = re.findall(r"#\w+", text)
    polarities = [TextBlob(tag[1:]).sentiment.polarity for tag in hashtags]
    
    if polarities:
        return {
            "hashtag_avg_polarity": sum(polarities)/len(polarities),
            "hashtag_max_polarity": max(polarities),
            "hashtag_min_polarity": min(polarities),
            "hashtag_count": len(polarities)
        }
    else:
        return {
            "hashtag_avg_polarity": 0,
            "hashtag_max_polarity": 0,
            "hashtag_min_polarity": 0,
            "hashtag_count": 0
        }
