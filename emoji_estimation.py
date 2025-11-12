import emoji
from collections import defaultdict

filename = "data/twitter_sentiment_test.csv"

emoji_counts = defaultdict(lambda: {'pos': 0, 'neg': 0})

def extract_emojis(text):
    """Return a list of all emoji characters in text."""
    return [ch for ch in text if ch in emoji.EMOJI_DATA]

# Count positive/negative occurrences
with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            text, label = line.rsplit(",", 1)
            label = int(label)
        except ValueError:
            continue

        for e in extract_emojis(text):
            if label == 1:
                emoji_counts[e]['pos'] += 1
            elif label == 0:
                emoji_counts[e]['neg'] += 1

# Compute polarity scores
emoji_scores = {}
for e, counts in emoji_counts.items():
    total = counts['pos'] + counts['neg']
    if total == 0:
        continue
    positivity = counts['pos'] / total
    negativity = counts['neg'] / total
    compound = (counts['pos'] - counts['neg']) / total  # -1 -> 1
    emoji_scores[e] = {
        'positivity': positivity,
        'negativity': negativity,
        'compound': compound,
        'total': total
    }

# Print results
print(f"{'Emoji':<5} {'Positive':>10} {'Negative':>10} {'Positivity':>10} {'Negativity':>10} {'Total':>10}")
print("-" * 70)
for e, s in sorted(emoji_scores.items(), key=lambda x: x[1]['total'], reverse=True):
    print(f"{e:<5} {emoji_counts[e]['pos']:>10} {emoji_counts[e]['neg']:>10} {s['positivity']:>10.2f} {s['negativity']:>10.2f} {s['total']:>10}")
