


import pandas as pd

df_train = pd.read_csv('../data/twitter_sentiment_train.csv')
df_test  = pd.read_csv('../data/twitter_sentiment_test.csv')

# Shuffle train set
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)




int_to_label = {1: 'Positive', 0: 'Negative'}

df_train.head(5)




import re
import string
from textblob import TextBlob
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from sklearn.metrics import classification_report

from pre_processing import *


with open('../data/profanity.txt', 'r') as f: 
    profanity_words = f.readlines()
profanity_words = [s.strip() for s in profanity_words]


def count_all_capital_tokens(text: str) -> dict:
    """
    Counts the number of fully capitalized tokens (all letters uppercase) in a given text.
    Returns: {'all_capital_token_count': count}
    """
    matches = re.findall(r'\b[A-Z][A-Z]+\b', text)
    return {'all_capital_token_count': len(matches)}

def count_punctuation(text: str) -> dict:
    """
    Counts the occurrences of each punctuation mark in a given text.
    Returns: {'punctuation_char1': count1, 'punctuation_char2': count2, ...}
    """
    punct_occur = {}
    for char in string.punctuation:
        punct_occur[char] = 0
    for char in text:
        if char in string.punctuation:
            punct_occur[char] += 1
    return punct_occur

def count_profanity_words(text: str, profanity_list: list) -> dict:
    """
    Counts the number of profanity words in a given text using a predefined list.
    Returns: {'profanity_word_count': count}
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

def count_sad_emoticons(text: str):
    """
    Returns the occurrences of sad emoticons.
    Returns: {
      "sad_emoticon": count
    }
    """
    # Sad, crying, angry, and negative emoticons
    matches = re.findall(r':\(|:\||:\/|:\\|:\'\(|>:\(|D:|:<|:c|;\(|T_T|T\.T', text)
    return {"sad_emoticon": len(matches)}

def count_happy_emoticons(text: str):
    """
    Returns the occurrences of happy emoticons.
    Returns: {
      "happy_emoticon": count
    }
    """
    # Happy, excited, laughing, and positive emoticons
    matches = re.findall(r':\)|:D|;D|=\)|;-\)|:\}\)|:>|=\]|8\)|;-D|XD|xD|x-D|X-D|<3|:\*|;-\*|;\)|=D', text)
    return {"happy_emoticon": len(matches)}


# TextBlob does not work on emojis !!!!
def get_sentiment_and_subjectivity(text: str) -> dict:
    """
    Returns the sentiment polarity and subjectivity scores of a given text using TextBlob.
    Returns: {
      "positive_sentiment": polarity if > 0, else 0,
      "negative_sentiment": |polarity| if < 0, else 0,
      "subjectivity": score
    }
    """
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    
    return {
        "positive_sentiment": pol if pol > 0 else 0,
        "negative_sentiment": abs(pol) if pol < 0 else 0,
        "subjectivity": subj
    }
    



def preprocessing_text(text):
    # Classical preprocessing steps
    text = twokenize.tokenizeRawTweetText(text)
    text = to_lower(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text


feature_functions = [
    count_punctuation,
    lambda text: count_profanity_words(text, profanity_words),
    count_all_capital_tokens,
    count_sad_emoticons,
    count_happy_emoticons,
    get_sentiment_and_subjectivity,
    
]

X_train = df_train.drop(columns='label')
X_test = df_test.drop(columns='label')

for func in tqdm(feature_functions):
    results = X_train['text'].apply(lambda x: func(str(x))).tolist()
    temp_df = pd.DataFrame(results)

    temp_df.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_train = pd.concat([X_train, temp_df], axis=1)

    results = X_test['text'].apply(lambda x: func(str(x))).tolist()
    temp_df = pd.DataFrame(results)

    temp_df.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    X_test = pd.concat([X_test, temp_df], axis=1)

X_train.head(5)


X_train_features = X_train.drop(columns=['text'])
X_test_features = X_test.drop(columns=['text'])

y_train = df_train.drop(columns=['text'])
y_test = df_test.drop(columns=['text'])

model = SGDClassifier(
    loss='log_loss',
    learning_rate='constant',
    eta0=0.01,
    random_state=123
)

model.fit(X_train_features, y_train)
y_pred = model.predict(X_test_features)

# %%
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
# Train a Random Forest on your features
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_features, y_train)
# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X_train_features.columns,
    'importance': model_rf.feature_importances_
})

# Sort by importance
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Display top 25 features
print("\nTop 25 Most Important Features:")
print(feature_importances.head(25))

# Plot top 25 features
plt.figure(figsize=(10, 8))
top_25 = feature_importances.head(25)
plt.barh(range(len(top_25)), top_25['importance'].values,color='skyblue')
plt.yticks(range(len(top_25)), top_25['feature'].values)
plt.xlabel('Importance')
plt.title('Top 25 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


