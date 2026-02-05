import nltk 

#dependencies for sentiment analysis
# nltk.download('vader_lexicon')
# nltk.download('punkt')  

from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer   

text = input("Enter the text to analyze sentiment: ")
tokens = word_tokenize(text)

def analyze_sentiment(message):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(message)
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    

text = input("Enter the text to analyze sentiment: ")
result = analyze_sentiment(text)
print(f"Sentiment: {result}")