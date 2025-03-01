# main.py
import pandas as pd
from utils import plot_sentiment_distribution, plot_text_length_distribution, plot_text_length_by_sentiment, generate_wordcloud, preprocess_text
from lstm_model import build_lstm_model
from svm_model import train_and_evaluate_model as run_svm_model
from transformer_model import run_transformer_model

# Loading the necessary dataset
financial_news = pd.read_csv("IndianFinancialNews.csv")
tweets = pd.read_csv("tweets.csv")

# Preprocess texts
financial_news["clean_text"] = financial_news["Title"].astype(str).apply(preprocess_text)
tweets["clean_text"] = tweets["tweet"].astype(str).apply(preprocess_text)

# Visualize overall analysis
print("Financial News Analysis:")
plot_sentiment_distribution(financial_news)
plot_text_length_distribution(financial_news)
plot_text_length_by_sentiment(financial_news)
generate_wordcloud(financial_news)

print("Tweets Analysis:")
plot_sentiment_distribution(tweets)
plot_text_length_distribution(tweets)
plot_text_length_by_sentiment(tweets)
generate_wordcloud(tweets)

# Importing the models and running them on the datasets

# transformer model
financial_news = run_transformer_model(financial_news, source="news")
tweets = run_transformer_model(tweets, source="tweet")

# # LSTM model
financial_news = build_lstm_model(financial_news, source="news")
tweets = build_lstm_model(tweets, source="tweet")

# # SVM model
financial_news = run_svm_model(financial_news, source="news")
tweets = run_svm_model(tweets, source="tweet")
