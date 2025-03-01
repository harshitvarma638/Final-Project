# data_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")

from utils import preprocess_text, plot_sentiment_distribution, plot_text_length_distribution, plot_text_length_by_sentiment, generate_wordcloud

def load_dataset(path, text_col="text", source_label="unknown"):
    """
    Load a dataset from CSV, preprocess the text, and add a source label.
    """
    df = pd.read_csv(path)
    df["clean_text"] = df[text_col].astype(str).apply(preprocess_text)
    df["source"] = source_label
    return df

def basic_stats(df, dataset_name="Dataset"):
    """
    Print basic statistics for the given DataFrame.
    """
    print(f"--- {dataset_name} Statistics ---")
    print("Total samples:", len(df))
    if "label" in df.columns:
        print("Sentiment distribution:")
        print(df["label"].value_counts())
    # Average text length
    df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))
    print("Average text length (words):", df["text_length"].mean())
    print("Median text length (words):", df["text_length"].median())
    print("\n")

def top_n_words(df, n=20):
    """
    Compute and display the top n most frequent words in the dataset.
    """
    stop_words = set(stopwords.words("english"))
    all_words = " ".join(df["clean_text"]).split()
    # Filter out stopwords and short tokens
    filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(n)
    
    # Plot top words
    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title("Top {} Words in the Dataset".format(n))
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()

def ngram_analysis(df, n=2, top_n=20):
    """
    Compute and plot the most common n-grams.
    """
    from nltk.util import ngrams
    stop_words = set(stopwords.words("english"))
    all_ngrams = []
    for text in df["clean_text"]:
        tokens = [token for token in text.split() if token not in stop_words]
        ngram_tokens = list(ngrams(tokens, n))
        all_ngrams.extend(ngram_tokens)
    
    ngram_counts = Counter(all_ngrams)
    top_ngrams = ngram_counts.most_common(top_n)
    
    ngram_labels = [" ".join(gram) for gram, _ in top_ngrams]
    counts = [count for _, count in top_ngrams]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts, y=ngram_labels, palette="magma")
    plt.title("Top {} {}-grams".format(top_n, n))
    plt.xlabel("Frequency")
    plt.ylabel("{}-gram".format(n))
    plt.show()

def run_analysis():
    # Load datasets - update the paths to your CSV files
    news_path = "IndianFinancialNews.csv" 
    tweets_path = "tweets.csv"
    
    news_df = load_dataset(news_path, text_col="Title", source_label="Title")
    tweets_df = load_dataset(tweets_path, text_col="tweet", source_label="sentiment")
    
    basic_stats(news_df, dataset_name="Financial News")
    basic_stats(tweets_df, dataset_name="Tweets")
    
    # Visualizations for Financial News
    print("Visualizations for Financial News:")
    plot_sentiment_distribution(news_df, title="Financial News Sentiment Distribution")
    plot_text_length_distribution(news_df, title="Financial News Text Length Distribution")
    plot_text_length_by_sentiment(news_df)
    generate_wordcloud(news_df, title="Financial News Word Cloud")
    top_n_words(news_df, n=20)
    ngram_analysis(news_df, n=2, top_n=15)
    
    # Visualizations for Tweets
    print("Visualizations for Tweets:")
    plot_sentiment_distribution(tweets_df, title="Tweet Sentiment Distribution")
    plot_text_length_distribution(tweets_df, title="Tweet Text Length Distribution")
    plot_text_length_by_sentiment(tweets_df)
    generate_wordcloud(tweets_df, title="Tweet Word Cloud")
    top_n_words(tweets_df, n=20)
    ngram_analysis(tweets_df, n=2, top_n=15)
    
    # Combined dataset analysis
    combined_df = pd.concat([news_df, tweets_df], ignore_index=True)
    print("Combined Dataset Statistics:")
    basic_stats(combined_df, dataset_name="Combined Dataset")
    print("Combined Dataset Visualizations:")
    plot_sentiment_distribution(combined_df, title="Combined Sentiment Distribution")
    generate_wordcloud(combined_df, title="Combined Word Cloud")

if __name__ == "__main__":
    run_analysis()
