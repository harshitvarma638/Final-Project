# utils.py
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix

def preprocess_text(text):
    """
    Clean and preprocess text: remove URLs, mentions, hashtags, punctuation,
    convert to lowercase and strip extra whitespace.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = text.lower().strip()
    return text

def plot_sentiment_distribution(df, label_col="label", title="Sentiment Distribution"):
    plt.figure(figsize=(6,4))
    sns.countplot(x=label_col, data=df, order=["negative", "neutral", "positive"])
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

def plot_text_length_distribution(df, text_col="clean_text", title="Text Length Distribution"):
    df['text_length'] = df[text_col].apply(lambda x: len(x.split()))
    plt.figure(figsize=(6,4))
    sns.histplot(df['text_length'], bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Number of words")
    plt.show()

def plot_text_length_by_sentiment(df, text_col="clean_text", label_col="label"):
    df['text_length'] = df[text_col].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8,4))
    sns.boxplot(x=label_col, y='text_length', data=df, order=["negative", "neutral", "positive"])
    plt.title("Text Length by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of words")
    plt.show()

def generate_wordcloud(df, text_col="clean_text", title="Word Cloud"):
    text = " ".join(df[text_col].tolist())
    wc = WordCloud(stopwords=set(stopwords.words("english")),
                   background_color="white",
                   width=800,
                   height=400).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

def print_classification_results(y_true, y_pred, target_names=["negative", "neutral", "positive"]):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
def plot_confusion_matrix(y_true, y_pred, classes=["negative", "neutral", "positive"], title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()