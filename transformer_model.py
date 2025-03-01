# transformer_model.py
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import preprocess_text, print_classification_results, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load pre-trained models (make sure to have the models downloaded)
print("Loading FinBERT model...")
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
print("FinBERT model loaded.")

print("Loading TweetBERT model...")
tweetbert_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tweetbert_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
print("TweetBERT model loaded.")

def predict_finbert(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return ["negative", "neutral", "positive"][np.argmax(scores)]

def predict_tweetbert(text):
    inputs = tweetbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = tweetbert_model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return ["negative", "neutral", "positive"][np.argmax(scores)]

def run_transformer_model(df, source="news", batch_size=256):
    """
    Run the transformer model on the provided dataframe in batches.
    Expects a 'clean_text' column and a ground truth 'label' column.
    """
    print(f"Running transformer model on {source} data...")
    print(f"Starting predictions for {source} data in batches of {batch_size}...")

    predictions = []
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_texts = df["clean_text"][start:end].tolist()
        
        if source == "news":
            inputs = finbert_tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = finbert_model(**inputs)
            batch_preds = torch.softmax(outputs.logits, dim=1).argmax(dim=1).tolist()
            batch_preds = ["negative" if pred == 0 else "neutral" if pred == 1 else "positive" for pred in batch_preds]
        else:
            inputs = tweetbert_tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = tweetbert_model(**inputs)
            batch_preds = torch.softmax(outputs.logits, dim=1).argmax(dim=1).tolist()
            batch_preds = ["negative" if pred == 0 else "neutral" if pred == 1 else "positive" for pred in batch_preds]
        
        predictions.extend(batch_preds)
        print(f"Processed batch {start // batch_size + 1}/{(len(df) + batch_size - 1) // batch_size}")
        print(f"Batch results: {batch_preds}")

    df["pred_trans"] = predictions
    print(f"Predictions completed for {source} data.")
    
    # Print classification report and confusion matrix if labels exist
    if "label" in df.columns:
        print("Generating classification report and confusion matrix...")
        print_classification_results(df["label"], df["pred_trans"])
        plot_confusion_matrix(df["label"], df["pred_trans"])
        print("Classification report and confusion matrix generated.")
    
    # Calculate and print accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(df["label"], df["pred_trans"])
    precision = precision_score(df["label"], df["pred_trans"], average='weighted')
    recall = recall_score(df["label"], df["pred_trans"], average='weighted')
    f1 = f1_score(df["label"], df["pred_trans"], average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print(f"Transformer model run completed for {source} data.")
    return df

if __name__ == "__main__":
    print("Loading news data...")
    news_df = pd.read_csv("IndianFinancialNews.csv")
    news_df["clean_text"] = news_df["Headline"].astype(str).apply(preprocess_text)
    print("News data loaded and preprocessed.")
    
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(news_df, test_size=0.2, random_state=42)
    
    # Run the transformer model on the training set
    train_df = run_transformer_model(train_df, source="news")
    
    # Evaluate the model on the testing set
    test_df = run_transformer_model(test_df, source="news")
    print("Classification report for news data:")
    print(classification_report(test_df["label"], test_df["pred_trans"]))
    
    print("Loading tweets data...")
    tweets_df = pd.read_csv("tweets.csv")
    tweets_df["clean_text"] = tweets_df["tweet"].astype(str).apply(preprocess_text)
    print("Tweets data loaded and preprocessed.")
    
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(tweets_df, test_size=0.2, random_state=42)
    
    # Run the transformer model on the training set
    train_df = run_transformer_model(train_df, source="tweet")
    
    #