import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import preprocess_text, print_classification_results, plot_confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

def load_and_preprocess(path):
    """
    Loads a CSV with no header.
    Assumes:
      - Column 0: label (string)
      - Column 1: text (string)
    """
    # Read CSV without header and assign column names
    # Try 'ISO-8859-1' or 'cp1252' if UTF-8 fails
    df = pd.read_csv(path, header=None, names=["label", "text"], encoding="ISO-8859-1")
    
    # Clean text using your existing preprocess_text function
    df["clean_text"] = df["text"].astype(str).apply(preprocess_text)
    return df

def prepare_data(df, max_features=5000, maxlen=100):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df["clean_text"])
    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    X = pad_sequences(sequences, maxlen=maxlen)

    # Example label mapping for 3 classes (negative, neutral, positive)
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    
    # Clean up label column: strip spaces, handle missing, map to numeric
    df["label"] = df["label"].astype(str).str.strip()
    df["label"] = df["label"].replace("", np.nan)
    df["label"] = df["label"].fillna("neutral")  # default if missing
    df["label"] = df["label"].map(label_mapping)
    df["label"] = df["label"].fillna(1).astype(int)  # default any unmapped to 'neutral'
    
    y = df["label"].values
    
    return X, y, tokenizer

def build_lstm_model(max_features=5000, maxlen=100):
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=128, input_length=maxlen),
        SpatialDropout1D(0.3),  # Increased dropout to help with regularization

        LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),  # Added BatchNormalization
        
        LSTM(64, dropout=0.3, recurrent_dropout=0.3,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),  # Added BatchNormalization

        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(3, activation='softmax')  # for 3-class classification
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1}: "
            f"loss = {logs['loss']:.4f}, "
            f"accuracy = {logs['accuracy']:.4f}, "
            f"val_loss = {logs['val_loss']:.4f}, "
            f"val_accuracy = {logs['val_accuracy']:.4f}"
        )

    def on_batch_end(self, batch, logs=None):
        print(
            f"Batch {batch + 1}: "
            f"loss = {logs['loss']:.4f}, "
            f"accuracy = {logs['accuracy']:.4f}"
        )

if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess("all-data.csv")
    X, y, tokenizer = prepare_data(df)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build and summarize the model
    model = build_lstm_model(max_features=5000, maxlen=100)
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[TrainingProgressCallback()]
    )
    
    # Plot training history
    plot_history(history)
    
    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print_classification_results(y_test, y_pred, target_names=["negative", "neutral", "positive"])
    plot_confusion_matrix(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"])
    print(report)