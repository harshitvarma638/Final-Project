import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import string

# Load dataset
file_path = "./all-data.csv"
df = pd.read_csv(file_path, header=None, encoding="ISO-8859-1")

# Assign column names explicitly
df.columns = ["label", "text"]

# Display first few rows to verify correctness
df.head()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df["clean_text"] = df["text"].astype(str).apply(preprocess_text)

# Tokenization & Padding
max_features = 5000  # Increase vocabulary size
maxlen = 50  # Increase sequence length

tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_text"])
sequences = tokenizer.texts_to_sequences(df["clean_text"])
X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

# Encode labels
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["label"].astype(str).str.strip().map(label_mapping).fillna(1).astype(int)
y = df["label"].values

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Architecture (Bidirectional LSTM + Regularization)
model = Sequential([
    Embedding(input_dim=max_features, output_dim=64, input_length=maxlen),
    SpatialDropout1D(0.3),  # Helps with overfitting
    Bidirectional(LSTM(64, return_sequences=True)),  
    GlobalMaxPool1D(),  
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer="l2"),
    Dropout(0.3),
    Dense(3, activation='softmax')  
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training with EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Validation")
    plt.title("Model Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Validation")
    plt.title("Model Loss")
    plt.legend()
    plt.show()

plot_history(history)

# Evaluate Model
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "neutral", "positive"], yticklabels=["negative", "neutral", "positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(y_test, y_pred)
