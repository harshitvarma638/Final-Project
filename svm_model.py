import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

# Function to clean the text (preprocessing)
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char if char.isalnum() else ' ' for char in text])
        return text
    return ''

# Function to load and preprocess the data
def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    
    if not {'Headline', 'Synopsis', 'Final Status'}.issubset(df.columns):
        print("Error: Missing required columns in the dataset.")
        return None
    
    # Combine the headline and synopsis columns into a single text column
    df['text'] = df['Headline'] + " " + df['Synopsis']
    # Clean the text data
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Function to plot the ROC curve
def plot_roc_curve(y_test, y_pred_prob, classes):
    try:
        # Binarize the output labels
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = len(classes)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        plt.figure()
        colors = ['red', 'green', 'blue']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()
    except Exception as e:
        print(f"Error while plotting ROC curve: {e}")

# Main function to train and evaluate the model
def train_and_evaluate_svm(filepath):
    df = load_and_preprocess_data(filepath)
    if df is None:
        return
    
    # Convert the text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])

    # Define the target labels (Final Status)
    y = df['Final Status']
    classes = sorted(y.unique())

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train an SVM classifier
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_model.predict(X_test)
    y_pred_prob = svm_model.predict_proba(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print the classification report
    report = classification_report(y_test, y_pred, target_names=classes)
    print("Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, classes=classes)

    # Plot the ROC curve
    plot_roc_curve(y_test, y_pred_prob, classes=classes)

# Run the training and evaluation with the provided dataset
train_and_evaluate_svm("IndianFinancialNews.csv")