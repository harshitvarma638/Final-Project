# svm_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from utils import preprocess_text, print_classification_results, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df["combined_text"] = (df["Headline"].astype(str) + " " + df["Synopsis"].astype(str)).apply(preprocess_text)
    return df

def train_and_evaluate_model(data_path):
    df = load_and_preprocess(data_path)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["combined_text"])
    
    # Assuming you have a labeled dataset with positive, negative, and neutral tags
    y = df["Final Status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm_clf = SVC(kernel="linear", probability=True)
    svm_clf.fit(X_train, y_train)
    
    y_pred = svm_clf.predict(X_test)
    print_classification_results(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    
    # Print accuracy, precision, and F1 score
    report = classification_report(y_test, y_pred, target_names=['Positive', 'Negative'])
    print(report)
    
    # Plot ROC curve for multi-class classification
    y_test_bin = label_binarize(y_test, classes=['Positive', 'Negative'])
    y_prob = svm_clf.predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    for i, color in zip(range(y_test_bin.shape[1]), ['red', 'green']):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-class')
    plt.legend(loc="lower right")
    plt.show()

train_and_evaluate_model("IndianFinancialNews.csv")
