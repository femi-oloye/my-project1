# i will use sklearn to train a logistic regression model
# i will use TF-IDF to vectorize the text data

# TF-IDF stands for term frequency-inverse document frequency. It is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
# The importance increases proportionally to the number of times a word appears in the document and is offset by the frequency of the word in the corpus.

import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #type: ignore

import os
import sys
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

from preprocess_data import preprocessData # type: ignore

def logisticRegression():
    data = preprocessData()
    # split the data into test and training sets
    X_train, X_test, y_train, y_test =  train_test_split(data["tweet"], data["target"], test_size=0.2, random_state=42, stratify=data["target"])

    # vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_Tfidf = vectorizer.fit_transform(X_train)
    X_test_Tfidf = vectorizer.transform(X_test)

    # train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_Tfidf, y_train)

    # model evaluation
    y_pred = model.predict(X_test_Tfidf)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Save the model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "/home/oluwafemi/AI-ENGINE/.venv/models/logistic_model.joblib")
    joblib.dump(vectorizer, "/home/oluwafemi/AI-ENGINE/.venv/models/vectorizer.joblib")
    print("✅ Model and vectorizer saved to models/")


    return model, vectorizer


if __name__ == "__main__":
    logisticRegression()