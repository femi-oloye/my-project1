import streamlit as st
import pandas as pd
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from lime.lime_text import LimeTextExplainer
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the fine-tuned BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    model = DistilBertForSequenceClassification.from_pretrained("/home/oluwafemi/sentiment-analysis-streamlit/bert_sentiment_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("/home/oluwafemi/sentiment-analysis-streamlit/bert_sentiment_model")
    return model, tokenizer

model, tokenizer = load_bert_model()

# Step 2: Define prediction function for BERT
def predict_sentiment(texts):
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        logits = model(**encodings).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()  # Convert logits to probabilities
    return probabilities

# Step 3: Define LIME explainer
def explain_with_lime(text):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    explanation = explainer.explain_instance(text, predict_sentiment, num_features=10)
    return explanation

# Step 4: Streamlit UI for interaction
st.title("🔍 BERT Sentiment Classifier for Tweets/Reviews")
st.write("This app uses a fine-tuned BERT model to classify text as Positive or Negative.")

option = st.radio("Choose input method:", ("Single Text Input", "Upload CSV"))

if option == "Single Text Input":
    user_input = st.text_area("Enter a tweet or review:")
    if st.button("Predict"):
        probabilities = predict_sentiment([user_input])[0]
        label = "Positive 😊" if np.argmax(probabilities) == 1 else "Negative 😠"

        st.success(f"Prediction: {label}")

        # LIME explanation
        explanation = explain_with_lime(user_input)

        # Plotting LIME explanation as bar plot
        weights = explanation.as_list()
        words, importances = zip(*weights)
        colors = ['green' if w > 0 else 'red' for w in importances]

        fig, ax = plt.subplots()
        ax.barh(words, importances, color=colors)
        ax.set_xlabel("Contribution to prediction")
        ax.set_title("LIME Explanation")
        ax.invert_yaxis()
        st.markdown("🟢 Words pushing toward Positive | 🔴 Words pushing toward Negative")
        st.pyplot(fig)

else:
    uploaded_file = st.file_uploader("Upload CSV with a column named 'text'", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            # Get predictions for the CSV
            probabilities = predict_sentiment(df["text"].tolist())
            predictions = np.argmax(probabilities, axis=1)  # Get class labels (0 or 1)
            df["label"] = ["Positive" if label == 1 else "Negative" for label in predictions]
            df["prediction"] = predictions

            st.write("🔹 Predictions Preview:")
            st.dataframe(df[["text", "label"]].head())

            # Sentiment Breakdown
            st.write("📊 Sentiment Breakdown:")
            st.bar_chart(df["label"].value_counts())

            # Evaluation metrics
            y_true = df["label"].apply(lambda x: 1 if x == "Positive" else 0).values
            y_pred = df["prediction"]

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            st.write("📊 Evaluation Report:")
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"F1 Score: {f1:.4f}")

            # Confusion matrix heatmap
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
