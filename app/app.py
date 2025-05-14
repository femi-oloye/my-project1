import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# 1) Import your cleaning & data loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
from preprocess_data import clean_text, preprocessData  # type: ignore

# ─── 2) Load & cache models ─────────────────────────────────────────────────────

@st.cache_resource
def load_logistic():
    m = joblib.load("/home/oluwafemi/AI-ENGINE/.venv/models/logistic_model.joblib")
    v = joblib.load("/home/oluwafemi/AI-ENGINE/.venv/models/vectorizer.joblib")
    return m, v

@st.cache_resource
def load_bert():
    m = DistilBertForSequenceClassification.from_pretrained("/home/oluwafemi/AI-ENGINE/bert_sentiment_model")
    t = DistilBertTokenizerFast.from_pretrained("/home/oluwafemi/AI-ENGINE/bert_sentiment_model")
    return m, t

log_model, log_vect = load_logistic()
bert_model, bert_tok = load_bert()

# ─── 3) Prediction helpers ──────────────────────────────────────────────────────

def predict_logistic(texts):
    cleaned = [clean_text(t) for t in texts]
    X       = log_vect.transform(cleaned)
    probs   = log_model.predict_proba(X)
    preds   = log_model.predict(X)
    return probs, preds

def predict_bert(texts):
    enc    = bert_tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        logits = bert_model(**enc).logits
    probs  = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    preds  = np.argmax(probs, axis=1)
    return probs, preds

def lime_explain(text, predict_fn, class_names):
    expl = LimeTextExplainer(class_names=class_names)
    exp  = expl.explain_instance(text, predict_fn, num_features=10)
    return exp.as_list()

# ─── 4) Cached training metrics ─────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def compute_train_metrics_logistic(sample_size=2000):
    df = preprocessData().dropna(subset=["tweet","target"])
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    texts = df["tweet"].astype(str).tolist()
    y_true= df["target"].values
    _, y_pred = predict_logistic(texts)
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        confusion_matrix(y_true, y_pred)
    )

@st.cache_data(ttl=3600)
def compute_train_metrics_bert(sample_size=500):
    df = preprocessData().dropna(subset=["tweet","target"])
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    texts = df["tweet"].astype(str).tolist()
    y_true= df["target"].values
    _, y_pred = predict_bert(texts)
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        confusion_matrix(y_true, y_pred)
    )

# ─── 5) Streamlit layout ────────────────────────────────────────────────────────

st.title("🧠 Sentiment Classifier: Logistic vs. BERT")
st.write("Compare models, view training metrics, and inspect LIME explanations.")

model_choice = st.selectbox("Select Model", ["Logistic Regression", "BERT (DistilBERT)"])
input_mode   = st.radio("Input Mode", ["Single Text", "Upload CSV"])

# ── Training Metrics ───────────────────────────────────────────────────────────
if st.checkbox("Show Training‑Set Metrics"):
    st.subheader("📊 Training‑Set Evaluation")
    with st.spinner("Computing (cached)…"):
        if model_choice == "Logistic Regression":
            acc, f1, cm = compute_train_metrics_logistic()
        else:
            acc, f1, cm = compute_train_metrics_bert()

    st.write(f"**Accuracy:** {acc:.4f} **F1 Score:** {f1:.4f}")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)

# ── Single Text Mode ────────────────────────────────────────────────────────────
if input_mode == "Single Text":
    text = st.text_area("Enter text to classify")
    if st.button("Predict"):
        if not text.strip():
            st.warning("Type something first.")
        else:
            if model_choice == "Logistic Regression":
                probs, preds = predict_logistic([text])
                predict_fn = lambda x: predict_logistic(x)[0]
            else:
                probs, preds = predict_bert([text])
                predict_fn = lambda x: predict_bert(x)[0]

            pred = preds[0]
            label = "Positive 😊" if pred == 1 else "Negative 😠"
            st.success(f"Prediction: {label}")

            st.subheader("🔍 LIME Explanation")
            explanation = lime_explain(text, predict_fn, ["Negative", "Positive"])
            # Prepare data for plotting
            words, weights = zip(*explanation)
            colors = ['green' if w > 0 else 'red' for w in weights]

            # Plot the horizontal bar chart
            fig, ax = plt.subplots()
            ax.barh(words, weights, color=colors)
            ax.set_xlabel("Contribution to Prediction")
            ax.set_title("LIME Explanation")
            ax.axvline(0, color='black', linewidth=0.5)  # vertical line at zero for reference
            st.pyplot(fig)

# ── Batch CSV Mode ──────────────────────────────────────────────────────────────
else:
    up = st.file_uploader("Upload CSV with 'text' column (optional 'target')", type="csv")
    if up:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            texts = df["text"].astype(str).tolist()
            _, preds = (predict_logistic(texts) if model_choice == "Logistic Regression"
                        else predict_bert(texts))
            df["prediction"] = preds
            df["label"]      = df["prediction"].map({0:"Negative",1:"Positive"})

            st.subheader("🔹 Predictions Preview")
            st.dataframe(df[["text","label"]].head())

            st.subheader("📊 Sentiment Breakdown")
            st.bar_chart(df["label"].value_counts())

            if "target" in df.columns:
                y_true = df["target"].values
                y_pred = preds
                acc    = accuracy_score(y_true, y_pred)
                f1     = f1_score(y_true, y_pred)
                cm2    = confusion_matrix(y_true, y_pred)

                st.subheader("📈 Evaluation on Uploaded Data")
                st.write(f"**Accuracy:** {acc:.4f} **F1 Score:** {f1:.4f}")
                fig2, ax2 = plt.subplots()
                sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"], ax=ax2)
                st.pyplot(fig2)

            st.subheader("🔍 Filter & Search")
            sentiment_filter = st.selectbox("Filter by Sentiment", ["All","Positive","Negative"])
            df_filt = df if sentiment_filter == "All" else df[df["label"] == sentiment_filter]
            search = st.text_input("Search text")
            if search:
                df_filt = df_filt[df_filt["text"].str.contains(search, case=False, na=False)]
            st.dataframe(df_filt[["text", "label"]])
