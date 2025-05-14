import streamlit as st
import pandas as pd
import joblib
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Adjust path to import preprocessing function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
from preprocess_data import clean_text  # Make sure this function exists # type: ignore

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("/home/oluwafemi/sentiment-analysis-streamlit/.venv/sentiment-analysis-streamlit/models/logistic_model.joblib")
    vectorizer = joblib.load("/home/oluwafemi/sentiment-analysis-streamlit/.venv/sentiment-analysis-streamlit/models/vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()

# Define prediction function for LIME
def predict_proba(texts):
    cleaned_texts = [clean_text(t) for t in texts]
    transformed = vectorizer.transform(cleaned_texts)
    return model.predict_proba(transformed)

# Streamlit UI
st.title("🔍 Sentiment Classifier for Tweets/Reviews")
st.write("This app uses a Logistic Regression model to classify text as Positive or Negative.")
st.markdown("---")

# Input method
option = st.radio("Choose input method:", ("Single Text Input", "Upload CSV"))

if option == "Single Text Input":
    user_input = st.text_area("Enter a tweet or review:")
    
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Predict
            cleaned = clean_text(user_input)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]
            label = "Positive 😊" if prediction == 1 else "Negative 😠"
            st.success(f"Prediction: {label}")

            # Explain with LIME
            explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
            explanation = explainer.explain_instance(
                user_input, predict_proba, num_features=10
            )
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
            df["clean_text"] = df["text"].apply(clean_text)
            df["prediction"] = model.predict(vectorizer.transform(df["clean_text"]))
            df["label"] = df["prediction"].apply(lambda x: "Positive" if x == 1 else "Negative")

            st.write("🔹 Predictions Preview:")
            st.dataframe(df[["text", "label"]].head())

            st.write("📊 Sentiment Breakdown:")
            st.bar_chart(df["label"].value_counts())

            sentiment_filter = st.selectbox("Filter predictions:", ["All", "Positive", "Negative"])
            if sentiment_filter != "All":
                df_filtered = df[df["label"] == sentiment_filter]
            else:
                df_filtered = df

            with st.expander("🔍 Filtered Predictions"):
                st.dataframe(df_filtered[["text", "label"]])
