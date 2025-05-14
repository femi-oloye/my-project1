# 🔍 Sentiment Insight App – Compare Logistic Regression & BERT

An interactive Streamlit app for classifying sentiment in tweets and customer reviews using two models: Logistic Regression and fine-tuned BERT. Built for NLP explainability, analysis, and deployment.

![App Screenshot](insert_app_screenshot.png) <!-- Optional: Add screenshot -->

---

## 📌 Features

- ✅ Compare Logistic Regression vs. BERT model performance
- 📝 Input single text or upload a CSV file of text data
- 📊 Sentiment prediction with real-time feedback
- 🧠 LIME explainability bar plots for each prediction
- 📈 Sentiment distribution charts
- 🔍 Filter results by sentiment
- 📋 Evaluation metrics for uploaded CSV (BERT only)
- 💡 Clean and modular Streamlit interface

---

## 🚀 Demo

📽️ [Watch the App Demo on YouTube](https://youtube.com/your-demo-link)  
🌐 [Live App (optional)](https://share.streamlit.io/your-app-link)

---

## 🧠 Models Used

| Model               | Description                                               |
|--------------------|-----------------------------------------------------------|
| Logistic Regression| TF-IDF + Logistic Regression for fast and lightweight use |
| BERT (DistilBERT)  | Fine-tuned Transformer model for high-accuracy predictions|

---

## 🛠️ How It Works

1. Preprocesses input text using a custom clean_text() function.
2. Logistic model uses scikit-learn TF-IDF + LogisticRegression.
3. BERT model loads fine-tuned DistilBERT from Hugging Face Transformers.
4. LIME explains individual predictions with interpretable bar plots.
5. CSV predictions offer summary charts and evaluation metrics.

---

## 📂 Project Structure

sentiment-analysis-streamlit/
│
├── app.py # Streamlit UI with model switcher
├── models/
│ ├── logistic_model.joblib
│ └── bert_sentiment_model/
├── utils/
│ └── preprocess_data.py
├── data/
│ └── example_dataset.csv
└── requirements.txt

---


---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/sentiment-analysis-streamlit.git
cd sentiment-analysis-streamlit
pip install -r requirements.txt
```

---
## Example Use Cases

- Analyzing Twitter sentiment about a product launch

- Processing customer reviews for support/marketing teams

- Comparing traditional ML vs Transformer models

- Visualizing important words driving sentiment with LIME

---
## Sample Output

- Prediction: Positive 😊

- LIME: Words like “great”, “amazing” push sentiment up

- Sentiment Bar Chart: 72% Positive, 28% Negative

- Evaluation: Accuracy = 79%, F1 Score = 0.80

---
## Author

Oluwafemi Ayodele
AI Engineer | Data Scientist | AI Automation Engineer
🔗 [LinkedIn](www.linkedin.com/in/oluwafemi-oloye-a3b772353) • [GitHub](https://github.com/femi-oloye/my-project1.git)

---
## License

This project is licensed under the MIT License.

Let me know if you'd like a version tailored for PromptBase, Gumroad, or if you want a short LinkedIn post caption to go with it.



