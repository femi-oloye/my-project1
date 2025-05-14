import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import sys
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

from preprocess_data import preprocessData # type: ignore

# Load and process data
data = preprocessData()
df = data[["tweet", "target"]].dropna()

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Shuffle the dataset and select a sample of 1000 entries for quicker training
dataset = dataset.shuffle(seed=42).select([i for i in range(1000)])

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize function
def tokenize(batch):
    return tokenizer(batch["tweet"], padding=True, truncation=True, max_length=128)

# Apply the tokenization
dataset = dataset.map(tokenize, batched=True)

# Rename the column for consistency with the model
dataset = dataset.rename_column("target", "labels")  # Make sure labels are named 'labels'
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Split the dataset
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load the model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Define metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# Set up TrainingArguments
training_args = TrainingArguments(
    output_dir="./bert_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2, # Accumulate gradients for 2 steps
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on the evaluation set
eval_results = trainer.evaluate()
print("🔍 Final Evaluation Metrics:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")


# Save the model
trainer.save_model("bert_sentiment_model")
tokenizer.save_pretrained("bert_sentiment_model")
