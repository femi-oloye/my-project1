from import_data import importData
import pandas as pd # type: ignore
import re
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocessData():
    data = importData()
    data["tweet"] = data["tweet"].apply(clean_text)
    data["target"] = data["target"].apply(lambda x: 1 if x == 4 else 0)
    return data


