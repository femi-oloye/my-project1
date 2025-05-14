import os
import json
from dotenv import load_dotenv # type: ignore
import kagglehub # type: ignore

load_dotenv()
def kaggleDataLoader():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise Exception("username or key not found")

    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, "w")as f:
        json.dump({"username": username, "key": key}, f)

    os.chmod(kaggle_json_path, 0o600)

    path = kagglehub.dataset_download("kazanova/sentiment140")
    return path

