from src.data.make_dataset import download_and_extract_data, load_raw_data
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/hopesb/student-depression-dataset"
RAW_DIR = "./data/raw"
CSV_FILE = Path(RAW_DIR) / "Student Depression Dataset.csv"

if not CSV_FILE.exists():
    download_and_extract_data(URL, RAW_DIR)

df_raw = load_raw_data(CSV_FILE)
