"""
Downloads and cleans the AI4I 2020 Predictive Maintenance Dataset (UCI).

Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
License: CC BY 4.0
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATASET_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
DATA_DIR = Path(__file__).parent / "data"
RAW_CSV_PATH = DATA_DIR / "ai4i2020.csv"
CLEAN_CSV_PATH = DATA_DIR / "ai4i2020_clean.csv"

# Original column names -> clean, analysis-friendly names
COLUMN_RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def download_dataset(force: bool = False) -> Path:
    """Download and extract the dataset zip if not already present locally."""
    DATA_DIR.mkdir(exist_ok=True)

    if RAW_CSV_PATH.exists() and not force:
        return RAW_CSV_PATH

    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV file found in the downloaded dataset archive.")
        with zf.open(csv_names[0]) as src, open(RAW_CSV_PATH, "wb") as dst:
            dst.write(src.read())

    return RAW_CSV_PATH


def clean_dataset(raw_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV, normalize column names/types, and drop exact duplicates."""
    df = pd.read_csv(raw_path)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df = df.drop_duplicates()
    df["type"] = df["type"].astype("category")

    numeric_cols = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    return df.reset_index(drop=True)


def load_data(force_download: bool = False) -> pd.DataFrame:
    """Download (if needed), clean, cache, and return the dataset as a DataFrame."""
    if CLEAN_CSV_PATH.exists() and not force_download:
        return pd.read_csv(CLEAN_CSV_PATH)

    raw_path = download_dataset(force=force_download)
    df = clean_dataset(raw_path)
    df.to_csv(CLEAN_CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} rows, {len(data.columns)} columns\n")
    print("Columns:")
    print(list(data.columns))
    print("\nFirst 5 rows:")
    print(data.head().to_string())
