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


SUPPORTED_UPLOAD_EXTENSIONS = (".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet")


def _read_any_format(path: Path, filename: str) -> pd.DataFrame:
    """Read a tabular file into a DataFrame based on its extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported file type '{suffix or filename}'. "
        f"Supported formats: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}"
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names/types and drop exact duplicates and bad rows.

    Works whether the DataFrame still has the original AI4I column names
    (e.g. "Tool wear [min]") or already-clean ones (e.g. "tool_wear_min") -
    COLUMN_RENAME_MAP only touches columns that match its original names.
    """
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df = df.drop_duplicates()

    numeric_cols = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    missing = [col for col in numeric_cols + ["type"] if col not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing required column(s): {missing}. "
            f"Expected AI4I 2020 dataset columns (original or cleaned names)."
        )

    df["type"] = df["type"].astype("category")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    return df.reset_index(drop=True)


def clean_dataset(raw_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV and clean it. Kept for the default-dataset pipeline."""
    return clean_dataframe(pd.read_csv(raw_path))


def load_uploaded_file(path: Path, filename: str) -> pd.DataFrame:
    """Read and clean a user-uploaded log file in any supported format."""
    return clean_dataframe(_read_any_format(path, filename))


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
