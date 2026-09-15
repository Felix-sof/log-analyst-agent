import pandas as pd
import pytest

import data_loader as dl


def _raw_ai4i_frame() -> pd.DataFrame:
    """A tiny frame using the ORIGINAL AI4I column names (as downloaded)."""
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3],
            "Product ID": ["L1", "L2", "L3"],
            "Type": ["L", "M", "H"],
            "Air temperature [K]": [298.1, 298.2, 298.3],
            "Process temperature [K]": [308.6, 308.7, 308.8],
            "Rotational speed [rpm]": [1551, 1408, 1498],
            "Torque [Nm]": [42.8, 46.3, 49.4],
            "Tool wear [min]": [0, 3, 5],
            "Machine failure": [0, 0, 1],
            "TWF": [0, 0, 0],
            "HDF": [0, 0, 1],
            "PWF": [0, 0, 0],
            "OSF": [0, 0, 0],
            "RNF": [0, 0, 0],
        }
    )


class TestCleanDataframe:
    def test_renames_original_columns(self):
        cleaned = dl.clean_dataframe(_raw_ai4i_frame())
        assert list(cleaned.columns) == [
            "udi",
            "product_id",
            "type",
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
            "machine_failure",
            "twf",
            "hdf",
            "pwf",
            "osf",
            "rnf",
        ]
        assert len(cleaned) == 3

    def test_already_clean_columns_pass_through(self):
        df = _raw_ai4i_frame().rename(columns=dl.COLUMN_RENAME_MAP)
        cleaned = dl.clean_dataframe(df)
        assert len(cleaned) == 3
        assert cleaned["torque_nm"].tolist() == [42.8, 46.3, 49.4]

    def test_drops_exact_duplicates(self):
        df = pd.concat([_raw_ai4i_frame(), _raw_ai4i_frame().iloc[[0]]], ignore_index=True)
        cleaned = dl.clean_dataframe(df)
        assert len(cleaned) == 3

    def test_missing_required_column_raises(self):
        df = _raw_ai4i_frame().drop(columns=["Torque [Nm]"])
        with pytest.raises(ValueError):
            dl.clean_dataframe(df)

    def test_non_numeric_junk_row_is_dropped(self):
        junk_row = _raw_ai4i_frame().iloc[[0]].copy()
        junk_row["UDI"] = 4
        junk_row["Torque [Nm]"] = "not-a-number"  # simulates a malformed raw CSV value
        df = pd.concat([_raw_ai4i_frame(), junk_row], ignore_index=True)
        cleaned = dl.clean_dataframe(df)
        assert len(cleaned) == 3  # the bad row is coerced to NaN and dropped


class TestReadAnyFormat:
    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("not tabular data")
        with pytest.raises(ValueError):
            dl.load_uploaded_file(path, "notes.txt")

    def test_csv_round_trip(self, tmp_path):
        path = tmp_path / "sample.csv"
        _raw_ai4i_frame().to_csv(path, index=False)
        cleaned = dl.load_uploaded_file(path, "sample.csv")
        assert len(cleaned) == 3
        assert "torque_nm" in cleaned.columns
