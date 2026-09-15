"""
Tool functions exposed to the Claude agent for analyzing the AI4I 2020
predictive maintenance sensor log.

Each public function here has a matching JSON schema entry in TOOL_SCHEMAS,
used by agent.py to register the tools with the Claude API (tool use /
function calling). TOOL_DISPATCH maps tool names to their implementations.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from data_loader import load_data

OUTPUT_DIR = Path(__file__).parent / "outputs"

NUMERIC_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

FAILURE_MODE_COLUMNS = {
    "twf": "Tool Wear Failure",
    "hdf": "Heat Dissipation Failure",
    "pwf": "Power Failure",
    "osf": "Overstrain Failure",
    "rnf": "Random Failure",
}

_df = None
_model = None
_model_metrics = None
_model_trained_on = None  # id() of the DataFrame the cached model was trained on


def _get_df():
    """Lazily load and cache the dataset."""
    global _df
    if _df is None:
        _df = load_data()
    return _df


def _get_model():
    """Lazily train (and cache) a failure-risk classifier on the current dataset.

    Retrains automatically if the dataset has changed (e.g. a new file was
    uploaded, which replaces the _df object) - detected via id(), since a
    cached model trained on the previous dataset would silently mispredict.
    """
    global _model, _model_metrics, _model_trained_on
    df = _get_df()

    if _model is not None and _model_trained_on == id(df):
        return _model, _model_metrics

    X = df[NUMERIC_COLUMNS]
    y = df["machine_failure"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # too few examples of one class to stratify - fall back to a plain split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_failure_rate_pct": round(float(y_test.mean() * 100), 3),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    _model = clf
    _model_metrics = metrics
    _model_trained_on = id(df)
    return _model, _model_metrics


def _validate_numeric_column(column: str) -> None:
    if column not in NUMERIC_COLUMNS:
        raise ValueError(
            f"Unknown column '{column}'. Valid numeric columns: {NUMERIC_COLUMNS}"
        )


def get_stats(column: str) -> dict:
    """Return basic descriptive statistics for a numeric sensor column.

    Args:
        column: One of the numeric sensor columns, e.g. "tool_wear_min".
    """
    _validate_numeric_column(column)
    series = _get_df()[column]
    return {
        "column": column,
        "count": int(series.count()),
        "mean": round(float(series.mean()), 3),
        "std": round(float(series.std()), 3),
        "min": round(float(series.min()), 3),
        "p25": round(float(series.quantile(0.25)), 3),
        "median": round(float(series.median()), 3),
        "p75": round(float(series.quantile(0.75)), 3),
        "max": round(float(series.max()), 3),
    }


def detect_anomalies(column: str, threshold: float = 3.0) -> dict:
    """Detect simple threshold-based anomalies in a sensor column using z-scores.

    A row is flagged as anomalous if abs((value - mean) / std) > threshold.

    Args:
        column: One of the numeric sensor columns, e.g. "torque_nm".
        threshold: Number of standard deviations from the mean beyond which
            a value is considered anomalous. Defaults to 3.0.
    """
    _validate_numeric_column(column)
    df = _get_df()
    series = df[column]
    mean = series.mean()
    std = series.std()

    if std == 0:
        return {
            "column": column,
            "threshold": threshold,
            "anomaly_count": 0,
            "anomaly_rate_pct": 0.0,
            "anomalies": [],
        }

    z_scores = (series - mean) / std
    mask = z_scores.abs() > threshold
    anomalies_df = df.loc[mask, ["udi", "product_id", column]].copy()
    anomalies_df["z_score"] = round(z_scores[mask], 3)

    anomalies = anomalies_df.head(50).to_dict(orient="records")

    return {
        "column": column,
        "threshold": threshold,
        "mean": round(float(mean), 3),
        "std": round(float(std), 3),
        "anomaly_count": int(mask.sum()),
        "anomaly_rate_pct": round(float(mask.mean() * 100), 3),
        "anomalies": anomalies,
        "note": "anomalies list capped at 50 rows" if mask.sum() > 50 else None,
    }


def plot_trend(column: str) -> dict:
    """Plot a sensor column's values across records and save the chart as a PNG.

    Args:
        column: One of the numeric sensor columns, e.g. "rotational_speed_rpm".
    """
    _validate_numeric_column(column)
    df = _get_df()

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"trend_{column}.png"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["udi"], df[column], linewidth=0.6)
    ax.set_xlabel("Record (UDI)")
    ax.set_ylabel(column)
    ax.set_title(f"Trend of {column} across records")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return {
        "column": column,
        "image_path": str(output_path),
    }


def get_correlation(column_a: str, column_b: str) -> dict:
    """Compute the Pearson correlation coefficient between two numeric sensor columns.

    Args:
        column_a: One of the numeric sensor columns, e.g. "torque_nm".
        column_b: Another numeric sensor column, e.g. "tool_wear_min".
    """
    _validate_numeric_column(column_a)
    _validate_numeric_column(column_b)
    df = _get_df()

    if column_a == column_b:
        return {
            "column_a": column_a,
            "column_b": column_b,
            "correlation": 1.0,
            "strength": "identical column",
            "direction": "n/a",
        }

    corr = df[column_a].corr(df[column_b])
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "column_a": column_a,
        "column_b": column_b,
        "correlation": round(float(corr), 4),
        "strength": strength,
        "direction": "positive" if corr >= 0 else "negative",
    }


def compare_by_failure(column: str) -> dict:
    """Compare a numeric sensor column's distribution between failed and non-failed records.

    Useful for root-cause questions like "does this sensor reading differ on
    machines that failed?" - splits the column by machine_failure (0 vs 1)
    and reports mean/std for each group plus the difference in means.

    Args:
        column: One of the numeric sensor columns, e.g. "process_temperature_k".
    """
    _validate_numeric_column(column)
    df = _get_df()

    normal = df.loc[df["machine_failure"] == 0, column]
    failed = df.loc[df["machine_failure"] == 1, column]

    normal_mean = float(normal.mean())
    failed_mean = float(failed.mean())

    return {
        "column": column,
        "normal_operation": {
            "count": int(normal.count()),
            "mean": round(normal_mean, 3),
            "std": round(float(normal.std()), 3),
        },
        "failed_operation": {
            "count": int(failed.count()),
            "mean": round(failed_mean, 3),
            "std": round(float(failed.std()), 3),
        },
        "mean_difference": round(failed_mean - normal_mean, 3),
    }


def get_failure_prediction_performance() -> dict:
    """Train (if needed) a failure-risk classifier and report its test-set performance.

    Uses a RandomForestClassifier on the five numeric sensor columns to
    predict machine_failure, with an 80/20 stratified train/test split.
    """
    _, metrics = _get_model()
    return {
        "model": "RandomForestClassifier (scikit-learn, 200 trees, class_weight=balanced)",
        "features_used": NUMERIC_COLUMNS,
        **metrics,
        "note": (
            "Evaluated on a held-out 20% test split from the currently loaded "
            "dataset. Failures are rare (a few percent of records), so "
            "precision/recall for the 'failure' class matter more than raw "
            "accuracy - a model that always predicts 'no failure' would still "
            "score a high accuracy while being useless."
        ),
    }


def predict_failure_probability(
    air_temperature_k: float,
    process_temperature_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> dict:
    """Predict the probability of machine failure for a hypothetical sensor reading.

    This estimates risk for a specific combination of sensor values using a
    classifier trained on the currently loaded dataset - it does NOT forecast
    *when* a failure will occur (the dataset has no time dimension, only
    independent snapshots), and is not reliable for inputs far outside the
    training data's range.

    Args:
        air_temperature_k: Air temperature in Kelvin.
        process_temperature_k: Process temperature in Kelvin.
        rotational_speed_rpm: Rotational speed in rpm.
        torque_nm: Torque in Nm.
        tool_wear_min: Tool wear in minutes.
    """
    model, _ = _get_model()
    row = pd.DataFrame(
        [[air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min]],
        columns=NUMERIC_COLUMNS,
    )

    classes = list(model.classes_)
    if 1 not in classes:
        failure_probability = 0.0  # model never saw a failure example to learn from
    else:
        failure_probability = float(model.predict_proba(row)[0][classes.index(1)])

    return {
        "input": {
            "air_temperature_k": air_temperature_k,
            "process_temperature_k": process_temperature_k,
            "rotational_speed_rpm": rotational_speed_rpm,
            "torque_nm": torque_nm,
            "tool_wear_min": tool_wear_min,
        },
        "failure_probability_pct": round(failure_probability * 100, 2),
        "predicted_label": "failure" if failure_probability >= 0.5 else "normal",
        "note": (
            "Risk estimate for this specific reading, not a time-to-failure "
            "forecast - the dataset has no timestamps to forecast from."
        ),
    }


def get_failure_summary() -> dict:
    """Return a summary of machine failures broken down by failure type and product type."""
    df = _get_df()
    total = len(df)
    total_failures = int(df["machine_failure"].sum())

    by_mode = {}
    for col, label in FAILURE_MODE_COLUMNS.items():
        count = int(df[col].sum())
        by_mode[label] = {
            "count": count,
            "rate_pct": round(count / total * 100, 3),
        }

    by_product_type = (
        df.groupby("type", observed=True)["machine_failure"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "failures", "count": "total"})
    )
    by_product_type["failure_rate_pct"] = round(
        by_product_type["failures"] / by_product_type["total"] * 100, 3
    )

    return {
        "total_records": total,
        "total_failures": total_failures,
        "overall_failure_rate_pct": round(total_failures / total * 100, 3),
        "by_failure_mode": by_mode,
        "by_product_type": by_product_type.to_dict(orient="index"),
    }


TOOL_SCHEMAS = [
    {
        "name": "get_stats",
        "description": (
            "Get basic descriptive statistics (count, mean, std, min, quartiles, max) "
            "for a numeric sensor column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "enum": NUMERIC_COLUMNS,
                    "description": "The numeric sensor column to summarize.",
                }
            },
            "required": ["column"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Detect threshold-based anomalies in a numeric sensor column using "
            "z-scores (values more than `threshold` standard deviations from the mean)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "enum": NUMERIC_COLUMNS,
                    "description": "The numeric sensor column to check for anomalies.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Number of standard deviations from the mean to flag as anomalous. Defaults to 3.0.",
                },
            },
            "required": ["column"],
        },
    },
    {
        "name": "plot_trend",
        "description": (
            "Generate and save a line chart showing a numeric sensor column's "
            "trend across records. Returns the saved image file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "enum": NUMERIC_COLUMNS,
                    "description": "The numeric sensor column to plot.",
                }
            },
            "required": ["column"],
        },
    },
    {
        "name": "get_correlation",
        "description": (
            "Compute the Pearson correlation coefficient between two numeric "
            "sensor columns, with a strength/direction interpretation. Use this "
            "instead of eyeballing two separate get_stats calls when asked how "
            "two sensor readings relate to each other."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column_a": {
                    "type": "string",
                    "enum": NUMERIC_COLUMNS,
                    "description": "The first numeric sensor column.",
                },
                "column_b": {
                    "type": "string",
                    "enum": NUMERIC_COLUMNS,
                    "description": "The second numeric sensor column.",
                },
            },
            "required": ["column_a", "column_b"],
        },
    },
    {
        "name": "compare_by_failure",
        "description": (
            "Compare a numeric sensor column's mean/std between records where "
            "the machine failed vs. did not fail. Use this for root-cause "
            "questions like 'is this sensor reading different on machines that "
            "failed?' instead of guessing from the overall statistics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "enum": NUMERIC_COLUMNS,
                    "description": "The numeric sensor column to compare.",
                }
            },
            "required": ["column"],
        },
    },
    {
        "name": "get_failure_summary",
        "description": (
            "Get a summary of machine failures: overall failure rate, breakdown "
            "by failure mode (tool wear, heat dissipation, power, overstrain, "
            "random), and breakdown by product quality type (L/M/H)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_failure_prediction_performance",
        "description": (
            "Train (if not already cached) a failure-risk classifier on the current "
            "dataset and report its test-set accuracy, precision, recall, and F1 for "
            "the failure class. Use this when asked how well failures can be "
            "predicted, or how reliable a risk prediction would be."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "predict_failure_probability",
        "description": (
            "Predict the failure probability for a hypothetical sensor reading "
            "(specific values, not from the dataset). Does NOT forecast when a "
            "failure will happen - the dataset has no time dimension - only the "
            "risk associated with a given combination of readings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "air_temperature_k": {"type": "number", "description": "Air temperature in Kelvin."},
                "process_temperature_k": {"type": "number", "description": "Process temperature in Kelvin."},
                "rotational_speed_rpm": {"type": "number", "description": "Rotational speed in rpm."},
                "torque_nm": {"type": "number", "description": "Torque in Nm."},
                "tool_wear_min": {"type": "number", "description": "Tool wear in minutes."},
            },
            "required": [
                "air_temperature_k",
                "process_temperature_k",
                "rotational_speed_rpm",
                "torque_nm",
                "tool_wear_min",
            ],
        },
    },
]

TOOL_DISPATCH = {
    "get_stats": get_stats,
    "detect_anomalies": detect_anomalies,
    "plot_trend": plot_trend,
    "get_correlation": get_correlation,
    "compare_by_failure": compare_by_failure,
    "get_failure_summary": get_failure_summary,
    "get_failure_prediction_performance": get_failure_prediction_performance,
    "predict_failure_probability": predict_failure_probability,
}
