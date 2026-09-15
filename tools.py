"""
Tool functions exposed to the Claude agent for analyzing the AI4I 2020
predictive maintenance sensor log.

Each public function here has a matching JSON schema entry in TOOL_SCHEMAS,
used by agent.py to register the tools with the Claude API (tool use /
function calling). TOOL_DISPATCH maps tool names to their implementations.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt

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


def _get_df():
    """Lazily load and cache the dataset."""
    global _df
    if _df is None:
        _df = load_data()
    return _df


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
]

TOOL_DISPATCH = {
    "get_stats": get_stats,
    "detect_anomalies": detect_anomalies,
    "plot_trend": plot_trend,
    "get_correlation": get_correlation,
    "compare_by_failure": compare_by_failure,
    "get_failure_summary": get_failure_summary,
}
