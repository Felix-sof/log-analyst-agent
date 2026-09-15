# Log Analyst Agent

An LLM-powered agent that analyzes industrial sensor logs using tool use
(function calling). You ask a question in natural language (e.g. *"Is there
anomalous behavior in the Tool wear column?"*), and the agent calls the
appropriate analysis tools, interprets the results, and explains them back
to you in plain language.

## How it works

1. **`data_loader.py`** downloads, cleans, and caches the dataset as a
   pandas DataFrame.
2. **`tools.py`** defines the analysis functions the agent can call:
   - `get_stats(column)` - descriptive statistics for a sensor column
   - `detect_anomalies(column, threshold)` - z-score based anomaly detection
   - `plot_trend(column)` - saves a trend chart (PNG) for a sensor column
   - `get_correlation(column_a, column_b)` - Pearson correlation between two columns
   - `compare_by_failure(column)` - compares a column's mean/std on failed vs. normal records
   - `get_failure_summary()` - failure counts by failure mode and product type
3. **`agent.py`** runs the agentic tool-use loop against the Google Gemini API
   (free tier): Gemini decides which tool(s) to call, the tools execute
   against the DataFrame, results are fed back to Gemini, and it produces a
   final natural-language answer. Supports multi-turn conversation memory -
   pass the `history` returned from one call into the next.
4. **`api.py`** exposes the agent over HTTP via FastAPI.
5. **`streamlit_app.py`** provides a chat-style web UI on top of the same
   agent (question in, answer + generated charts out).

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning
Repository, dataset #601).

- Source: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- License: **CC BY 4.0** (Creative Commons Attribution 4.0 International)
- 10,000 synthetic records of milling-machine sensor readings (air/process
  temperature, rotational speed, torque, tool wear) with machine failure
  labels and failure-mode flags (TWF, HDF, PWF, OSF, RNF).

The dataset is downloaded automatically on first run; it is not bundled in
this repository.

**Grounding note:** because this is a well-known public dataset, an LLM can
recognize it and be tempted to recite memorized statistics instead of
reading the data actually loaded (which may be a different uploaded file).
`agent.py`'s system prompt explicitly forbids stating any number that
didn't come from a tool call made in the current conversation - see the
"Critical grounding rule" in `SYSTEM_PROMPT`.

## Setup

1. Get a **free** Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey)
   (no credit card required). The free tier currently gives 1,500 requests/day
   on Flash models - more than enough for this project.
2. Install dependencies and configure the key:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY
```

The API key is read from `.env` via `python-dotenv` at runtime - it is never
hardcoded in source.

## Usage

### Download and inspect the data

```bash
python data_loader.py
```

### Run the agent interactively (CLI)

```bash
python agent.py
```

```
Log Analyst Agent - ask a question about the sensor log (Ctrl+C to exit).

> Tool wear kolonunda anormallik var mı?

Tool wear kolonunda (0-253 dakika, ortalama ~108 dk) 3 standart sapma
eşiğine göre herhangi bir anormallik tespit edilmedi...
```

The CLI remembers the conversation across questions in the same session
(follow-ups like "peki ya torque için?" work without repeating context).

### Run the web UI (Streamlit)

```bash
streamlit run streamlit_app.py
```

Opens a chat-style UI at `http://localhost:8501` - ask a question, see the
agent's answer, and view any generated trend charts inline. The chat
remembers earlier turns in the session (use the sidebar's "Konuşmayı
temizle" button to reset). You can also upload a different log file (CSV,
TSV, Excel, JSON, or Parquet) from the sidebar - this also resets the
conversation, since old tool results no longer apply to the new data.

### Run the API server

```bash
uvicorn api:app --reload
```

```bash
curl -X POST http://localhost:8000/analyze \
  -F "question=Are there any anomalies in the torque readings?"
```

Example response:

```json
{
  "question": "Are there any anomalies in the torque readings?",
  "dataset_source": "default",
  "model": "gemini-3.6-flash",
  "answer": "I checked torque_nm for outliers beyond 3 standard deviations...",
  "tool_calls": [
    {
      "name": "detect_anomalies",
      "input": {"column": "torque_nm", "threshold": 3.0},
      "result": { "anomaly_count": 25, "anomaly_rate_pct": 0.25, "...": "..." }
    }
  ],
  "error": false
}
```

You can also upload a different log file (same columns as the AI4I dataset,
original or cleaned names) by adding `-F "file=@my_log.xlsx"` to the
request. Supported formats: `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`,
`.parquet`.

For multi-turn conversations, pass the `conversation_id` from a response
back into the next request's form data to continue that thread with memory:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "question=Peki ya rotational speed için?" \
  -F "conversation_id=<id-from-previous-response>"
```

(The conversation store is in-memory per server process - it resets on
restart and isn't shared across multiple workers.)

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover `tools.py` and `data_loader.py` against a small synthetic
dataset (fast, deterministic, no network or API key required) - they don't
test the Gemini integration itself, since that needs a live API call.

## Project structure

```
log-analyst-agent/
├── data_loader.py     # download + clean the dataset, multi-format upload parsing
├── tools.py            # tool functions + JSON schemas for the agent
├── agent.py             # Gemini tool-use agentic loop, multi-turn memory
├── api.py                 # FastAPI /analyze endpoint
├── streamlit_app.py    # chat-style web UI
├── tests/                 # pytest suite (tools.py, data_loader.py)
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
└── README.md
```

## License

This project's code has no specific license attached. The underlying
dataset is licensed **CC BY 4.0** by its original authors (S. Matzka,
UCI Machine Learning Repository) - attribution to the dataset source is
required for any reuse or redistribution of the data itself.
