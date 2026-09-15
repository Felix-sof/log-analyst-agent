"""
FastAPI wrapper around the Log Analyst Agent.

POST /analyze accepts a natural-language question (and, optionally, a log
file in CSV/TSV/Excel/JSON/Parquet, same columns as the AI4I 2020 dataset)
and returns a JSON report with the agent's answer and the tool calls it
made to produce it.
"""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

import agent
import tools
from data_loader import SUPPORTED_UPLOAD_EXTENSIONS, load_uploaded_file

app = FastAPI(
    title="Log Analyst Agent API",
    description=(
        "Analyzes industrial sensor logs (AI4I 2020 Predictive Maintenance "
        "Dataset format) using a Claude tool-use agent."
    ),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": agent.MODEL}


@app.post("/analyze")
async def analyze(
    question: str = Form(..., description="Natural-language question about the log data"),
    file: Optional[UploadFile] = File(
        None,
        description=(
            "Optional log file to analyze instead of the default dataset. "
            f"Supported formats: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}"
        ),
    ),
):
    """Run the Log Analyst Agent against the default dataset or an uploaded log file."""
    dataset_source = "default"

    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Unsupported file type '{suffix or file.filename}'. "
                        f"Supported formats: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}"
                    )
                },
            )
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            tools._df = load_uploaded_file(tmp_path, file.filename)
            dataset_source = file.filename
        except Exception as exc:  # noqa: BLE001 - report bad uploads to the caller
            return JSONResponse(
                status_code=400,
                content={"error": f"Could not parse uploaded file: {exc}"},
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    result = agent.run_agent(question)

    return JSONResponse(
        content={
            "question": question,
            "dataset_source": dataset_source,
            "model": agent.MODEL,
            "answer": result["answer"],
            "tool_calls": result["tool_calls"],
            "error": result["error"],
        }
    )
