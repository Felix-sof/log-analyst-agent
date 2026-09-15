"""
Log Analyst Agent: a Gemini-powered agent that answers natural-language
questions about the AI4I 2020 predictive maintenance sensor log by calling
tool functions defined in tools.py (tool use / function calling).
"""

import os
import ssl
import time

import certifi
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from tools import TOOL_DISPATCH, TOOL_SCHEMAS

load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_TOOL_ITERATIONS = 8
MAX_REQUEST_RETRIES = 6
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _default_client() -> genai.Client:
    """Build a Gemini client with a hardened SSL context.

    Disabling TLS session tickets works around an "SSL: INVALID_SESSION_ID"
    connection error seen with httpx on Windows machines behind antivirus /
    TLS-inspecting network software; certificate verification is unaffected.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.options |= ssl.OP_NO_TICKET
    return genai.Client(http_options=types.HttpOptions(client_args={"verify": ctx}))


SYSTEM_PROMPT = """You are a Log Analyst Agent for an industrial predictive-maintenance
system. You analyze sensor logs from the AI4I 2020 Predictive Maintenance Dataset,
which records readings from a milling machine (air/process temperature, rotational
speed, torque, tool wear) along with machine failure labels.

Critical grounding rule: the dataset currently loaded may or may not be the public
AI4I 2020 dataset - the user can upload a different log file with the same columns.
You may recognize this dataset's schema from your training data, but you must NEVER
state a number (a count, percentage, mean, min/max, anomaly count, failure count,
etc.) unless it came from a tool call made in THIS conversation. Even if a number
looks familiar, treat it as unknown until a tool confirms it. If a question asks for
any statistic and you have not yet called a tool for it, call the tool first.

General descriptions that are not tied to specific values (e.g. what a column means,
what a failure mode represents) may be answered directly. Any concrete figure may not.

Use the available tools to answer the user's question with concrete numbers pulled
from the data - do not guess or fabricate statistics. When you report anomalies or
failure summaries, briefly explain what the numbers mean in plain language. If you
generate a plot, mention the file path where it was saved.
"""


def _build_tool() -> types.Tool:
    """Convert TOOL_SCHEMAS (name/description/input_schema) into a Gemini Tool."""
    declarations = [
        types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters_json_schema=schema["input_schema"],
        )
        for schema in TOOL_SCHEMAS
    ]
    return types.Tool(function_declarations=declarations)


def _generate_with_retry(client: genai.Client, contents: list, config: types.GenerateContentConfig):
    """Call generate_content, retrying transient connection errors and 429/5xx responses."""
    last_exc = None
    for attempt in range(MAX_REQUEST_RETRIES):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as exc:
            if exc.code not in RETRYABLE_STATUS_CODES or attempt == MAX_REQUEST_RETRIES - 1:
                raise
            last_exc = exc
        except httpx.TransportError as exc:
            # e.g. intermittent "SSL: INVALID_SESSION_ID" behind TLS-inspecting network software
            if attempt == MAX_REQUEST_RETRIES - 1:
                raise
            last_exc = exc
        time.sleep(min(2**attempt, 8))
    raise last_exc


def _execute_tool(name: str, tool_args: dict) -> dict:
    """Run a tool function by name and return its (JSON-serializable) result."""
    if name not in TOOL_DISPATCH:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return TOOL_DISPATCH[name](**tool_args)
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
        return {"error": str(exc)}


def run_agent(question: str, client: genai.Client = None) -> dict:
    """Run the agentic tool-use loop for a single user question.

    Returns a dict with the final natural-language answer and a trace of the
    tool calls the agent made along the way.
    """
    client = client or _default_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[_build_tool()],
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    tool_call_trace = []

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = _generate_with_retry(client, contents, config)
        except errors.APIError as exc:
            return {
                "answer": f"API error ({exc.code}): {exc.message}",
                "tool_calls": tool_call_trace,
                "error": True,
            }
        except httpx.TransportError as exc:
            return {
                "answer": f"Network error while contacting the Gemini API: {exc}",
                "tool_calls": tool_call_trace,
                "error": True,
            }

        function_calls = response.function_calls or []
        if not function_calls:
            return {"answer": response.text or "", "tool_calls": tool_call_trace, "error": False}

        contents.append(response.candidates[0].content)

        response_parts = []
        for call in function_calls:
            call_args = dict(call.args or {})
            result = _execute_tool(call.name, call_args)
            tool_call_trace.append({"name": call.name, "input": call_args, "result": result})
            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )

        # The API rejects role="tool"/"function" here; function responses go back as "user".
        contents.append(types.Content(role="user", parts=response_parts))

    return {
        "answer": "Stopped after too many tool-use iterations without a final answer.",
        "tool_calls": tool_call_trace,
        "error": True,
    }


if __name__ == "__main__":
    print("Log Analyst Agent - ask a question about the sensor log (Ctrl+C to exit).\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        result = run_agent(question)
        print(f"\n{result['answer']}\n")
