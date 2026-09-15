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
MAX_HISTORY_TURNS = 12  # cap on stored user-question turns, to bound context growth


def _default_client() -> genai.Client:
    """Build a Gemini client with a hardened SSL context.

    Disabling TLS session tickets and connection keep-alive works around an
    intermittent "SSL: INVALID_SESSION_ID" connection error seen with httpx
    on Windows machines behind antivirus / TLS-inspecting network software:
    without this, a poisoned pooled connection gets reused across retries,
    so every retry fails identically. Forcing a fresh TCP+TLS handshake per
    request costs a little latency but avoids that failure mode; certificate
    verification is unaffected.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.options |= ssl.OP_NO_TICKET
    client_args = {"verify": ctx, "limits": httpx.Limits(max_keepalive_connections=0)}
    return genai.Client(http_options=types.HttpOptions(client_args=client_args))


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
what a failure mode represents) may be answered directly. Any concrete figure may not -
including ones that feel like harmless background context. For example, if asked to
list and explain the columns, do NOT add asides like "UDI ranges from 1 to 10,000" or
"type L is roughly 60% of records" - those are data-derived facts, not schema
definitions, even though they are true for the public dataset. Describe what each
column IS and means; leave out counts, ranges, and proportions unless a tool just gave
them to you this turn.

Use the available tools to answer the user's question with concrete numbers pulled
from the data - do not guess or fabricate statistics. When you report anomalies or
failure summaries, briefly explain what the numbers mean in plain language. If you
generate a plot, mention the file path where it was saved.

You also have a trained failure-risk classifier (predict_failure_probability,
get_failure_prediction_performance). Be precise about what it can and cannot do: it
estimates failure risk for a specific combination of sensor readings, and reports how
reliable that estimate is (precision/recall on held-out data) - it does NOT forecast
*when* a failure will happen, since the dataset has no time dimension. If asked "when
will the next failure happen" or similar, say plainly that this can't be answered from
this data, and offer the risk-prediction tool as the closest available alternative.
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


def _generate_with_retry(get_client, contents: list, config: types.GenerateContentConfig):
    """Call generate_content, retrying transient connection errors and 429/5xx responses.

    `get_client(attempt)` returns the client to use for a given attempt. For
    the default self-managed client this returns a brand-new client on every
    retry, so a retry never reuses a connection pool that just failed - a
    stale/poisoned pooled connection is the usual cause of the intermittent
    httpx "SSL: INVALID_SESSION_ID" error seen behind TLS-inspecting network
    software. A caller-supplied client is returned unchanged on every attempt.
    """
    last_exc = None
    for attempt in range(MAX_REQUEST_RETRIES):
        client = get_client(attempt)
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as exc:
            if exc.code not in RETRYABLE_STATUS_CODES or attempt == MAX_REQUEST_RETRIES - 1:
                raise
            last_exc = exc
            # Rate limits (429) reset on the order of tens of seconds on the free tier.
            wait = 20 if exc.code == 429 else min(2**attempt, 8)
        except httpx.TransportError as exc:
            # e.g. intermittent "SSL: INVALID_SESSION_ID" behind TLS-inspecting network software
            if attempt == MAX_REQUEST_RETRIES - 1:
                raise
            last_exc = exc
            wait = min(2**attempt, 8)
        time.sleep(wait)
    raise last_exc


def _execute_tool(name: str, tool_args: dict) -> dict:
    """Run a tool function by name and return its (JSON-serializable) result."""
    if name not in TOOL_DISPATCH:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return TOOL_DISPATCH[name](**tool_args)
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
        return {"error": str(exc)}


def run_agent(question: str, history: list = None, client: genai.Client = None) -> dict:
    """Run the agentic tool-use loop for a single user question.

    `history` is the list of prior turns as returned in a previous call's
    "history" field (each turn is itself a list of google.genai Content
    objects) - pass it back in to give the agent multi-turn conversation
    memory. Omit it (or pass None) for a fresh, single-turn conversation.

    Returns a dict with the final natural-language answer, a trace of the
    tool calls made along the way, the updated "history" to pass into the
    next call, and "usage" (summed prompt/candidates/total token counts
    across every generate_content call this turn made).
    """
    external_client = client
    default_client = None if external_client is not None else _default_client()

    def get_client(attempt: int):
        if external_client is not None:
            return external_client
        nonlocal default_client
        if attempt > 0:
            default_client = _default_client()  # fresh connection pool per retry
        return default_client

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[_build_tool()],
    )

    past_turns = (history or [])[-MAX_HISTORY_TURNS:]
    contents = [c for turn in past_turns for c in turn]

    user_content = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    contents.append(user_content)
    this_turn = [user_content]

    tool_call_trace = []
    usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0, "requests": 0}

    def track_usage(response) -> None:
        meta = response.usage_metadata
        if meta is None:
            return
        usage["prompt_tokens"] += meta.prompt_token_count or 0
        usage["candidates_tokens"] += meta.candidates_token_count or 0
        usage["total_tokens"] += meta.total_token_count or 0
        usage["requests"] += 1

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = _generate_with_retry(get_client, contents, config)
        except errors.APIError as exc:
            return {
                "answer": f"API error ({exc.code}): {exc.message}",
                "tool_calls": tool_call_trace,
                "history": past_turns,  # don't persist a failed turn
                "usage": usage,
                "error": True,
            }
        except httpx.TransportError as exc:
            return {
                "answer": f"Network error while contacting the Gemini API: {exc}",
                "tool_calls": tool_call_trace,
                "history": past_turns,
                "usage": usage,
                "error": True,
            }

        track_usage(response)

        function_calls = response.function_calls or []
        if not function_calls:
            model_content = response.candidates[0].content
            this_turn.append(model_content)
            return {
                "answer": response.text or "",
                "tool_calls": tool_call_trace,
                "history": past_turns + [this_turn],
                "usage": usage,
                "error": False,
            }

        model_content = response.candidates[0].content
        contents.append(model_content)
        this_turn.append(model_content)

        response_parts = []
        for call in function_calls:
            call_args = dict(call.args or {})
            result = _execute_tool(call.name, call_args)
            tool_call_trace.append({"name": call.name, "input": call_args, "result": result})
            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )

        # The API rejects role="tool"/"function" here; function responses go back as "user".
        tool_result_content = types.Content(role="user", parts=response_parts)
        contents.append(tool_result_content)
        this_turn.append(tool_result_content)

    return {
        "answer": "Stopped after too many tool-use iterations without a final answer.",
        "tool_calls": tool_call_trace,
        "history": past_turns,
        "usage": usage,
        "error": True,
    }


if __name__ == "__main__":
    print("Log Analyst Agent - ask a question about the sensor log (Ctrl+C to exit).\n")
    conversation_history = None
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        result = run_agent(question, history=conversation_history)
        conversation_history = result["history"]
        print(f"\n{result['answer']}\n")
