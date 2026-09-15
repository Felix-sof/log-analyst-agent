"""
Log Analyst Agent: a Claude-powered agent that answers natural-language
questions about the AI4I 2020 predictive maintenance sensor log by calling
tool functions defined in tools.py (tool use / function calling).
"""

import json
import os

import anthropic
from dotenv import load_dotenv

from tools import TOOL_DISPATCH, TOOL_SCHEMAS

load_dotenv()

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 4096
MAX_TOOL_ITERATIONS = 8

SYSTEM_PROMPT = """You are a Log Analyst Agent for an industrial predictive-maintenance
system. You analyze sensor logs from the AI4I 2020 Predictive Maintenance Dataset,
which records readings from a milling machine (air/process temperature, rotational
speed, torque, tool wear) along with machine failure labels.

Use the available tools to answer the user's question with concrete numbers pulled
from the data - do not guess or fabricate statistics. When you report anomalies or
failure summaries, briefly explain what the numbers mean in plain language. If you
generate a plot, mention the file path where it was saved.
"""


def _execute_tool(name: str, tool_input: dict) -> str:
    """Run a tool function by name and serialize its result for the API."""
    if name not in TOOL_DISPATCH:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        result = TOOL_DISPATCH[name](**tool_input)
        return json.dumps(result)
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
        return json.dumps({"error": str(exc)})


def run_agent(question: str, client: anthropic.Anthropic = None) -> dict:
    """Run the agentic tool-use loop for a single user question.

    Returns a dict with the final natural-language answer and a trace of the
    tool calls the agent made along the way.
    """
    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    tool_call_trace = []

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        except anthropic.APIStatusError as exc:
            return {
                "answer": f"API error: {exc.message}",
                "tool_calls": tool_call_trace,
                "error": True,
            }
        except anthropic.APIConnectionError:
            return {
                "answer": "Network error while contacting the Claude API.",
                "tool_calls": tool_call_trace,
                "error": True,
            }

        if response.stop_reason != "tool_use":
            answer = next(
                (block.text for block in response.content if block.type == "text"),
                "",
            )
            return {"answer": answer, "tool_calls": tool_call_trace, "error": False}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result_json = _execute_tool(block.name, block.input)
            tool_call_trace.append(
                {"name": block.name, "input": block.input, "result": json.loads(result_json)}
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_json,
                }
            )

        messages.append({"role": "user", "content": tool_results})

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
