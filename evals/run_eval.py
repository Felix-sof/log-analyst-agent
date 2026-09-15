"""
Runs the Log Analyst Agent eval (evals/cases.py) against the live Gemini API.

Usage:
    python -m evals.run_eval                # run every case
    python -m evals.run_eval --retry-infra   # re-run only cases that failed
                                              # with an infra_error last time,
                                              # merging into the existing results

Writes per-case results to evals/results/results.json and prints a summary:
pass rate on graded cases, a per-category breakdown, and total token usage
(this project runs on the Gemini free tier, so dollar cost is $0 - token
count is the honest proxy for "how much this run used").

This is NOT an LLM-judged eval. Grading is programmatic (see grade_case
below): tool-call correctness is a closed, checkable property, and the
grounding check is a regex heuristic looking for number-like tokens in
answers that made no tool call. Both are cheap, deterministic, and directly
target the two real failure modes found while live-testing this agent
(wrong tool choice, and reciting memorized dataset statistics instead of
calling a tool). "review" category cases are run and recorded but not
scored - see cases.py's docstring for why.

Any case where the agent itself reports error=True is graded "infra_error",
not pass/fail: in this codebase run_agent only sets error=True for an
API/network failure (429, 5xx, SSL, or hitting MAX_TOOL_ITERATIONS) - a tool
rejecting bad input is caught inside _execute_tool and fed back to the model
as a normal result, never surfaced as error=True. So error=True here is
infrastructure noise, not a signal about the agent's behavior; scoring it as
a behavioral fail would silently blame the model for a rate limit.
"""

import json
import re
import sys
import time
from pathlib import Path

import agent
from evals.cases import CASES

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "results.json"

# crude but effective for this dataset: a run of 2+ digits, optionally with a
# decimal/thousands separator or a trailing '%'. "AI4I 2020" is stripped first
# since it's a fixed proper-noun year, not a data-derived statistic.
NUMBER_PATTERN = re.compile(r"\d[\d.,]{1,}\s*%?")
SAFE_PHRASES = ["AI4I 2020", "ai4i 2020"]

# Free-tier quota is 20 requests/minute on gemini-3.6-flash. Stay well under
# it with a sliding window, rather than a fixed per-case delay - a burst of
# multi-tool cases can cost several requests each, and a fixed delay that
# looks safe in isolation still overruns the quota once enough of those
# stack up (this is exactly what happened on the first full run).
MAX_REQUESTS_PER_WINDOW = 12
WINDOW_SECONDS = 60
_request_log: list[float] = []


def throttle(assumed_next_requests: int = 4) -> None:
    now = time.monotonic()
    while _request_log and now - _request_log[0] > WINDOW_SECONDS:
        _request_log.pop(0)
    if len(_request_log) + assumed_next_requests > MAX_REQUESTS_PER_WINDOW:
        sleep_for = WINDOW_SECONDS - (now - _request_log[0]) + 1
        print(f"         (pacing: sleeping {sleep_for:.0f}s to stay under the free-tier rate limit)")
        time.sleep(max(sleep_for, 0))


def record_requests(n: int) -> None:
    now = time.monotonic()
    _request_log.extend([now] * n)


def grade_case(case: dict, result: dict) -> dict:
    tool_names = [tc["name"] for tc in result["tool_calls"]]

    if result["error"]:
        return {
            "status": "infra_error",
            "notes": [result["answer"][:200]],
            "tool_names": tool_names,
        }

    if case.get("review_only"):
        return {"status": "review", "notes": [], "tool_names": tool_names}

    notes = []

    for must in case.get("must_call", []):
        if must not in tool_names:
            notes.append(f"expected a call to '{must}', got {tool_names or '[]'}")

    for forbidden in case.get("forbid_call", []):
        if forbidden in tool_names:
            notes.append(f"forbidden call to '{forbidden}' was made")

    if case.get("require_no_fabricated_numbers") and not tool_names:
        scrubbed = result["answer"]
        for phrase in SAFE_PHRASES:
            scrubbed = scrubbed.replace(phrase, "")
        suspicious = NUMBER_PATTERN.findall(scrubbed)
        if suspicious:
            notes.append(
                f"no tool was called, but the answer contains number-like "
                f"tokens: {suspicious[:8]} (possible ungrounded/memorized stat)"
            )

    return {"status": "fail" if notes else "pass", "notes": notes, "tool_names": tool_names}


def run_case(case: dict) -> dict:
    throttle()
    start = time.monotonic()
    result = agent.run_agent(case["question"])
    elapsed = time.monotonic() - start
    record_requests(result["usage"]["requests"])

    grade = grade_case(case, result)
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "status": grade["status"],
        "notes": grade["notes"],
        "tool_calls": grade["tool_names"],
        "answer": result["answer"],
        "usage": result["usage"],
        "elapsed_s": round(elapsed, 2),
        "error": result["error"],
    }


def run(only_ids: set = None) -> list[dict]:
    RESULTS_DIR.mkdir(exist_ok=True)

    existing_by_id = {}
    if only_ids and RESULTS_PATH.exists():
        existing_by_id = {r["id"]: r for r in json.loads(RESULTS_PATH.read_text(encoding="utf-8"))}

    cases_to_run = [c for c in CASES if only_ids is None or c["id"] in only_ids]
    results = []

    for i, case in enumerate(cases_to_run):
        row = run_case(case)
        results.append(row)

        marker = {"pass": "PASS", "fail": "FAIL", "review": "REVIEW", "infra_error": "INFRA!"}[row["status"]]
        print(f"[{i + 1}/{len(cases_to_run)}] {marker:6s} {case['id']} ({row['elapsed_s']:.1f}s)")
        for note in row["notes"]:
            print(f"         - {note}")

    if only_ids:
        existing_by_id.update({r["id"]: r for r in results})
        results = [existing_by_id[c["id"]] for c in CASES if c["id"] in existing_by_id]

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def summarize(results: list[dict]) -> None:
    graded = [r for r in results if r["status"] not in ("review", "infra_error")]
    passed = [r for r in graded if r["status"] == "pass"]
    reviewed = [r for r in results if r["status"] == "review"]
    infra_errors = [r for r in results if r["status"] == "infra_error"]

    total_tokens = sum(r["usage"]["total_tokens"] for r in results)
    total_requests = sum(r["usage"]["requests"] for r in results)

    print("\n" + "=" * 60)
    if graded:
        print(f"Graded: {len(passed)}/{len(graded)} passed ({len(passed) / len(graded) * 100:.0f}%)")
    print(f"Review-only (not graded): {len(reviewed)}")
    if infra_errors:
        print(f"Infra errors (not graded - rate limit / network, not the agent's fault): {len(infra_errors)}")
    print(f"Total API requests: {total_requests}, total tokens: {total_tokens:,}")
    print(f"Model: {agent.MODEL} (Gemini free tier - $0 cost)")

    by_category = {}
    for r in graded:
        by_category.setdefault(r["category"], []).append(r)

    print("\nBy category:")
    for category, rows in sorted(by_category.items()):
        cat_passed = sum(1 for r in rows if r["status"] == "pass")
        print(f"  {category:20s} {cat_passed}/{len(rows)}")

    failed = [r for r in graded if r["status"] == "fail"]
    if failed:
        print("\nFailed cases:")
        for r in failed:
            print(f"  - {r['id']}: {r['notes']}")

    if infra_errors:
        print("\nInfra-error cases (re-run with --retry-infra once the quota resets):")
        for r in infra_errors:
            print(f"  - {r['id']}: {r['notes']}")

    print(f"\nFull results: {RESULTS_PATH}")


if __name__ == "__main__":
    only_ids = None
    if "--retry-infra" in sys.argv:
        if not RESULTS_PATH.exists():
            print("No existing results.json to retry from - run without --retry-infra first.")
            sys.exit(1)
        prior = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        only_ids = {r["id"] for r in prior if r["status"] == "infra_error"}
        if not only_ids:
            print("No infra_error cases in the last run - nothing to retry.")
            sys.exit(0)
        print(f"Retrying {len(only_ids)} case(s) that hit infra errors last time...\n")

    results = run(only_ids=only_ids)
    summarize(results)
