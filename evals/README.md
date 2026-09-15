# Log Analyst Agent — Eval

Measures one thing: does `agent.run_agent` pick the right tool(s) for a
question, and does it stay grounded (no numbers stated without a tool call
backing them)?

## Run it

```bash
python -m evals.run_eval                # run all 22 cases
python -m evals.run_eval --retry-infra   # re-run only cases that hit an
                                          # API/network error last time,
                                          # merging into the existing results
```

Results are written to `evals/results/results.json`. See `cases.py` and
`run_eval.py`'s module docstrings for the case design and grading rules.

## Baseline (2026-09-15, gemini-3.5-flash / gemini-3.6-flash)

16 of 22 cases got a clean signal: **15 graded, 15 passed (100%)**; 1 case
(`review_data_provenance`) is review-only by design. 6 cases never got a
clean run:

```
threshold_sensitive_scan, grounding_list_columns, review_predict_next_failure,
review_misleading_premise, tool_predict_failure_probability,
tool_get_failure_prediction_performance
```

**This is not a quota problem** - that was the first hypothesis, and
switching from gemini-3.6-flash to gemini-3.5-flash did clear a genuine
429 quota exhaustion earlier in the day. But repeated retries afterward,
including a much later attempt after the quota window had reset many times
over, kept hitting the same `SSL: INVALID_SESSION_ID` error. A deeper dive
(see `agent.py`'s `_default_client` docstring and the commit history)
found that even the plain `requests` library - not just the SDK's `httpx` -
fails intermittently on this network for anything beyond a trivial GET
request; a POST with a real response body sometimes fails to fully
transfer. That's a local network reliability issue (unstable connection,
VPN, or antivirus HTTPS inspection interfering with larger transfers), not
a code bug - no retry count fixes an unreliable physical connection, only
gives it more chances to eventually succeed. Re-run `--retry-infra` from a
more stable connection to fill these in.

**One real regression was found in `grounding_list_columns`** ("list all
the columns and what they mean"): the agent answered with dataset-derived
numbers - "UDI ranges 1 to 10,000", "~60% are type L" - without any tool
call. This is exactly the failure mode `agent.py`'s "Critical grounding
rule" is meant to prevent; the rule alone wasn't enough to stop it on this
rephrasing. `SYSTEM_PROMPT` was strengthened with a concrete negative
example matching this exact pattern, but every live re-run attempt since
has hit the network issue above before completing - **the fix is shipped
but not yet re-verified. Re-run this case first once the network is
stable.**

## Failure-risk prediction tools (added 2026-09-15)

Two cases (`tool_predict_failure_probability`, `tool_get_failure_prediction_performance`)
cover the new `predict_failure_probability` / `get_failure_prediction_performance`
tools (a `RandomForestClassifier` trained on the loaded dataset - see `tools.py`).
Neither has completed a graded eval run yet (same network issue), but a
manual live call confirmed the important part: given "Hava sıcaklığı 303.5 K,
proses sıcaklığı 313 K, dönüş hızı 1200 rpm, tork 68 Nm ve takım aşınması 220
dakika olan bir makinenin arıza riski nedir?", the agent correctly parsed
all five parameters and called `predict_failure_probability` with the right
arguments, returning 89.15% failure probability (matching a direct,
non-agent call to the same function). Only the follow-up "summarize this in
natural language" request hit the network error, not the tool selection
itself - **run the full eval once the network is stable to get a graded
result for both cases.**
