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

17 of 20 cases got a clean signal: **16 graded, 15 passed**; 1 case
(`review_data_provenance`) is review-only by design. 3 cases
(`threshold_sensitive_scan`, `review_predict_next_failure`,
`review_misleading_premise`) never got a clean run - the free tier's
per-minute quota was exhausted by a long day of manual + automated testing
against this same key. Re-run `--retry-infra` once the quota resets to fill
those in; they are not evidence of anything about the agent.

**One real regression was found and is not yet re-verified:**
`grounding_list_columns` ("list all the columns and what they mean") got
answered with dataset-derived numbers - "UDI ranges 1 to 10,000", "~60% are
type L" - without any tool call. This is exactly the failure mode
`agent.py`'s "Critical grounding rule" is meant to prevent; the rule alone
wasn't enough to stop it on this rephrasing. `SYSTEM_PROMPT` was
strengthened with a concrete negative example matching this exact pattern,
but a live re-run to confirm the fix hit the same exhausted quota before it
could complete - **re-run this case (or the full grounding category) once
the quota resets, and update this note with the result.**

## Failure-risk prediction tools (added 2026-09-15)

Two cases (`tool_predict_failure_probability`, `tool_get_failure_prediction_performance`)
cover the new `predict_failure_probability` / `get_failure_prediction_performance`
tools (a `RandomForestClassifier` trained on the loaded dataset - see `tools.py`).
Same quota exhaustion blocked a full graded run, but a manual live call
confirmed the important part: given "Hava sıcaklığı 303.5 K, proses sıcaklığı
313 K, dönüş hızı 1200 rpm, tork 68 Nm ve takım aşınması 220 dakika olan bir
makinenin arıza riski nedir?", the agent correctly parsed all five parameters
and called `predict_failure_probability` with the right arguments, returning
89.15% failure probability (matching a direct, non-agent call to the same
function). Only the follow-up "summarize this in natural language" request
hit the infra error, not the tool selection itself - **run the full eval once
the quota resets to get a graded result for these two cases.**
