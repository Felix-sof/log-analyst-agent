# Log Analyst Agent — Eval

Measures one thing: does `agent.run_agent` pick the right tool(s) for a
question, and does it stay grounded (no numbers stated without a tool call
backing them)?

## Run it

```bash
python -m evals.run_eval                # run all 20 cases
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
