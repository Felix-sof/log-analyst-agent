"""Tests for evals/run_eval.py's grading logic (no API calls)."""

from evals.run_eval import grade_case

MUST_CALL_CASE = {"id": "x", "category": "tool_selection", "question": "q", "must_call": ["get_stats"]}
GROUNDING_CASE = {"id": "y", "category": "grounding", "question": "q", "require_no_fabricated_numbers": True}
REVIEW_CASE = {"id": "z", "category": "review", "question": "q", "review_only": True}


def _result(tool_names, error=False, answer="ok"):
    return {"tool_calls": [{"name": n} for n in tool_names], "error": error, "answer": answer}


class TestMustCall:
    def test_pass_when_expected_tool_called(self):
        grade = grade_case(MUST_CALL_CASE, _result(["get_stats"]))
        assert grade["status"] == "pass"

    def test_fail_when_wrong_tool_called(self):
        grade = grade_case(MUST_CALL_CASE, _result(["detect_anomalies"]))
        assert grade["status"] == "fail"
        assert "expected a call to 'get_stats'" in grade["notes"][0]


class TestGrounding:
    def test_pass_when_no_numbers_and_no_tool(self):
        grade = grade_case(GROUNDING_CASE, _result([], answer="Bu bir kolon açıklamasıdır."))
        assert grade["status"] == "pass"

    def test_fail_when_numbers_leak_without_a_tool_call(self):
        grade = grade_case(GROUNDING_CASE, _result([], answer="Toplam 6000 kayıt vardır."))
        assert grade["status"] == "fail"
        assert "6000" in grade["notes"][0]

    def test_pass_when_a_tool_was_called_even_with_numbers(self):
        # numbers are fine once a tool actually produced them this turn
        grade = grade_case(GROUNDING_CASE, _result(["get_stats"], answer="Ortalama 40 Nm."))
        assert grade["status"] == "pass"

    def test_ai4i_2020_is_not_a_false_positive(self):
        grade = grade_case(GROUNDING_CASE, _result([], answer="AI4I 2020 veri setinden bahsediyoruz."))
        assert grade["status"] == "pass"


class TestInfraError:
    def test_any_error_flag_short_circuits_to_infra_error(self):
        # even a case with must_call is graded infra_error, not fail, when
        # result["error"] is True - see run_eval.py's module docstring for why
        grade = grade_case(MUST_CALL_CASE, _result([], error=True, answer="API error (429): ..."))
        assert grade["status"] == "infra_error"


class TestReviewOnly:
    def test_review_only_case_is_never_graded(self):
        grade = grade_case(REVIEW_CASE, _result([], answer="anything at all, 12345"))
        assert grade["status"] == "review"
