import math

import pytest

import tools


class TestGetStats:
    def test_basic_stats(self):
        result = tools.get_stats("tool_wear_min")
        assert result["count"] == 10
        assert result["mean"] == 55.0
        assert result["min"] == 10.0
        assert result["max"] == 100.0
        assert result["median"] == 55.0

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError):
            tools.get_stats("humidity")


class TestDetectAnomalies:
    def test_flags_the_outlier(self):
        result = tools.detect_anomalies("rotational_speed_rpm", threshold=2.0)
        assert result["anomaly_count"] == 1
        assert result["anomalies"][0]["rotational_speed_rpm"] == 9000.0

    def test_high_threshold_flags_nothing(self):
        result = tools.detect_anomalies("rotational_speed_rpm", threshold=10.0)
        assert result["anomaly_count"] == 0

    def test_zero_variance_column_is_never_anomalous(self):
        result = tools.detect_anomalies("torque_nm", threshold=0.001)
        assert result["anomaly_count"] == 0
        assert result["anomalies"] == []

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError):
            tools.detect_anomalies("humidity")


class TestGetCorrelation:
    def test_perfectly_correlated_columns(self):
        result = tools.get_correlation("air_temperature_k", "process_temperature_k")
        assert result["correlation"] == pytest.approx(1.0, abs=1e-6)
        assert result["strength"] == "strong"
        assert result["direction"] == "positive"

    def test_same_column_short_circuits(self):
        result = tools.get_correlation("torque_nm", "torque_nm")
        assert result["correlation"] == 1.0
        assert result["direction"] == "n/a"

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError):
            tools.get_correlation("humidity", "torque_nm")


class TestCompareByFailure:
    def test_mean_difference(self):
        result = tools.compare_by_failure("process_temperature_k")
        assert result["normal_operation"]["count"] == 8
        assert result["failed_operation"]["count"] == 2
        assert result["normal_operation"]["mean"] == pytest.approx(308.5)
        assert result["failed_operation"]["mean"] == pytest.approx(313.5)
        assert result["mean_difference"] == pytest.approx(5.0)

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError):
            tools.compare_by_failure("humidity")


class TestGetFailureSummary:
    def test_totals(self):
        result = tools.get_failure_summary()
        assert result["total_records"] == 10
        assert result["total_failures"] == 2
        assert result["overall_failure_rate_pct"] == 20.0

    def test_by_failure_mode(self):
        result = tools.get_failure_summary()
        assert result["by_failure_mode"]["Heat Dissipation Failure"]["count"] == 1
        assert result["by_failure_mode"]["Power Failure"]["count"] == 1
        assert result["by_failure_mode"]["Tool Wear Failure"]["count"] == 0

    def test_by_product_type(self):
        result = tools.get_failure_summary()
        by_type = result["by_product_type"]
        assert by_type["L"]["failures"] == 0
        assert by_type["L"]["total"] == 6
        assert by_type["M"]["failures"] == 1
        assert by_type["M"]["total"] == 3
        assert by_type["H"]["failures"] == 1
        assert by_type["H"]["total"] == 1


class TestPlotTrend:
    def test_creates_png_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
        result = tools.plot_trend("torque_nm")
        output_path = tmp_path / "trend_torque_nm.png"
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert result["image_path"] == str(output_path)

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError):
            tools.plot_trend("humidity")


class TestToolRegistry:
    def test_every_schema_has_a_dispatch_entry(self):
        schema_names = {schema["name"] for schema in tools.TOOL_SCHEMAS}
        dispatch_names = set(tools.TOOL_DISPATCH.keys())
        assert schema_names == dispatch_names

    def test_results_are_json_serializable(self):
        import json

        for name, func in tools.TOOL_DISPATCH.items():
            if name == "get_failure_summary":
                result = func()
            elif name == "get_correlation":
                result = func("air_temperature_k", "process_temperature_k")
            else:
                result = func("torque_nm")
            json.dumps(result)  # raises if anything (e.g. a numpy type) isn't serializable
