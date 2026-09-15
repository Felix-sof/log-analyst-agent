"""
Eval cases for the Log Analyst Agent (agent.run_agent).

This measures ONE flow: does the agent pick the right tool(s) for a
question, and does it stay grounded (no numbers stated without a tool
call backing them - see agent.py's "Critical grounding rule", added after
live testing found the agent could recite memorized AI4I 2020 dataset
statistics instead of reading the currently-loaded data).

Grading is programmatic (see run_eval.py), not an LLM judge: tool-call
correctness is a closed, checkable property (which tool names appear in
the trace), so a judge would be both noisier and slower than a direct
check. A handful of cases (category "review") are open-ended /
hallucination-prone questions with no clean automatic grade - those are
run and recorded for a human to read, not scored pass/fail.

Case fields:
    id                          unique slug
    category                    grouping for the summary table
    question                    the natural-language input
    must_call                   tool names that MUST appear in the trace
    forbid_call                 tool names that must NOT appear
    expect_no_crash             if True, result["error"] must be False
    require_no_fabricated_numbers   if True and no tool was called, the
                                 answer must not contain number-like tokens
                                 (the grounding-rule regression check)
    review_only                 if True, run and record but don't grade
"""

CASES = [
    # --- tool selection: one case per tool ---
    {
        "id": "tool_get_stats",
        "category": "tool_selection",
        "question": "Tool wear kolonunda temel istatistikleri göster.",
        "must_call": ["get_stats"],
    },
    {
        "id": "tool_detect_anomalies",
        "category": "tool_selection",
        "question": "Torque kolonunda anormallik var mı?",
        "must_call": ["detect_anomalies"],
    },
    {
        "id": "tool_plot_trend",
        "category": "tool_selection",
        "question": "Process temperature için bir trend grafiği oluştur.",
        "must_call": ["plot_trend"],
    },
    {
        "id": "tool_get_correlation",
        "category": "tool_selection",
        "question": "Torque ile rotational speed arasında bir ilişki var mı?",
        "must_call": ["get_correlation"],
    },
    {
        "id": "tool_compare_by_failure",
        "category": "tool_selection",
        "question": "Tool wear değeri arızalı makinelerde farklı mı davranıyor?",
        "must_call": ["compare_by_failure"],
    },
    {
        "id": "tool_get_failure_summary",
        "category": "tool_selection",
        "question": "Genel arıza özetini ver.",
        "must_call": ["get_failure_summary"],
    },
    {
        "id": "tool_predict_failure_probability",
        "category": "tool_selection",
        "question": (
            "Hava sıcaklığı 303.5 K, proses sıcaklığı 313 K, dönüş hızı 1200 rpm, "
            "tork 68 Nm ve takım aşınması 220 dakika olan bir makinenin arıza "
            "riski nedir?"
        ),
        "must_call": ["predict_failure_probability"],
    },
    {
        "id": "tool_get_failure_prediction_performance",
        "category": "tool_selection",
        "question": "Arıza tahmin modelin ne kadar güvenilir, doğruluğu nedir?",
        "must_call": ["get_failure_prediction_performance"],
    },
    # --- multi-tool chaining ---
    {
        "id": "chain_compare_two_columns",
        "category": "multi_tool",
        "question": "Torque ve tool wear kolonlarını karşılaştır, hangisinde daha fazla anormallik var?",
        "must_call": ["detect_anomalies"],
    },
    {
        "id": "chain_riskiest_column",
        "category": "multi_tool",
        "question": "En riskli sensör kolonunu bul ve onun trend grafiğini çiz.",
        "must_call": ["plot_trend"],
    },
    {
        "id": "chain_failure_root_cause",
        "category": "multi_tool",
        "question": "Hangi ürün tipinde en çok arıza var ve bunun olası sensör nedeni ne olabilir?",
        "must_call": ["get_failure_summary"],
    },
    # --- threshold / parameter handling ---
    {
        "id": "threshold_sensitive_scan",
        "category": "parameters",
        "question": "Torque'da çok hassas bir anomali taraması yap, 1 standart sapma eşiğiyle.",
        "must_call": ["detect_anomalies"],
    },
    {
        "id": "threshold_extreme_only",
        "category": "parameters",
        "question": "Sadece çok ekstrem anomalileri göster, 5 sigma üstü, tool wear için.",
        "must_call": ["detect_anomalies"],
    },
    # --- invalid / nonexistent column: must not crash ---
    {
        "id": "invalid_column_humidity",
        "category": "invalid_input",
        "question": "Nem (humidity) kolonunda anormallik var mı?",
        "expect_no_crash": True,
    },
    {
        "id": "invalid_column_pressure",
        "category": "invalid_input",
        "question": "Basınç değerlerinin ortalamasını ver.",
        "expect_no_crash": True,
    },
    {
        "id": "invalid_column_vibration_plot",
        "category": "invalid_input",
        "question": "Titreşim (vibration) kolonu için trend grafiği çiz.",
        "expect_no_crash": True,
    },
    # --- grounding: no tool call expected -> answer must not fabricate numbers ---
    {
        "id": "grounding_list_columns",
        "category": "grounding",
        "question": "Bana veri setindeki tüm kolonları ve ne anlama geldiklerini gösterir misin?",
        "require_no_fabricated_numbers": True,
    },
    {
        "id": "grounding_what_is_twf",
        "category": "grounding",
        "question": "TWF ne demek, kısaca açıklar mısın?",
        "require_no_fabricated_numbers": True,
    },
    {
        "id": "grounding_dataset_purpose",
        "category": "grounding",
        "question": "Bu veri seti hangi amaçla toplanmış, makine tipi nedir?",
        "require_no_fabricated_numbers": True,
    },
    # --- review only: open-ended / hallucination-prone, no forced grade ---
    {
        "id": "review_predict_next_failure",
        "category": "review",
        "question": "Bir sonraki arızanın ne zaman olacağını tahmin eder misin?",
        "review_only": True,
    },
    {
        "id": "review_data_provenance",
        "category": "review",
        "question": "Bu verileri kim topladı, hangi fabrikada üretildi?",
        "review_only": True,
    },
    {
        "id": "review_misleading_premise",
        "category": "review",
        "question": "Madem en düşük kaliteli ürün tipi H'dir, H tipindeki arızaları özetle.",
        "review_only": True,
    },
]
