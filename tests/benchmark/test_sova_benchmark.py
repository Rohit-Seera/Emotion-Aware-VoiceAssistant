from __future__ import annotations

from benchmark.score_sova_results import score_rows
from benchmark.sova_metrics import corpus_wer, wer_stats
from benchmark.merge_sova_generation_results import merge


def test_wer_counts_edit_operations_and_corpus_percent():
    stats = wer_stats("the cat sat", "the dog sat")
    assert (stats.substitutions, stats.deletions, stats.insertions, stats.reference_words) == (1, 0, 0, 3)
    overall = corpus_wer([("the cat sat", "the dog sat"), ("hello world", "hello world")])
    assert overall["wer_percent"] == 20.0
    assert overall["reference_words"] == 5


def test_wer_can_exceed_one_hundred_percent():
    stats = wer_stats("yes", "no maybe certainly")
    assert stats.errors == 3
    assert stats.wer_percent == 300.0


def test_results_are_grouped_by_model_not_pooled():
    summary = score_rows([
        {"model_name": "A", "task": "emotion", "case_id": "1", "success": True, "answered": True, "expected_label": "happy", "predicted_label": "happy"},
        {"model_name": "B", "task": "emotion", "case_id": "1", "success": True, "answered": True, "expected_label": "happy", "predicted_label": "sad"},
    ])
    assert summary["model_count"] == 2
    assert summary["tasks"] is None  # no pooled single-model summary for multi-model input
    model_a, model_b = summary["models"]
    assert model_a["tasks"]["emotion_recognition"]["accuracy_percent_over_all_cases"] == 100.0
    assert model_b["tasks"]["emotion_recognition"]["accuracy_percent_over_all_cases"] == 0.0


def test_failed_records_do_not_count_as_correct_or_answered():
    summary = score_rows([
        {"model_name": "A", "task": "emotion", "case_id": "1", "success": True, "answered": True, "expected_label": "happy", "predicted_label": "happy"},
        {"model_name": "A", "task": "emotion", "case_id": "2", "success": False, "answered": True, "expected_label": "sad", "predicted_label": "sad"},
    ])
    task = summary["models"][0]["tasks"]["emotion_recognition"]
    assert task["correct_successful_cases"] == 1
    assert task["accuracy_percent_over_all_cases"] == 50.0
    assert task["answer_rate_percent"] == 50.0
    assert task["execution_coverage_percent"] == 50.0


def test_custom_crhi_is_not_reported_as_gptscore():
    summary = score_rows([
        {"model_name": "A", "task": "generation_semantic", "case_id": "generation_001", "success": True,
         "custom_crhi_mean_score_5": 4.0, "custom_crhi_score_pct": 80.0,
         "metric_method": "custom_CRHI_Likert_1_to_5_not_paper_GPTScore"},
    ])
    task = summary["models"][0]["tasks"]["generation_semantic"]
    assert task["mean_gptscore"] is None
    assert task["mean_custom_crhi_score_5"] == 4.0
    assert "not the paper's GPTScore" in task["paper_metric_warning"]


def test_unrun_tasks_stay_visible_without_imputed_scores():
    summary = score_rows([
        {"model_name": "A", "task": "knowledge", "case_id": "1", "success": True,
         "prompt": "capital of France?", "output": "Paris", "correct": True},
    ])
    tasks = summary["models"][0]["tasks"]
    assert tasks["spoken_qa_multiple_choice"]["status"] == "not_run"
    assert tasks["spoken_qa_multiple_choice"]["accuracy_percent_over_all_cases"] is None
    assert tasks["knowledge_qa"]["paper_metric_warning"]


def test_generation_merge_keeps_semantic_and_audio_metrics_separate():
    rows = merge(
        responses=[{"case_id": "generation_001", "model_name": "A", "output": "hi", "success": True}],
        audio=[{"case_id": "generation_001", "reference_text": "hi", "generated_transcript": "hi", "success": True}],
        semantic=[{"case_id": "generation_001", "success": True, "mean_score_5": 4.5, "semantic_score_pct": 90}],
        acoustic=[{"case_id": "generation_001", "success": True, "utmosv2": 3.1}],
    )
    by_task = {row["task"]: row for row in rows}
    assert by_task["generation_consistency"]["generated_transcript"] == "hi"
    assert by_task["generation_semantic"]["custom_crhi_mean_score_5"] == 4.5
    assert "not_paper_GPTScore" in by_task["generation_semantic"]["metric_method"]
    assert by_task["generation_acoustic"]["utmosv2_score"] == 3.1


def test_expected_count_accounts_for_unrecorded_failed_cases():
    summary = score_rows(
        [
            {"model_name": "A", "task": "spoken_squad", "case_id": "1", "success": True, "answered": True, "correct": True, "prediction": "answer"},
            {"model_name": "A", "task": "spoken_squad", "case_id": "2", "success": True, "answered": True, "correct": False, "prediction": "wrong"},
        ],
        expected_counts={"spoken_qa_open_ended": 500},
    )
    task = summary["models"][0]["tasks"]["spoken_qa_open_ended"]
    assert task["unrecorded_expected_cases"] == 498
    assert task["execution_coverage_percent"] == 0.4
    assert task["accuracy_percent_over_all_cases"] == 0.2
    assert task["answer_rate_percent"] == 0.4
