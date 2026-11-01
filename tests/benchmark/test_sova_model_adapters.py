from __future__ import annotations

import json
from pathlib import Path

from benchmark.compare_sova_table4 import aggregate_model, build_table
from benchmark.sova_model_adapters import (
    MiniOmni2HTTPAdapter,
    canonical_task,
    load_registry,
    run_model,
)


def test_registry_contains_six_open_paper_models_and_weings_ai():
    registry = load_registry()
    ids = {item["id"] for item in registry["models"]}
    assert ids == {
        "mini_omni", "mini_omni2", "moshi", "llama_omni", "freeze_omni",
        "glm4_voice", "weings_ai",
    }
    assert "cascade" not in ids
    assert "cascade" in {item["id"] for item in registry["excluded_models"]}


def test_model_that_has_dash_in_paper_is_excluded_not_zero_scored(tmp_path: Path):
    rows = run_model(
        "moshi",
        [{"case_id": "emotion_1", "task": "emotion_recognition", "audio_path": "missing.wav"}],
        tmp_path,
    )
    assert len(rows) == 1
    assert rows[0]["task"] == "not_evaluated_by_paper"
    assert rows[0]["measurement_status"] == "not_evaluated_by_paper"
    assert rows[0]["requested_task"] == "emotion_recognition"
    assert rows[0]["success"] is False
    summary = aggregate_model("Moshi", rows)
    assert summary["emotion_accuracy_pct"] is None


def test_mini_omni2_multipart_parser_extracts_text_and_audio_part():
    content_type = "multipart/x-mixed-replace; boundary=frame"
    wav_payload = b"RIFF" + b"x" * 32
    body = (
        b"--frame\r\nContent-Type: text/plain\r\n\r\nHello there.\r\n"
        b"--frame\r\nContent-Type: audio/wav\r\n\r\n" + wav_payload + b"\r\n"
        b"--frame--\r\n"
    )
    parts = MiniOmni2HTTPAdapter.__dict__  # class existence/surface remains lightweight
    del parts
    from benchmark.sova_model_adapters import _parse_multipart_response

    parsed = _parse_multipart_response(content_type, body)
    assert any(mime == "text/plain" and payload == b"Hello there." for mime, payload in parsed)
    assert any(mime == "audio/wav" and payload.startswith(b"RIFF") for mime, payload in parsed)


def test_task_aliases_map_to_paper_dimensions():
    assert canonical_task("Spoken SQuAD") == "spoken_qa_open_ended"
    assert canonical_task("ASR") == "speech_recognition"
    assert canonical_task("UTMOSv2") == "generation_acoustic"


def test_table4_rows_keep_paper_reference_separate_from_new_runs(tmp_path: Path):
    runs = tmp_path / "runs"
    result_dir = runs / "weings_ai"
    result_dir.mkdir(parents=True)
    (result_dir / "results.json").write_text(json.dumps([
        {"model": "weings_ai", "model_name": "Emotion Aware Voice Assistant", "task": "generation_semantic", "success": True,
         "gptscore": 3.5, "metric_method": "test"}
    ]), encoding="utf-8")
    rows = build_table(runs, include_reference=True)
    paper_rows = [r for r in rows if r["source_type"] == "published_paper_table4_reference"]
    measured_rows = [r for r in rows if r["source_type"] == "measured_local_run"]
    assert len(paper_rows) == 7
    assert len(measured_rows) == 1
    assert measured_rows[0]["model"] == "Emotion Aware Voice Assistant"
    assert measured_rows[0]["generation_gptscore"] == 3.5
    assert all(r["measurement_status"] == "published_reference_not_reproduced" for r in paper_rows)


def test_glm_missing_split_is_excluded_when_paper_reports_only_test_splits(tmp_path: Path):
    rows = run_model(
        "glm4_voice",
        [{"case_id": "asr_without_split", "task": "speech_recognition", "audio_path": "missing.wav"}],
        tmp_path,
    )
    assert rows[0]["task"] == "not_evaluated_by_paper"
    assert rows[0]["measurement_status"] == "not_evaluated_by_paper"


def test_bridge_ignores_non_json_progress_logs(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace
    import benchmark.sova_model_adapters as module

    monkeypatch.setenv("SOVA_MOSHI_BRIDGE_CMD", "fake-bridge")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='Loading checkpoint...\n{"case_id":"x1","success":true,"prediction":"hello","model_text_output":"hello"}\n',
            stderr="model loaded",
        ),
    )
    rows = run_model(
        "moshi",
        [{"case_id": "x1", "task": "generation_semantic", "audio_path": "placeholder.wav"}],
        tmp_path,
    )
    assert rows[0]["success"] is True
    assert rows[0]["prediction"] == "hello"


def test_attach_acoustic_metrics_adds_separate_generation_acoustic_rows(tmp_path: Path, monkeypatch):
    import sys
    from benchmark import attach_sova_metrics

    results_path = tmp_path / "results.json"
    audio_results_path = tmp_path / "acoustic_results.json"
    results_path.write_text(json.dumps([
        {"case_id": "gen_1", "model": "weings_ai", "model_name": "Emotion Aware Voice Assistant", "task": "generation_semantic", "success": True, "prediction": "hello"}
    ]), encoding="utf-8")
    audio_results_path.write_text(json.dumps([
        {"case_id": "gen_1", "model": "weings_ai", "model_name": "Emotion Aware Voice Assistant", "success": True, "utmosv2": 3.7, "utmosv2_percent": 74.0, "audio_file": "gen_1.wav"},
        {"case_id": "gen_2", "model": "weings_ai", "model_name": "Emotion Aware Voice Assistant", "success": False, "utmosv2": None, "error": "bad audio"},
    ]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["attach_sova_metrics", "--results", str(results_path), "--acoustic-results", str(audio_results_path)])
    attach_sova_metrics.main()
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    metric_rows = [row for row in rows if row.get("task") == "generation_acoustic"]
    assert len(metric_rows) == 2
    assert metric_rows[0]["utmosv2_score"] == 3.7
    assert metric_rows[0]["success"] is True
    assert metric_rows[1]["utmosv2_score"] is None
    assert metric_rows[1]["success"] is False
    assert results_path.with_name("results.pre_acoustic_attach.json").exists()


def test_transcribe_defaults_to_generation_task_only(tmp_path: Path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from benchmark import transcribe_sova_outputs_local as transcriber

    qa_audio = tmp_path / "qa.wav"
    gen_audio = tmp_path / "gen.wav"
    qa_audio.write_bytes(b"fake wav")
    gen_audio.write_bytes(b"fake wav")
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps([
        {"case_id": "qa_1", "model": "weings_ai", "task": "spoken_qa_open_ended", "success": True,
         "model_text_output": "Paris", "response_audio_path": str(qa_audio)},
        {"case_id": "gen_1", "model": "weings_ai", "task": "generation_semantic", "success": True,
         "model_text_output": "Hello there.", "response_audio_path": str(gen_audio)},
    ]), encoding="utf-8")
    seen = []
    class FakeWhisper:
        def __init__(self, *args, **kwargs):
            pass
        def transcribe(self, path, **kwargs):
            seen.append(Path(path).name)
            return iter([SimpleNamespace(text="Hello there.")]), SimpleNamespace(language="en")
    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = FakeWhisper
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    output_path = tmp_path / "results.local_asr.json"
    monkeypatch.setattr(sys, "argv", ["transcribe_sova_outputs_local", "--input", str(results_path), "--output", str(output_path)])
    transcriber.main()
    assert seen == ["gen.wav"]
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert not any(row.get("case_id") == "qa_1" for row in rows)
    assert any(row.get("case_id") == "gen_1" and row.get("task") == "generation_consistency_local" for row in rows)


def test_table4_case_counts_are_unique_across_metric_rows():
    rows = [
        {"case_id": "same_case", "model": "weings_ai", "model_name": "Emotion Aware Voice Assistant", "task": "generation_semantic", "success": True, "gptscore": 4},
        {"case_id": "same_case", "model": "weings_ai", "model_name": "Emotion Aware Voice Assistant", "task": "generation_acoustic", "success": True, "utmosv2_score": 3.5},
    ]
    summary = aggregate_model("Emotion Aware Voice Assistant", rows)
    assert summary["n_cases"] == 1
    assert summary["n_success"] == 1
    assert summary["generation_gptscore"] == 4
    assert summary["generation_utmosv2_score"] == 3.5
