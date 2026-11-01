from __future__ import annotations

import json
from pathlib import Path

from benchmark.compare_sova_table4 import build_table
from benchmark.prepare_librispeech_manifest import build_manifest
from benchmark.run_sova_model import load_items


def test_librispeech_manifest_builder_uses_local_paths(tmp_path: Path) -> None:
    root = tmp_path / "LibriSpeech"
    chapter = root / "test-clean" / "1089" / "134686"
    chapter.mkdir(parents=True)
    (chapter / "1089-134686.trans.txt").write_text(
        "1089-134686-0000 HELLO THERE\n"
        "1089-134686-0001 SECOND SAMPLE\n",
        encoding="utf-8",
    )
    (chapter / "1089-134686-0000.flac").write_bytes(b"audio0")
    (chapter / "1089-134686-0001.flac").write_bytes(b"audio1")

    records, summary = build_manifest(root, ["test-clean"], limit_per_split=1)

    assert len(records) == 1
    assert records[0]["task"] == "speech_recognition"
    assert records[0]["dataset_split"] == "test-clean"
    assert Path(records[0]["audio_path"]).is_absolute()
    assert records[0]["reference_text"] == "HELLO THERE"
    assert summary["records_written"] == 1


def test_model_runner_loads_jsonl_records(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id":"a","task":"speech_recognition"}\n'
        '{"case_id":"b","task":"speech_recognition"}\n',
        encoding="utf-8",
    )
    assert [x["case_id"] for x in load_items(path)] == ["a", "b"]


def test_table4_scaffold_has_reference_rows_and_unrun_assistant_placeholder(tmp_path: Path) -> None:
    rows = build_table(tmp_path / "runs")
    paper_rows = [row for row in rows if row["source_type"] == "published_paper_table4_reference"]
    proposed = [row for row in rows if row["source_type"] == "proposed_model_not_run"]
    assert len(paper_rows) == 7
    assert len(proposed) == 1
    assert proposed[0]["model"] == "Emotional Voice Assistant"
    assert proposed[0]["measurement_status"] == "not_run"
    assert proposed[0]["knowledge_accuracy_pct"] is None
