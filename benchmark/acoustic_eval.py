from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import utmosv2


def find_audio_files(audio_dir: Path, metadata=None, generation_only: bool = False, metadata_path: Path | None = None):
    """Find WAVs, optionally restricting scoring to the Alpaca generation rows.

    Without ``generation_only`` this retains the script's original recursive scan.
    With it, only WAV paths referenced by generation-task records are scored, so
    spoken-QA responses cannot accidentally inflate the Table 4 acoustic score.
    """
    if generation_only:
        if metadata is None or metadata_path is None:
            raise ValueError("--generation-only requires --input results.json")
        generation_tasks = {"generation", "generation_response", "generation_semantic", "speech_generation", "open_ended"}
        candidates = []
        for row in metadata.values():
            task = str(row.get("task", "")).strip().lower().replace("-", "_").replace(" ", "_")
            if task not in generation_tasks:
                continue
            raw = row.get("response_audio_path") or row.get("generated_audio_path") or row.get("audio_file")
            if not raw:
                continue
            path = Path(str(raw))
            if not path.is_absolute():
                options = [(metadata_path.parent / path).resolve(), (audio_dir / path).resolve()]
                path = next((candidate for candidate in options if candidate.is_file()), options[0])
            if path.is_file() and path.suffix.lower() == ".wav":
                candidates.append(path.resolve())
        files = sorted(set(candidates))
    else:
        files = sorted(audio_dir.rglob("*.wav"))
    if not files:
        scope = "generation-task WAVs referenced in results.json" if generation_only else str(audio_dir)
        raise FileNotFoundError(f"No .wav files found for acoustic evaluation inside: {scope}")
    return files


def case_id_from_path(path: Path) -> str:
    """
    Convert filename into case_id.

    generation_001.wav
    -> generation_001
    """
    return path.stem


def main():
    parser = argparse.ArgumentParser(
        description="UTMOSv2 acoustic quality evaluation"
    )

    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing generated WAV files",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Optional generation results.json for metadata",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="UTMOSv2 prediction repetitions per file",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Only score response WAVs belonging to generation tasks in --input, not all voice responses.",
    )

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not audio_dir.exists():
        raise SystemExit(
            f"Audio directory does not exist: {audio_dir}"
        )

    # ---------------------------------------------------------
    # Load optional generation metadata
    # ---------------------------------------------------------

    metadata = {}

    if args.input:
        input_path = Path(args.input)

        if input_path.exists():
            try:
                rows = json.loads(
                    input_path.read_text(
                        encoding="utf-8"
                    )
                )

                # Prefer a generation-task source row for duplicate case IDs: the
                # transcription stage also appends generation_consistency rows.
                generation_tasks = {"generation", "generation_response", "generation_semantic", "speech_generation", "open_ended"}
                metadata = {}
                for row in rows:
                    cid = row.get("case_id")
                    if not cid:
                        continue
                    task = str(row.get("task", "")).strip().lower().replace("-", "_").replace(" ", "_")
                    has_audio = bool(row.get("response_audio_path") or row.get("generated_audio_path") or row.get("audio_file"))
                    previous = metadata.get(cid)
                    previous_task = str((previous or {}).get("task", "")).strip().lower().replace("-", "_").replace(" ", "_")
                    previous_has_audio = bool((previous or {}).get("response_audio_path") or (previous or {}).get("generated_audio_path") or (previous or {}).get("audio_file"))
                    if previous is None or (task in generation_tasks and has_audio) or (not previous_has_audio and has_audio):
                        metadata[cid] = row

            except Exception as exc:
                print(
                    f"Warning: could not read metadata: {exc}"
                )

    # ---------------------------------------------------------
    # Find audio
    # ---------------------------------------------------------

    audio_files = find_audio_files(
        audio_dir,
        metadata=metadata,
        generation_only=args.generation_only,
        metadata_path=Path(args.input).resolve() if args.input else None,
    )

    print(
        f"Found {len(audio_files)} WAV files."
    )

    # ---------------------------------------------------------
    # Load UTMOSv2 once
    # ---------------------------------------------------------

    print(
        "\nLoading pretrained UTMOSv2 model..."
    )

    load_start = time.perf_counter()

    model = utmosv2.create_model(
        pretrained=True,
        device="cpu",
    )

    load_latency = (
        time.perf_counter()
        - load_start
    ) * 1000

    print(
        f"Model loaded in "
        f"{load_latency:.2f} ms\n"
    )

    # ---------------------------------------------------------
    # Evaluate each file
    # ---------------------------------------------------------

    results = []

    for index, audio_path in enumerate(
        audio_files,
        start=1,
    ):
        cid = case_id_from_path(
            audio_path
        )

        print(
            f"[{index}/{len(audio_files)}] "
            f"{cid}",
            end=" ",
            flush=True,
        )

        start = time.perf_counter()

        try:
            mos = model.predict(
                input_path=str(audio_path),
                device="cpu",
                num_repetitions=args.repetitions,
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            # Convert possible numpy/tensor scalar
            mos = float(mos)

            # Clamp only for numerical safety
            mos = max(
                1.0,
                min(5.0, mos),
            )

            mos_pct = (
                mos / 5.0
            ) * 100.0

            item = {
                "case_id": cid,
                "audio_file": str(
                    audio_path
                ),
                "success": True,
                "utmosv2": mos,
                "utmosv2_percent": mos_pct,
                "prediction_latency_ms": elapsed,
                "error": None,
            }

            # Attach generation metadata
            if cid in metadata:
                item["model_name"] = metadata[
                    cid
                ].get("model_name")

                item["provider"] = metadata[
                    cid
                ].get("provider")

                item["model"] = metadata[
                    cid
                ].get("model")

            print(
                f"UTMOSv2={mos:.4f} "
                f"({mos_pct:.2f}%) "
                f"{elapsed:.1f} ms"
            )

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            item = {
                "case_id": cid,
                "audio_file": str(
                    audio_path
                ),
                "success": False,
                "utmosv2": None,
                "utmosv2_percent": None,
                "prediction_latency_ms": elapsed,
                "error": str(exc),
            }

            print(
                f"ERROR: {exc}"
            )

        results.append(item)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_path = (
        output_dir / "acoustic_results.json"
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    good = [
        x
        for x in results
        if x.get("success")
    ]

    failed = [
        x
        for x in results
        if not x.get("success")
    ]

    mean_mos = (
        sum(
            x["utmosv2"]
            for x in good
        )
        / len(good)
        if good
        else None
    )

    mean_pct = (
        sum(
            x["utmosv2_percent"]
            for x in good
        )
        / len(good)
        if good
        else None
    )

    mean_latency = (
        sum(
            x["prediction_latency_ms"]
            for x in good
        )
        / len(good)
        if good
        else None
    )

    summary = {
        "metric": "UTMOSv2",
        "cases_requested": len(results),
        "successful_cases": len(good),
        "failed_cases": len(failed),
        "failed_case_ids": [
            x["case_id"]
            for x in failed
        ],
        "mean_utmosv2": mean_mos,
        "mean_utmosv2_percent": mean_pct,
        "mean_prediction_latency_ms": mean_latency,
        "score_range": "1-5",
        "normalized_percent": "(MOS/5)*100",
        "repetitions": args.repetitions,
        "model": (
            "UTMOSv2 pretrained"
        ),
    }

    summary_path = (
        output_dir
        / "acoustic_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    csv_path = (
        output_dir
        / "acoustic_results.csv"
    )

    fieldnames = [
        "case_id",
        "audio_file",
        "success",
        "utmosv2",
        "utmosv2_percent",
        "prediction_latency_ms",
        "model_name",
        "provider",
        "model",
        "error",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "UTMOSv2 ACOUSTIC EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {results_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {csv_path}"
    )


if __name__ == "__main__":
    main()

