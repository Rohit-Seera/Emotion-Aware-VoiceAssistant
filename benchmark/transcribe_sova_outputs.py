"""Backward-compatible entry point for local Whisper-large-v3 transcription.

This script no longer calls hosted transcription APIs. Use the local open Whisper
checkpoint named in the SOVA paper via faster-whisper; first run may download weights.
See `benchmark/transcribe_sova_outputs_local.py` for all flags and output semantics.
"""
from benchmark.transcribe_sova_outputs_local import main

if __name__ == "__main__":
    main()
