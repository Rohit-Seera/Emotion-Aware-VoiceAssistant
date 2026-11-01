"""Optional acoustic speech-emotion recognition for the voice companion.

The research path uses the public SUPERB Wav2Vec2 emotion-recognition model when
`transformers` is installed and `VOICE_EMOTION_ACOUSTIC_ENABLED=true`. Inference
is lazy and runs off the FastAPI event loop. If the model cannot be loaded, the
hybrid detector falls back to the existing lexical/prosody-safe path instead of
breaking voice conversations.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
from pathlib import Path
from typing import Any
import wave


_LABEL_MAP = {
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "hap": "happy",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "neu": "neutral",
    "neutral": "neutral",
    "fear": "fearful",
    "fearful": "fearful",
    "fru": "frustrated",
    "frustrated": "frustrated",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault(
    "HF_HOME",
    str(_PROJECT_ROOT / "runtime_models" / "huggingface"),
)


class AcousticEmotionUnavailable(RuntimeError):
    """Raised when the local acoustic SER path is unavailable."""


class Wav2Vec2AcousticEmotionDetector:
    """Lazy Wav2Vec2 acoustic emotion classifier.

    The model is trained for 16 kHz speech, so incoming PCM WAV is decoded and
    resampled in memory before inference. The class never writes user audio to
    disk.
    """

    def __init__(
        self,
        *,
        model_name: str,
        enabled: bool = True,
        min_confidence: float = 0.45,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.min_confidence = min_confidence
        self._classifier: Any | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether the local classifier has been loaded."""

        return self._classifier is not None

    @property
    def available(self) -> bool:
        """Whether acoustic inference is currently usable."""

        return self.enabled and self._classifier is not None

    @property
    def load_error(self) -> str | None:
        """Return the last lazy-loading error, when any."""

        return self._load_error

    def status(self) -> dict[str, Any]:
        """Return safe diagnostics for the research demo UI."""

        transformers_available = False
        torch_available = False

        try:
            import transformers  # noqa: F401

            transformers_available = True
        except Exception:
            pass

        try:
            import torch  # noqa: F401

            torch_available = True
        except Exception:
            pass

        return {
            "enabled": self.enabled,
            "model": self.model_name,
            "loaded": self.loaded,
            "transformers_available": transformers_available,
            "torch_available": torch_available,
            "load_error": self._load_error,
            "status": (
                "ready"
                if self.available
                else (
                    "load-failed"
                    if self._load_error
                    else "lazy"
                )
            ),
        }

    async def warmup(self) -> dict[str, Any]:
        """Load the model without blocking the async application."""

        if not self.enabled:
            return self.status()

        await asyncio.to_thread(self._load)
        return self.status()

    def _load(self) -> None:
        """Load the Transformers pipeline once per process."""

        if self._classifier is not None:
            return

        with self._load_lock:
            if self._classifier is not None:
                return

            try:
                from transformers import pipeline

                self._classifier = pipeline(
                    "audio-classification",
                    model=self.model_name,
                )
                self._load_error = None
            except Exception as error:
                self._load_error = (
                    f"{error.__class__.__name__}: {str(error).strip()}"
                )
                raise AcousticEmotionUnavailable(
                    "Acoustic emotion model could not be loaded."
                ) from error

    async def detect(
        self,
        audio: bytes,
    ) -> dict[str, Any] | None:
        """Infer emotion from PCM WAV bytes.

        Returns `None` for unsupported/invalid audio so the caller can use the
        text-based fallback without turning a voice turn into an error.
        """

        if not self.enabled or not audio:
            return None

        try:
            samples, sample_rate = _decode_pcm_wav(audio)
        except Exception as error:
            self._load_error = (
                f"Audio decode skipped: {error.__class__.__name__}"
            )
            return None

        try:
            await asyncio.to_thread(self._load)
        except AcousticEmotionUnavailable:
            return None

        try:
            return await asyncio.to_thread(
                self._predict,
                samples,
                sample_rate,
            )
        except Exception as error:
            self._load_error = (
                f"Inference skipped: {error.__class__.__name__}: "
                f"{str(error).strip()}"
            )
            return None

    def _predict(
        self,
        samples: Any,
        sample_rate: int,
    ) -> dict[str, Any]:
        """Run a serialized local classifier call."""

        import numpy as np

        if not isinstance(samples, np.ndarray) or samples.size == 0:
            raise AcousticEmotionUnavailable("Empty audio sample.")

        target_rate = 16_000
        if sample_rate != target_rate:
            samples = _resample_linear(
                samples,
                sample_rate,
                target_rate,
            )
            sample_rate = target_rate

        # Avoid feeding hours of audio into the classifier if a caller sends an
        # unexpectedly long WAV. Voice endpoints already enforce a byte ceiling,
        # but this keeps inference bounded too.
        max_samples = target_rate * 30
        if samples.size > max_samples:
            samples = samples[:max_samples]

        with self._inference_lock:
            classifier = self._classifier
            if classifier is None:
                raise AcousticEmotionUnavailable("Classifier is not loaded.")

            raw = classifier(
                {
                    "raw": samples.astype(np.float32, copy=False),
                    "sampling_rate": sample_rate,
                },
                top_k=4,
            )

        predictions: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            raw_label = str(item.get("label", "neutral"))
            normalized = _LABEL_MAP.get(
                raw_label.casefold(),
                raw_label.casefold(),
            )

            score = float(item.get("score", 0.0))
            predictions.append(
                {
                    "emotion": normalized,
                    "confidence": round(score, 4),
                    "raw_label": raw_label,
                }
            )

        if not predictions:
            raise AcousticEmotionUnavailable(
                "The acoustic classifier returned no predictions."
            )

        predictions.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        best = predictions[0]
        return {
            "emotion": best["emotion"],
            "confidence": best["confidence"],
            "source": "acoustic_wav2vec2",
            "model": self.model_name,
            "sample_rate": sample_rate,
            "top_emotions": predictions,
        }


def _decode_pcm_wav(audio: bytes) -> tuple[Any, int]:
    """Decode a mono/stereo 16-bit PCM WAV entirely in memory."""

    import numpy as np

    with wave.open(io.BytesIO(audio), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if width != 2:
        raise ValueError(
            "Acoustic SER currently expects 16-bit PCM WAV input."
        )

    samples = np.frombuffer(
        raw,
        dtype="<i2",
    ).astype(np.float32)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    samples /= 32768.0
    return samples, int(sample_rate)


def _resample_linear(
    samples: Any,
    source_rate: int,
    target_rate: int,
) -> Any:
    """Resample a small waveform with dependency-light linear interpolation."""

    import numpy as np

    if samples.size == 0 or source_rate == target_rate:
        return samples

    target_length = max(
        1,
        int(round(samples.size * target_rate / source_rate)),
    )

    source_positions = np.linspace(
        0.0,
        1.0,
        num=samples.size,
    )
    target_positions = np.linspace(
        0.0,
        1.0,
        num=target_length,
    )

    return np.interp(
        target_positions,
        source_positions,
        samples,
    ).astype(np.float32)
