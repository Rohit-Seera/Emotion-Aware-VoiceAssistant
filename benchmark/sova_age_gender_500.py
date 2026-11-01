from __future__ import annotations

import json
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import AutoProcessor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel


# ============================================================
# CONFIG
# ============================================================

N = 500

DATASET = "saeedzou/common-voice-17-en-age-gender-sampled"
MODEL_NAME = "audeering/wav2vec2-large-robust-6-ft-age-gender"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_sova_age_gender_500"
RESULTS_PATH = OUTPUT_DIR / "results.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

DEVICE = "cpu"


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        features = self.dropout(features)
        features = torch.tanh(self.dense(features))
        features = self.dropout(features)
        return self.out_proj(features)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
        age = self.age(hidden_states)
        gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, age, gender


# ============================================================
# SOVA AGE MAPPING
# ============================================================

def age_to_sova_class(age_label: str) -> str | None:
    """
    Common Voice age labels -> 5 broad SOVA-style age groups.

    We use:
      1 = child/teen
      2 = 20s/30s
      3 = 40s/50s
      4 = 60s/70s
      5 = 80s/90s
    """

    age = str(age_label).lower().strip()

    if age in {"teens", "teens"}:
        return "1"

    if age in {"twenties", "thirties"}:
        return "2"

    if age in {"fourties", "fifties"}:
        return "3"

    if age in {"sixties", "seventies"}:
        return "4"

    if age in {"eighties", "nineties"}:
        return "5"

    return None


def predicted_age_to_sova_class(age_years: float) -> str:
    if age_years < 20:
        return "1"

    if age_years < 40:
        return "2"

    if age_years < 60:
        return "3"

    if age_years < 80:
        return "4"

    return "5"


def normalize_gender(label: str) -> str | None:
    label = str(label).lower().strip()

    if "female" in label:
        return "female"

    if "male" in label:
        return "male"

    return None


def predicted_gender(
    probs: np.ndarray,
) -> str:

    # Model output:
    # [child, female, male]

    child = float(probs[0])
    female = float(probs[1])
    male = float(probs[2])

    # SOVA is 2-class, so we ignore child probability
    # and choose between female/male.
    return "female" if female >= male else "male"


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Common Voice subset...")

    dataset = load_dataset(
        DATASET,
        split="train",
        streaming=True,
    )

    try:
        dataset = dataset.cast_column(
            "audio",
            Audio(decode=False),
        )
    except Exception:
        pass

    print("Loading age/gender model...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AgeGenderModel.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded.")

    results = []

    count = 0

    for item in dataset:

        if count >= N:
            break

        age_class = age_to_sova_class(
            item.get("age", "")
        )

        gender_class = normalize_gender(
            item.get("gender", "")
        )

        # Ignore records without valid labels.
        if age_class is None or gender_class is None:
            continue

        audio = item["audio"]

        audio_source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
        array, sampling_rate = sf.read(audio_source, dtype="float32")
        array = np.asarray(array, dtype=np.float32)

        # Model expects 16 kHz.
        if sampling_rate != 16000:
            array = librosa.resample(
                array,
                orig_sr=sampling_rate,
                target_sr=16000,
            )
            sampling_rate = 16000

        started = time.perf_counter()

        inputs = processor(
            array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_values = inputs[
            "input_values"
        ].to(DEVICE)

        with torch.no_grad():

            output = model(
                input_values
            )

            # Hugging Face model output:
            # hidden_states, age_logits, gender_probs

            age_raw = output[1]
            gender_probs = output[2]

            age_value = float(
                age_raw.squeeze().detach().cpu().numpy()
            )

            gender_probs = (
                gender_probs
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        # Model age is approximately 0...1
        # => convert to years.
        predicted_age_years = (
            age_value * 100.0
        )

        predicted_age_class = (
            predicted_age_to_sova_class(
                predicted_age_years
            )
        )

        predicted_gender_class = (
            predicted_gender(
                gender_probs
            )
        )

        age_correct = (
            predicted_age_class
            == age_class
        )

        gender_correct = (
            predicted_gender_class
            == gender_class
        )

        count += 1

        result = {
            "id": count,
            "age_ground_truth": item.get("age"),
            "age_sova_class": age_class,
            "predicted_age_years": predicted_age_years,
            "predicted_age_sova_class":
                predicted_age_class,

            "gender_ground_truth":
                gender_class,

            "predicted_gender":
                predicted_gender_class,

            "age_correct": age_correct,
            "gender_correct": gender_correct,

            "latency_ms": latency_ms,

            "success": True,
        }

        results.append(result)

        print(
            f"{count:03d}/{N} "
            f"age={age_correct} "
            f"gender={gender_correct} "
            f"pred_age={predicted_age_years:.1f} "
            f"pred_gender={predicted_gender_class} "
            f"latency={latency_ms:.1f}ms"
        )

        RESULTS_PATH.write_text(
            json.dumps(
                results,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    age_accuracy = (
        sum(
            r["age_correct"]
            for r in results
        )
        / len(results)
        if results else 0.0
    )

    gender_accuracy = (
        sum(
            r["gender_correct"]
            for r in results
        )
        / len(results)
        if results else 0.0
    )

    mean_latency = (
        sum(
            r["latency_ms"]
            for r in results
        )
        / len(results)
        if results else 0.0
    )

    summary = {
        "dataset": DATASET,
        "target_samples": N,
        "successful_cases": len(results),
        "failed_cases":
            N - len(results),

        "age_accuracy":
            age_accuracy,

        "gender_accuracy":
            gender_accuracy,

        "mean_latency_ms":
            mean_latency,

        "model":
            MODEL_NAME,

        "device":
            DEVICE,

        "age_metric":
            "5-class accuracy",

        "gender_metric":
            "2-class accuracy",
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 55)
    print("AGE + GENDER 500 COMPLETE")
    print("=" * 55)

    print(
        f"Age Accuracy: "
        f"{age_accuracy * 100:.2f}%"
    )

    print(
        f"Gender Accuracy: "
        f"{gender_accuracy * 100:.2f}%"
    )

    print(
        f"Mean latency: "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
