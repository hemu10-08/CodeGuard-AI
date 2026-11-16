
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "codeguard-multilingual-v1"
    / "best_model"
)

MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.008

LABELS = {
    0: "NON_VULNERABLE",
    1: "VULNERABLE",
}


def load_multilingual_model():
    """Load the trained CodeGuard multilingual model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_multilingual(
    code,
    tokenizer,
    model,
    device,
    threshold=DEFAULT_THRESHOLD,
):
    """Predict whether a code snippet is vulnerable."""

    if not isinstance(code, str) or not code.strip():
        raise ValueError("Please provide non-empty source code.")

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    safe_probability = probabilities[0].item()
    vulnerable_probability = probabilities[1].item()

    predicted_class = int(
        vulnerable_probability >= threshold
    )

    return {
        "result": LABELS[predicted_class],
        "label": LABELS[predicted_class],
        "predicted_class": predicted_class,
        "confidence": probabilities[predicted_class].item(),
        "safe_probability": safe_probability,
        "vulnerable_probability": vulnerable_probability,
        "threshold": threshold,
    }
