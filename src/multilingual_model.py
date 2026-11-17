from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_REPO_ID = "hemu10-08/CodeGuard-Multilingual-CodeBERT"

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "codeguard-multilingual-v1"
)

MODEL_PATH = MODEL_DIR / "best_model"

MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.008

LABELS = {
    0: "NON_VULNERABLE",
    1: "VULNERABLE",
}


def ensure_model_available():
    """Use the local model or download required files from Hugging Face."""

    required_files = [
        MODEL_PATH / "config.json",
        MODEL_PATH / "model.safetensors",
        MODEL_PATH / "tokenizer.json",
        MODEL_PATH / "tokenizer_config.json",
    ]

    if all(path.is_file() for path in required_files):
        print("Using existing local CodeGuard model.")
        return

    print("Downloading CodeGuard model from Hugging Face...")

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        local_dir=str(MODEL_DIR),
        allow_patterns=[
            "best_model/config.json",
            "best_model/model.safetensors",
            "best_model/tokenizer.json",
            "best_model/tokenizer_config.json",
            "inference_config.json",
        ],
    )

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Model download incomplete. Missing files: "
            + ", ".join(missing)
        )

    print("CodeGuard model downloaded successfully.")


def load_multilingual_model():
    """Load the trained CodeGuard multilingual model."""

    ensure_model_available()

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
