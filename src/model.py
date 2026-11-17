from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

FULL_MODEL_PATH = MODELS_DIR / "codebert-full"
UPLOADED_MODEL_PATH = MODELS_DIR / "codebert-uploaded"
SAVED_TEST_MODEL_PATH = MODELS_DIR / "codebert-save-test"
TEST_TOKENIZER_PATH = MODELS_DIR / "codebert-test"

# Keep this equal to the MAX_LENGTH used when fine-tuning train_full.py.
MAX_LENGTH = 128

# CodeXGlue defect-detection labels:
# 0 = SAFE / non-defective, 1 = VULNERABLE / defective.
LABELS = {0: "SAFE", 1: "VULNERABLE"}
WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")

# Cache loaded objects inside this module so Streamlit can reuse them.
_TOKENIZER = None
_MODEL = None
_DEVICE = None
_MODEL_PATH = None


# ============================================================
# MODEL FILE CHECKS AND SELECTION
# ============================================================

def has_model_weights(path: str | Path) -> bool:
    """Return True only when a supported weight file exists directly in path."""
    folder = Path(path)
    return folder.is_dir() and any(
        (folder / filename).is_file() for filename in WEIGHT_FILENAMES
    )


def select_model_path() -> Path:
    """
    Select a local fine-tuned model.

    Priority:
      1. CODEGUARD_MODEL_PATH, if explicitly configured and valid
      2. models/codebert-full (the main fine-tuned project model)
      3. models/codebert-uploaded
      4. known full-training checkpoints
      5. models/codebert-save-test

    This function never writes to or modifies model files.
    """
    candidates: list[Path] = []

    override = os.environ.get("CODEGUARD_MODEL_PATH", "").strip()
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            override_path = BASE_DIR / override_path
        candidates.append(override_path)

    # Prefer the main trained model over an older uploaded model.
    candidates.extend(
        [
            FULL_MODEL_PATH,
            UPLOADED_MODEL_PATH,
            FULL_MODEL_PATH / "checkpoint-5000",
            FULL_MODEL_PATH / "checkpoint-2500",
            SAVED_TEST_MODEL_PATH,
        ]
    )

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)

        if has_model_weights(resolved):
            print(f"Using local CodeGuard model: {resolved}")
            return resolved

    locations = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "No local model weights were found. Expected model.safetensors "
        "or pytorch_model.bin in:\n" + locations
    )


def tokenizer_folder(model_path: str | Path) -> Path:
    """Find tokenizer files beside the selected model or in known local folders."""
    model_folder = Path(model_path).resolve()
    candidates = [
        model_folder,
        model_folder.parent,
        FULL_MODEL_PATH,
        TEST_TOKENIZER_PATH,
        SAVED_TEST_MODEL_PATH,
    ]

    seen: set[str] = set()
    for candidate in candidates:
        folder = Path(candidate).resolve()
        if str(folder) in seen:
            continue
        seen.add(str(folder))

        has_tokenizer_json = (folder / "tokenizer.json").is_file()
        has_roberta_vocab = (
            (folder / "vocab.json").is_file()
            and (folder / "merges.txt").is_file()
        )
        if has_tokenizer_json or has_roberta_vocab:
            return folder

    raise FileNotFoundError(
        "Tokenizer files were not found. Expected tokenizer.json, or both "
        "vocab.json and merges.txt, beside the model or in models/codebert-test."
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(force_reload: bool = False):
    """
    Load the selected local fine-tuned CodeBERT model.

    Returns:
        tokenizer, model, device

    Uses local_files_only=True so loading does not download or replace weights.
    """
    global _TOKENIZER, _MODEL, _DEVICE, _MODEL_PATH

    selected_path = select_model_path()

    if (
        not force_reload
        and _TOKENIZER is not None
        and _MODEL is not None
        and _DEVICE is not None
        and _MODEL_PATH == selected_path
    ):
        return _TOKENIZER, _MODEL, _DEVICE

    tok_path = tokenizer_folder(selected_path)

    print("=" * 65)
    print("CODEGUARD LOCAL MODEL LOADER")
    print("=" * 65)
    print("Model folder     :", selected_path)
    print("Tokenizer folder :", tok_path)
    print("Maximum length   :", MAX_LENGTH)
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_path),
        local_files_only=True,
    )

    print("Loading model weights...")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(selected_path),
        local_files_only=True,
    )

    if int(model.config.num_labels) != 2:
        raise ValueError(
            f"Expected a 2-class classifier, but model config has "
            f"num_labels={model.config.num_labels}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Only publish the cache after all loading steps succeeded.
    _TOKENIZER = tokenizer
    _MODEL = model
    _DEVICE = device
    _MODEL_PATH = selected_path

    print("Model loaded successfully.")
    print("Device           :", device)
    print("Number of labels :", model.config.num_labels)
    print("Model parameters :", f"{sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model, device


# ============================================================
# PREDICTION
# ============================================================

def predict_vulnerability(
    code: str,
    tokenizer=None,
    model=None,
    device=None,
) -> dict[str, Any]:
    """
    Classify source code and return the actual model probabilities.

    Can be called as:
        predict_vulnerability(code)
    or:
        predict_vulnerability(code, tokenizer, model, device)

    Confidence is the softmax probability of the predicted class. It is not
    model accuracy, and this function deliberately does not artificially
    increase, calibrate, or alter the returned probability.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Please provide non-empty source code.")

    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = load_model()

    model.eval()

    encoded = tokenizer(
        code,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {
        key: tensor.to(device)
        for key, tensor in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**encoded)

    logits = outputs.logits
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected classifier output shape {tuple(logits.shape)}; "
            "expected [1, 2] for SAFE and VULNERABLE."
        )

    probabilities = torch.softmax(logits.float(), dim=-1)[0]
    predicted_class = int(torch.argmax(probabilities).item())

    safe_probability = float(probabilities[0].item())
    vulnerable_probability = float(probabilities[1].item())
    confidence = float(probabilities[predicted_class].item())

    return {
        "result": LABELS[predicted_class],
        "label": LABELS[predicted_class],
        "predicted_class": predicted_class,
        "confidence": confidence,  # fraction from 0.0 to 1.0
        "safe_probability": safe_probability,
        "vulnerable_probability": vulnerable_probability,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> dict[str, Any]:
    """Return local model artifact information for the Streamlit diagnostics page."""
    path = select_model_path()
    weight_path = next(
        (path / name for name in WEIGHT_FILENAMES if (path / name).is_file()),
        None,
    )
    tokenizer_path = tokenizer_folder(path)

    return {
        "model_path": str(path),
        "tokenizer_path": str(tokenizer_path),
        "weights_file": weight_path.name if weight_path else None,
        "weights_size_bytes": weight_path.stat().st_size if weight_path else 0,
        "config_exists": (path / "config.json").is_file(),
        "tokenizer_exists": True,
        "device_available": "cuda" if torch.cuda.is_available() else "cpu",
    }


# ============================================================
# TERMINAL SMOKE TEST
# ============================================================

def run_terminal_demo():
    """Run a small smoke test; predictions are model outputs, not guarantees."""
    tokenizer, model, device = load_model()

    examples = [
        (
            "Simple addition (expected safe pattern; model decides)",
            "int add(int a, int b) { return a + b; }",
        ),
        (
            "Unbounded input with gets (known unsafe coding pattern)",
            '#include <stdio.h>\n'
            'int main(void) { char input[10]; gets(input); return 0; }',
        ),
    ]

    for name, code in examples:
        result = predict_vulnerability(code, tokenizer, model, device)
        print("\n" + "-" * 65)
        print(name)
        print("Classification        :", result["result"])
        print(f"Confidence             : {result['confidence'] * 100:.2f}%")
        print(f"SAFE probability       : {result['safe_probability'] * 100:.2f}%")
        print(
            "VULNERABLE probability : "
            f"{result['vulnerable_probability'] * 100:.2f}%"
        )

    print("\nSmoke test completed. Review predictions against ground-truth labels.")


if __name__ == "__main__":
    try:
        run_terminal_demo()
    except Exception as exc:
        print(f"\nCODEGUARD MODEL ERROR: {type(exc).__name__}: {exc}")
        sys.exit(1)
