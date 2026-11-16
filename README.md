# CodeGuard AI

### AI-Powered Multilingual Source Code Vulnerability Analysis

CodeGuard AI is a Streamlit application that uses a fine-tuned CodeBERT model to classify source code snippets as potentially vulnerable or non-vulnerable.

## Features

- Source code vulnerability classification
- Local model inference
- Multilingual code support
- Vulnerability probability and confidence display
- Analysis history and JSON export
- Model evaluation metrics and diagnostics
- Model Center with evaluation charts

## Supported Languages

- Python
- Java
- JavaScript
- C and C++
- C#

## Technology Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Microsoft CodeBERT
- NumPy and Pandas
- Scikit-learn

## Project Structure

```text
CodeGuard_AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── codeguard-multilingual-v1/
│       └── best_model/
├── results/
│   └── model_metrics.json
└── src/
    └── multilingual_model.py
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CodeGuard_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install PyTorch

Install a PyTorch build compatible with your operating system and hardware.

For CPU-only installation:

```bash
python -m pip install torch
```

For NVIDIA GPU support, use the appropriate CUDA installation command from the official PyTorch installation selector.

### 4. Install project dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Add the trained model

The trained model weights are not included in this Git repository.

Place the model files and tokenizer files in:

```text
models/codeguard-multilingual-v1/best_model/
```

The directory must contain the required Hugging Face model and tokenizer files.

A public model download location has not yet been configured.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Model Evaluation

The following results were obtained on a held-out test dataset of 33,703 examples.

| Metric | Score |
|---|---:|
| Accuracy | 83.68% |
| Precision | 21.36% |
| Recall | 43.66% |
| F1-score | 28.69% |

Classification threshold: 0.008

These metrics are dataset-level evaluation results. They do not guarantee the correctness of individual predictions.

## Important Limitations

- The model can produce false-positive and false-negative predictions.
- Model performance varies across programming languages.
- Input is limited to 512 tokens; longer code may be truncated.
- Vulnerability localization is not enabled.
- Dataset labels may contain noise.
- This tool is an experimental code-analysis aid, not a replacement for professional security review, static analysis, or penetration testing.

Do not rely on CodeGuard as the sole security control for production software.

## Model and Dataset

Base model: `microsoft/codebert-base`

Training dataset: Multilingual Vulnerability Detection (MVD).

The model weights and training datasets are excluded from this repository.

## License

No license has been selected yet. All rights reserved by default until a license is added.

## Author

Developed as an AI and cybersecurity project.
