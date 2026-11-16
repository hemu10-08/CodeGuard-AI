from __future__ import annotations
import importlib, json
import math
from datetime import datetime
from pathlib import Path
try:
    pd = importlib.import_module("pandas")
except ModuleNotFoundError:
    pd = None
st = importlib.import_module("streamlit")
try:
    np = importlib.import_module("numpy")
except ModuleNotFoundError:
    np = None

ROOT = Path(__file__).resolve().parent
MODEL_DIRS = [
    ROOT/"models"/"codeguard-multilingual-v1"/"best_model",
    ROOT/"models"/"codebert-full",
    ROOT/"models"/"codebert-uploaded",
    Path("models/codebert-full"),
    Path("models/codebert-uploaded"),
]
WEIGHTS = {"model.safetensors", "pytorch_model.bin"}
METRICS_FILE = ROOT / "results" / "model_metrics.json"

st.set_page_config(page_title="CodeGuard AI | Security Intelligence", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700;800&display=swap');
.stApp{background:radial-gradient(ellipse at 72% -5%,#132a43 0%,#080e16 46%,#070c12 100%);color:#eaf3fc;font-family:Manrope,sans-serif}
[data-testid="stHeader"]{background:rgba(7,12,18,.84)} [data-testid="stSidebar"]{background:#080f18;border-right:1px solid #20374b}
[data-testid="stSidebar"] *{color:#dceaf7} h1,h2,h3{letter-spacing:-.035em;color:#eff7ff}
[data-testid="stMetric"]{background:linear-gradient(145deg,#101e2c,#0b131d);border:1px solid #20384d;border-radius:14px;padding:14px 16px}
[data-testid="stMetricLabel"]{color:#94aabe}[data-testid="stMetricValue"]{color:#f2f8ff}
.stTextArea textarea,.stTextInput input{background:#08111b!important;color:#e7f3ff!important;border:1px solid #25435a!important;border-radius:10px!important;font-family:'DM Mono',monospace!important}
[data-testid="stFileUploader"]{background:#0b1520;border:1px dashed #2c7597;border-radius:12px}
div.stButton>button,div.stDownloadButton>button{background:linear-gradient(100deg,#32f5a0,#28baff);color:#06111a;border:0;border-radius:10px;font-weight:800;min-height:2.65rem}
hr{border-color:#20374b}
.hero{padding:13px 22px;border:1px solid #214661;border-radius:16px;background:linear-gradient(115deg,#0d1f30,#08101b);margin-bottom:10px}
.kicker{color:#32f5a0;font:600 11px 'DM Mono',monospace;letter-spacing:.17em}
.bigtitle{font-size:clamp(25px,2.7vw,34px);line-height:1.08;font-weight:800;margin:5px 0;color:#f1f7ff}
.bigtitle span{color:#32f5a0}.sub{color:#9eb4c9;font-size:12px;line-height:1.5}
.pill{display:inline-block;padding:5px 9px;border-radius:999px;border:1px solid #2c5268;color:#9edcff;background:#0b2030;font:11px 'DM Mono',monospace;margin:3px}
.panel{background:linear-gradient(145deg,#0e1927,#09111b);border:1px solid #1e3448;border-radius:15px;padding:17px}
.caption{color:#91a7bb;font-size:12px;line-height:1.6}
.ok{border:1px solid #218b69;background:#0c2a25;border-radius:13px;padding:15px}
.bad{border:1px solid #9a3849;background:#2a141c;border-radius:13px;padding:15px}
.neutral{border:1px solid #31516a;background:#101f2c;border-radius:13px;padding:15px}
.workspace-toolbar{background:linear-gradient(145deg,#0e1927,#09111b);border:1px solid #1e3448;border-radius:14px;padding:12px 14px;margin-bottom:10px}
.workspace-toolbar [data-testid="stFileUploader"]{padding:7px 10px;min-height:unset}
.workspace-toolbar [data-testid="stFileUploaderDropzone"]{min-height:54px;padding:7px 10px}
.workspace-toolbar [data-testid="stFileUploaderDropzoneInstructions"]{font-size:11px}
.workspace-editor-label{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:5px 0 8px}
.workspace-editor-label .hint{color:#91a7bb;font:11px 'DM Mono',monospace}
.workspace-run button{min-height:3.35rem!important;font-size:15px!important;letter-spacing:.02em;box-shadow:0 0 22px rgba(50,245,160,.13)}
.result-panel{border:1px solid #25445b;background:linear-gradient(145deg,#0d1a27,#09111b);border-radius:15px;padding:18px;margin-top:14px}
@media (max-width: 900px){.workspace-toolbar{padding:10px}.workspace-editor-label{align-items:flex-start;flex-direction:column}.result-panel{padding:12px}}

</style>""", unsafe_allow_html=True)

def artifact_status():
    checked=[]; seen=set()
    for candidate in MODEL_DIRS:
        p=candidate.resolve()
        if str(p) in seen: continue
        seen.add(str(p)); checked.append(str(p))
        if p.is_dir():
            files={x.name for x in p.iterdir() if x.is_file()}
            weights=sorted(files & WEIGHTS)
            return {"path":str(p),"exists":True,"weights":weights,
                    "config":(p/"config.json").is_file(),
                    "tokenizer":any((p/n).is_file() for n in ("tokenizer.json","tokenizer_config.json","vocab.json")),
                    "checked":checked}
    return {"path":None,"exists":False,"weights":[],"config":False,"tokenizer":False,"checked":checked}

@st.cache_resource(show_spinner=False)
def get_model_runtime():
    """Load the real local CodeGuard classifier once per Streamlit session cache."""
    from src.multilingual_model import load_multilingual_model
    return load_multilingual_model()


def import_predictor():
    try:
        from src.multilingual_model import predict_multilingual
        return predict_multilingual, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def invoke(fn, code):
    """Call this project's actual predictor with its required model arguments."""
    try:
        tokenizer, model, device = get_model_runtime()
        return fn(code, tokenizer, model, device), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def evaluate_labeled_dataset(df):
    """Evaluate real model predictions against ground-truth labels in a CSV."""
    try:
        metrics = importlib.import_module("sklearn.metrics")
        accuracy_score = metrics.accuracy_score
        precision_score = metrics.precision_score
        recall_score = metrics.recall_score
        f1_score = metrics.f1_score
        confusion_matrix = metrics.confusion_matrix
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for dataset evaluation. Install it with: pip install scikit-learn"
        ) from exc

    required = {"code", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain exactly the required columns 'code' and 'label' (additional columns are allowed).")
    data = df[["code", "label"]].dropna().copy()
    if data.empty:
        raise ValueError("The evaluation dataset contains no usable rows.")

    def parse_label(value):
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        value = str(value).strip().upper()
        if value in {"0", "SAFE"}:
            return 0
        if value in {"1", "VULNERABLE"}:
            return 1
        raise ValueError(f"Unsupported label {value!r}; use SAFE/VULNERABLE or 0/1.")

    data["label"] = data["label"].map(parse_label)
    data["code"] = data["code"].astype(str)
    if not data["code"].str.strip().any():
        raise ValueError("The evaluation dataset contains only empty code.")
    data = data[data["code"].str.strip().ne("")]
    if data["label"].nunique() < 2:
        raise ValueError("Evaluation needs both SAFE and VULNERABLE ground-truth examples.")
    if not predictor:
        raise RuntimeError("The model predictor is unavailable: " + str(predictor_error))

    progress = st.progress(0, text="Starting evaluation…")
    predictions = []
    try:
        for index, row in enumerate(data.itertuples(index=False), start=1):
            raw, error = invoke(predictor, row.code)
            if error:
                raise RuntimeError(f"Inference failed for evaluation row {index}: {error}")
            predicted = normalize(raw)["label"]
            if predicted == "SAFE":
                predictions.append(0)
            elif predicted == "VULNERABLE":
                predictions.append(1)
            else:
                raise ValueError(f"Unrecognized model output for evaluation row {index}: {predicted!r}")
            progress.progress(index / len(data), text=f"Evaluating {index}/{len(data)} examples…")
    finally:
        progress.empty()

    actual = data["label"].tolist()
    return {
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1_score": float(f1_score(actual, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(actual, predictions)),
        "evaluation_samples": len(actual),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=[0, 1]).tolist(),
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def run_manual_finetuning(examples, output_dir, epochs, batch_size, learning_rate, max_length, progress_bar, status_box, log_box):
    """Fine-tune the local sequence classifier using manually labeled examples."""
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("Manual fine-tuning requires PyTorch. Install it in the active Python environment.") from exc
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError("Manual fine-tuning requires transformers. Install it in the active Python environment.") from exc
    AutoTokenizer = transformers.AutoTokenizer
    AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification
    Trainer = transformers.Trainer
    TrainingArguments = transformers.TrainingArguments
    TrainerCallback = transformers.TrainerCallback
    from src.model import select_model_path, tokenizer_folder

    if len(examples) < 4:
        raise ValueError("Add at least 4 labeled examples before training.")
    labels = [int(item["label"]) for item in examples]
    if len(set(labels)) < 2:
        raise ValueError("Training requires examples from both SAFE and VULNERABLE classes.")

    base_path = select_model_path()
    tok_path = tokenizer_folder(base_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(base_path), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class ManualCodeDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            item = self.items[idx]
            encoded = tokenizer(item["code"], truncation=True, padding="max_length", max_length=int(max_length), return_tensors="pt")
            result = {key: value.squeeze(0) for key, value in encoded.items()}
            result["labels"] = torch.tensor(int(item["label"]), dtype=torch.long)
            return result

    train_data = ManualCodeDataset(examples)
    out_path = (ROOT / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_lines = []
    total_steps = max(1, (len(train_data) + int(batch_size) - 1) // int(batch_size) * int(epochs))

    class StreamlitProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            status_box.info(f"Training started · {len(train_data)} labeled examples · {int(epochs)} epoch(s)")
            progress_bar.progress(0.01, text="Preparing training…")
        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step or 0)
            progress_bar.progress(min(0.95, step / max(1, state.max_steps)), text=f"Training step {step}/{state.max_steps}")
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                clean = {k: round(float(v), 5) if isinstance(v, (int, float)) else v for k, v in logs.items() if k in {"loss", "learning_rate", "epoch", "train_loss"}}
                if clean:
                    log_lines.append(json.dumps(clean))
                    log_box.code("\n".join(log_lines[-30:]) or "Waiting for training logs…")
        def on_train_end(self, args, state, control, **kwargs):
            progress_bar.progress(0.97, text="Training finished · saving model artifacts…")

    # Use compatible TrainingArguments names across Transformers versions.
    import inspect
    args_params = inspect.signature(TrainingArguments.__init__).parameters
    arg_values = dict(
        output_dir=str(out_path / "trainer-output"),
        learning_rate=float(learning_rate),
        per_device_train_batch_size=int(batch_size),
        num_train_epochs=int(epochs),
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=(device.type == "cpu") if "use_cpu" in args_params else False,
    )
    if "evaluation_strategy" in args_params:
        arg_values["evaluation_strategy"] = "no"
    elif "eval_strategy" in args_params:
        arg_values["eval_strategy"] = "no"
    arg_values = {k:v for k,v in arg_values.items() if k in args_params}
    training_args = TrainingArguments(**arg_values)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, callbacks=[StreamlitProgressCallback()])
    trainer.train()

    trainer.model.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer.save_pretrained(str(out_path))
    weights = [name for name in ("model.safetensors", "pytorch_model.bin") if (out_path / name).is_file()]
    if not weights:
        raise FileNotFoundError(f"Training finished but no model weights were verified in {out_path}")
    if not ((out_path / "tokenizer.json").is_file() or ((out_path / "vocab.json").is_file() and (out_path / "merges.txt").is_file())):
        raise FileNotFoundError(f"Model weights saved, but tokenizer files were not verified in {out_path}")
    progress_bar.progress(1.0, text="Training complete · model and tokenizer verified")
    return {"output_dir": str(out_path), "weights": weights, "examples": len(examples), "device": str(device), "logs": log_lines}

def normalize(raw):
    """Normalize the real predictor output without changing its prediction."""
    label = score = details = None
    safe_probability = vulnerable_probability = None

    if isinstance(raw, dict):
        label = raw.get("label", raw.get("prediction", raw.get("class", raw.get("result"))))
        score = raw.get("confidence", raw.get("score", raw.get("probability")))
        details = raw.get("explanation", raw.get("details", raw.get("message")))
        safe_probability = raw.get("safe_probability")
        vulnerable_probability = raw.get("vulnerable_probability")
    elif isinstance(raw, (tuple, list)) and raw:
        label = raw[0]
        if len(raw) > 1 and isinstance(raw[1], (int, float)):
            score = raw[1]
        if len(raw) > 2:
            details = raw[2]
    elif isinstance(raw, (str, int, float)):
        label = raw

    if isinstance(label, (int, float)):
        label = "VULNERABLE" if int(label) == 1 else "SAFE" if int(label) == 0 else str(label)
    elif isinstance(label, str):
        value = label.strip().upper()
        if value in {"NON_VULNERABLE", "NOT_VULNERABLE", "NON-VULNERABLE"}:
            label = "SAFE"
        elif value in {"1", "TRUE"} or "VULN" in value or "UNSAFE" in value or "DEFECT" in value:
            label = "VULNERABLE"
        elif value in {"0", "FALSE"} or "SAFE" in value or "SECURE" in value:
            label = "SAFE"
        else:
            label = label.strip()
    else:
        label = None

    def percent(value):
        if isinstance(value, (int, float)) and np.isfinite(value):
            value = float(value)
            return value * 100 if 0 <= value <= 1 else value
        return None

    return {
        "label": label,
        "confidence": percent(score),
        "safe_probability": percent(safe_probability),
        "vulnerable_probability": percent(vulnerable_probability),
        "details": details,
        "raw": raw,
    }

def load_saved_metrics():
    """Load genuine metrics from either the CSV evaluator or evaluate.py."""
    try:
        if not METRICS_FILE.is_file():
            return None
        with METRICS_FILE.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)

        # evaluate.py schema: show VULNERABLE-class scores as positive-class metrics.
        if "sample_count" in saved and "vulnerable_precision" in saved:
            normalized = {
                "accuracy": saved.get("accuracy"),
                "precision": saved.get("vulnerable_precision"),
                "recall": saved.get("vulnerable_recall"),
                "f1_score": saved.get("vulnerable_f1"),
                "evaluation_samples": saved.get("sample_count"),
                "confusion_matrix": saved.get("confusion_matrix"),
                "evaluated_at": saved.get("evaluated_at_utc", "time unavailable"),
                "metric_basis": "VULNERABLE-class scores from held-out test split",
                "weighted_precision": saved.get("weighted_precision"),
                "weighted_recall": saved.get("weighted_recall"),
                "weighted_f1": saved.get("weighted_f1"),
                "distribution": saved.get("distribution"),
            }
            required = {"accuracy", "precision", "recall", "f1_score",
                        "evaluation_samples", "confusion_matrix"}
            if required.issubset(normalized) and all(
                normalized[key] is not None for key in required
            ):
                return normalized

        # Compatibility with the app's own labeled-CSV evaluation schema.
        required = {"accuracy", "precision", "recall", "f1_score",
                    "evaluation_samples", "confusion_matrix"}
        if required.issubset(saved):
            saved.setdefault("metric_basis", "labeled CSV evaluation")
            return saved
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def save_evaluation_metrics(metrics):
    """Persist metrics produced by real labeled-data evaluation."""
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


st.session_state.setdefault("history",[])
st.session_state.setdefault("latest",None)
st.session_state.setdefault("evaluation_metrics", load_saved_metrics())
st.session_state.setdefault("last_inference_test", None)
st.session_state.setdefault("editor_code",'def read_file(filename):\n    with open(filename, "r", encoding="utf-8") as file:\n        return file.read()\n')
predictor,predictor_error=import_predictor()
status=artifact_status()

with st.sidebar:
    st.markdown("## 🛡️ CodeGuard **AI**")
    st.caption("CODE SECURITY, MADE CLEAR")
    page=st.radio("WORKSPACE",["Home","Scan Code","My Reports","Model Center","Model Diagnostics"], label_visibility="visible")
    st.markdown("---")
    st.caption("● PREDICTOR IMPORTED" if predictor else "● PREDICTOR UNAVAILABLE")
    st.caption("CodeBERT · Local inference")

st.markdown("""<div class="hero"><div class="kicker">● SECURITY INTELLIGENCE / CODEBERT</div>
<div class="bigtitle">Find the weakness.<br><span>Ship with confidence.</span></div>
<div class="sub">Scan source code, review model results, and manage your local security model in one place.</div>
<div style="margin-top:13px"><span class="pill">LOCAL MODEL INTEGRATION</span><span class="pill">SOURCE CODE ANALYSIS</span><span class="pill">AUDITABLE OUTPUTS</span></div></div>""",unsafe_allow_html=True)

if page=="Home":
    h=st.session_state.history
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Session analyses",len(h)); c2.metric("Vulnerable labels",sum(x.get("label")=="VULNERABLE" for x in h))
    c3.metric("Safe labels",sum(x.get("label")=="SAFE" for x in h)); c4.metric("Predictor import","Available" if predictor else "Unavailable")
    l,r=st.columns([1.1,.9],gap="large")
    with l:
        st.markdown('<div class="panel"><b>Quick scan</b><div class="caption">Check a code snippet with your locally installed CodeBERT model.</div></div>',unsafe_allow_html=True)
        st.code('def validate_input(value):\n    return value.strip()', language="python")
        st.info("Open Scan Code to check a source snippet with your local model.")
    with r:
        st.markdown('<div class="panel"><b>Model readiness</b><div class="caption">Artifact checks do not prove inference works.</div></div>',unsafe_allow_html=True)
        st.write("Weights:",", ".join(status["weights"]) if status["weights"] else "Not found")
        st.write("Predictor:", "Importable" if predictor else "Unavailable")
        if not predictor:
            with st.expander("Import diagnostic"): st.code(predictor_error or "Unknown error")
    home_metrics = st.session_state.get("evaluation_metrics")
    if home_metrics:
        st.markdown("### Latest model evaluation")
        home_metric_df = pd.DataFrame(
            {"Score (%)": [
                home_metrics["accuracy"] * 100,
                home_metrics["precision"] * 100,
                home_metrics["recall"] * 100,
                home_metrics["f1_score"] * 100,
            ]},
            index=["Accuracy", "Precision", "Recall", "F1-score"],
        )
        st.bar_chart(home_metric_df, y="Score (%)", height=220)
        st.caption(
            f"Based on {home_metrics['evaluation_samples']} labeled examples. "
            "These are evaluation metrics, not individual scan confidence."
        )

    if h:
        st.markdown("### Recent activity")
        st.dataframe(pd.DataFrame([{"Timestamp":x["timestamp"],"Language":x["language"],"Label":x.get("label") or "Unrecognized"} for x in h[:8]]),use_container_width=True,hide_index=True)

elif page=="Scan Code":
    latest = st.session_state.get("latest")
    st.markdown("""
    <div class="panel" style="margin-bottom:12px">
      <div class="kicker">LIVE WORKSPACE / SOURCE INSPECTION</div>
      <div style="font-size:27px;font-weight:800;letter-spacing:-.04em;margin:6px 0">Scan your code <span style="color:#32f5a0">with confidence</span></div>
      <div class="caption">Paste code or open a source file, then select Scan code. Results come from your local model.</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset-level metrics are only calculated from labeled ground truth.
    st.subheader("Model Evaluation Metrics")
    metrics = st.session_state.get("evaluation_metrics")
    if metrics is None:
        st.caption("Metrics require a labeled evaluation dataset. Prediction confidence is not model accuracy.")
        metric_cols = st.columns(4)
        for col, label in zip(metric_cols, ["Precision", "Recall", "F1-score", "Accuracy"]):
            with col:
                st.metric(label, "Not evaluated")
    else:
        metric_cols = st.columns(4)
        for col, label, key in zip(metric_cols, ["Precision", "Recall", "F1-score", "Accuracy"], ["precision", "recall", "f1_score", "accuracy"]):
            with col:
                st.metric(label, f"{metrics[key] * 100:.2f}%")
        st.caption(f"Evaluated on {metrics['evaluation_samples']} labeled examples · {metrics.get('evaluated_at', 'time unavailable')}. These are dataset-level results, not the confidence of an individual scan.")
        # Visual metric chart: values come only from the labeled evaluation run.
        st.markdown("#### Evaluation metrics chart")
        metric_chart = pd.DataFrame(
            {"Score (%)": [
                metrics["accuracy"] * 100,
                metrics["precision"] * 100,
                metrics["recall"] * 100,
                metrics["f1_score"] * 100,
            ]},
            index=["Accuracy", "Precision", "Recall", "F1-score"],
        )
        st.bar_chart(metric_chart, y="Score (%)", height=260)

        with st.expander("View confusion matrix", expanded=True):
            cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
            st.markdown("Rows = actual labels · Columns = predicted labels")
            cm_df = pd.DataFrame(
                cm,
                index=["Actual SAFE", "Actual VULNERABLE"],
                columns=["Predicted SAFE", "Predicted VULNERABLE"],
            )
            st.dataframe(cm_df, use_container_width=True)
            st.markdown("#### Confusion matrix chart")
            st.bar_chart(cm_df.T, height=260)

    with st.expander("Evaluate against a labeled CSV", expanded=False):
        st.caption("Upload a CSV with columns `code` and `label`. Labels must be SAFE/VULNERABLE or 0/1. Use a held-out dataset that was not used for training.")
        evaluation_file = st.file_uploader("Labeled evaluation dataset (.csv)", type=["csv"], key="evaluation_dataset_upload")
        if evaluation_file is not None:
            try:
                evaluation_df = pd.read_csv(evaluation_file)
                if not {"code", "label"}.issubset(evaluation_df.columns):
                    st.error("CSV needs both 'code' and 'label' columns.")
                else:
                    st.write(f"{len(evaluation_df)} rows loaded")
                    st.dataframe(evaluation_df[["code", "label"]].head(5), use_container_width=True, hide_index=True)
                    if st.button("Run model evaluation", type="primary", key="run_model_evaluation", disabled=not bool(predictor)):
                        try:
                            with st.spinner("Running real CodeBERT predictions on labeled examples…"):
                                evaluated = evaluate_labeled_dataset(evaluation_df)
                            st.session_state.evaluation_metrics = evaluated
                            save_evaluation_metrics(evaluated)
                            st.success(f"Evaluation completed on {evaluated['evaluation_samples']} labeled examples. Metrics saved locally.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Evaluation failed. Metrics were not updated: {type(exc).__name__}: {exc}")
            except Exception as exc:
                st.error(f"Could not read CSV: {type(exc).__name__}: {exc}")
        if not predictor:
            st.warning("The predictor is unavailable. Fix the model import before evaluating.")

    # Compact toolbar directly above the full-width editor.
    st.markdown('<div class="workspace-toolbar">', unsafe_allow_html=True)
    language_col, upload_col = st.columns([0.85, 1.5], gap="medium")
    with language_col:
        language = st.selectbox("SOURCE LANGUAGE", ["Python", "Java", "JavaScript", "C/C++", "PHP", "C#", "Go", "Other"], key="workspace_language")
    with upload_col:
        upload = st.file_uploader("IMPORT SOURCE FILE", type=["py", "java", "js", "ts", "c", "cpp", "h", "hpp", "php", "cs", "go", "rb"], key="workspace_source_upload")
    st.markdown('</div>', unsafe_allow_html=True)

    if upload is not None:
        try:
            uploaded_code = upload.getvalue().decode("utf-8")
            if st.session_state.get("_last_uploaded_source") != (upload.name, uploaded_code):
                st.session_state.editor_code = uploaded_code
                st.session_state._last_uploaded_source = (upload.name, uploaded_code)
        except UnicodeDecodeError:
            st.error("This source file could not be decoded as UTF-8 text.")

    st.markdown('<div class="workspace-editor-label"><span class="kicker">SOURCE BUFFER</span><span class="hint">LOCAL INPUT · UTF-8 TEXT</span></div>', unsafe_allow_html=True)
    code = st.text_area("Source code editor", key="editor_code", height=560, label_visibility="collapsed",
                        placeholder="# Paste the function or source file you want to inspect…")
    line_count = len(code.splitlines()) if code else 0
    char_count = len(code) if code else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Lines", line_count)
    m2.metric("Characters", char_count)
    m3.metric("Language", language)

    # Prominent action sits immediately after the editor; clear is secondary.
    run_col, clear_col, spacer_col = st.columns([1.8, 0.8, 3.4], gap="small")
    with run_col:
        st.markdown('<div class="workspace-run">', unsafe_allow_html=True)
        run = st.button("▶  SCAN MY CODE", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with clear_col:
        if st.button("Clear buffer", use_container_width=True):
            st.session_state.editor_code = ""
            st.session_state.pop("_last_uploaded_source", None)
            st.session_state.latest = None
            st.rerun()

    if run:
        if not code.strip():
            st.warning("Enter source code before running analysis.")
        elif not predictor:
            st.error("Predictor unavailable. No prediction was made.")
            st.code(predictor_error or "Import failed")
        else:
            with st.spinner("Running your existing vulnerability classifier…"):
                raw, error = invoke(predictor, code)
            if error:
                st.error("Inference failed. No result was recorded.")
                st.code(error)
            else:
                out = normalize(raw)
                record = {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "language": language,
                    "label": out["label"],
                    "confidence": out["confidence"],
                    "safe_probability": out["safe_probability"],
                    "vulnerable_probability": out["vulnerable_probability"],
                    "details": str(out["details"]) if out["details"] is not None else None,
                    "code": code,
                    "raw": repr(raw),
                }
                st.session_state.history.insert(0, record)
                st.session_state.latest = record
                latest = record

    # Results render below the editor and action, never inside a cramped sidebar.
    if latest:
        st.markdown('<div class="result-panel"><div class="kicker">ANALYSIS RESULT / REAL MODEL OUTPUT</div><h3 style="margin:7px 0 12px">Latest analysis</h3></div>', unsafe_allow_html=True)
        label = latest.get("label")
        if label == "VULNERABLE":
            st.markdown('<div class="bad"><b>⚠ VULNERABLE — model classification</b><br><span class="caption">Validate this finding independently; this is not a complete security audit.</span></div>', unsafe_allow_html=True)
        elif label == "SAFE":
            st.markdown('<div class="ok"><b>✓ SAFE — model classification</b><br><span class="caption">This label does not guarantee that the code is vulnerability-free.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="neutral"><b>Result returned — label not recognized</b><br><span class="caption">Inspect the raw predictor output before interpreting it.</span></div>', unsafe_allow_html=True)
        result_cols = st.columns([1, 1, 1.5])
        result_cols[0].metric("Classification", label or "Unrecognized")
        if latest.get("confidence") is not None:
            result_cols[1].metric("Model-reported confidence", f"{latest['confidence']:.2f}%")
        else:
            result_cols[1].markdown('<div class="panel"><div class="caption">MODEL-REPORTED CONFIDENCE</div><div style="font-size:15px;font-weight:800;margin-top:8px;color:#91a7bb">Not provided</div></div>', unsafe_allow_html=True)
        result_cols[2].metric("Lines analyzed", len(latest.get("code", "").splitlines()))
        probability_cols = st.columns(2)
        safe_score = latest.get("safe_probability")
        vulnerable_score = latest.get("vulnerable_probability")
        if safe_score is not None:
            probability_cols[0].metric("SAFE probability", f"{safe_score:.2f}%")
        if vulnerable_score is not None:
            probability_cols[1].metric("VULNERABLE probability", f"{vulnerable_score:.2f}%")
        if latest.get("details"):
            st.markdown("**Predictor-provided details**")
            st.write(latest["details"])
        with st.expander("Inspect raw predictor output"):
            st.code(latest.get("raw", "Not available"))
        st.download_button("Export analysis JSON", json.dumps(latest, indent=2), "codeguard-analysis.json", "application/json")
    else:
        st.markdown('<div class="neutral"><b>Awaiting analysis</b><br><span class="caption">Your result will appear here after a successful real inference call. No prediction is generated automatically.</span></div>', unsafe_allow_html=True)

    # Responsive, compact pipeline guidance follows the primary workspace.
    with st.expander("How scanning works", expanded=False):
        p1, p2, p3 = st.columns(3)
        p1.markdown("**01 · Input**\n\nPaste or import source code.")
        p2.markdown("**02 · Inference**\n\nCalls `src.model.predict_vulnerability`; failures are shown explicitly.")
        p3.markdown("**03 · Review**\n\nShows returned label, confidence only if supplied, and raw output.")
        st.caption("Predictor import: " + ("available" if predictor else "unavailable"))
        st.caption("Artifact presence alone does not prove inference works.")

elif page=="My Reports":
    st.markdown("## My reports")
    report_tab, history_tab = st.tabs(["Latest report", "Scan history"])
    with report_tab:
        r = st.session_state.latest
        if not r:
            st.info("No report yet. Scan a code snippet to create your first report.")
        else:
            label = r.get("label")
            if label == "VULNERABLE":
                st.markdown('<div class="bad"><b>⚠ Potential vulnerability detected</b><br><span class="caption">This is a model classification, not a complete security audit. Review the code before acting.</span></div>', unsafe_allow_html=True)
            elif label == "SAFE":
                st.markdown('<div class="ok"><b>✓ Model classified this code as safe</b><br><span class="caption">A SAFE result does not guarantee that code is vulnerability-free.</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="neutral"><b>Result could not be interpreted</b></div>', unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.metric("Scanned at", r.get("timestamp", "—"))
            b.metric("Language", r.get("language", "—"))
            c.metric("Model confidence", f"{r['confidence']:.2f}%" if r.get("confidence") is not None else "Not provided")
            st.markdown("### Submitted code")
            st.code(r.get("code", ""), language="python")
            if r.get("details"):
                st.markdown("### Model details")
                st.write(r["details"])
            with st.expander("Technical details"):
                st.code(r.get("raw", "Not available"))
            st.download_button("Download report (JSON)", json.dumps(r, indent=2), "codeguard-report.json", "application/json")
    with history_tab:
        hist = st.session_state.history
        if not hist:
            st.info("Your completed scans will appear here.")
        else:
            filter_choice = st.radio("Show", ["All", "Potential vulnerability", "Safe"], horizontal=True, key="reports_filter")
            filtered = [x for x in hist if filter_choice == "All" or (filter_choice == "Potential vulnerability" and x.get("label") == "VULNERABLE") or (filter_choice == "Safe" and x.get("label") == "SAFE")]
            df = pd.DataFrame([{"Timestamp":x.get("timestamp"), "Language":x.get("language"), "Result":x.get("label") or "Unrecognized", "Confidence (%)":x.get("confidence"), "Lines":len(x.get("code", "").splitlines())} for x in filtered])
            st.dataframe(df, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            c1.download_button("Export history CSV", df.to_csv(index=False).encode("utf-8"), "codeguard-history.csv", "text/csv", use_container_width=True)
            c2.download_button("Export history JSON", json.dumps(filtered, indent=2, ensure_ascii=False).encode("utf-8"), "codeguard-history.json", "application/json", use_container_width=True)
            with st.expander("Open a scan record"):
                chosen = st.selectbox("Scan", range(len(filtered)), format_func=lambda i: f"{filtered[i].get('timestamp')} · {filtered[i].get('label') or 'Unrecognized'}")
                st.code(filtered[chosen].get("code", ""), language="python")
                if st.button("Clear session history"):
                    st.session_state.history = []
                    st.session_state.latest = None
                    st.rerun()

elif page=="Model Center":
    st.markdown("## Model Center")
    st.markdown('<div class="panel"><div class="kicker">MODEL STATUS</div><b>Local CodeBERT classifier</b><div class="caption">Run a real test here. Fine-tuning is optional and stays in advanced settings.</div></div>', unsafe_allow_html=True)
    model_status = artifact_status()
    a, b, c = st.columns(3)
    a.metric("Model weights", "Found" if model_status["weights"] else "Not found")
    b.metric("Configuration", "Found" if model_status["config"] else "Not found")
    c.metric("Tokenizer", "Found" if model_status["tokenizer"] else "Not detected")
    st.caption("File presence is not proof of successful loading or prediction accuracy.")

    saved_metrics = st.session_state.get("evaluation_metrics")
    st.markdown("### Evaluation performance")
    if saved_metrics:
        performance_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Score (%)": [
                saved_metrics["accuracy"] * 100,
                saved_metrics["precision"] * 100,
                saved_metrics["recall"] * 100,
                saved_metrics["f1_score"] * 100,
            ],
        })
        st.bar_chart(performance_df, x="Metric", y="Score (%)", height=250)
        st.caption(f"Real evaluation · {saved_metrics['evaluation_samples']} labeled examples.")
    else:
        st.info("No saved evaluation metrics yet. Run evaluation from Scan Code using a labeled CSV.")

    if not predictor:
        st.error("The model predictor could not be imported; no prediction can be made.")
        with st.expander("Developer details", expanded=False):
            st.code(predictor_error or "Unknown import error")

    if st.button("Test model", type="primary", disabled=not bool(predictor), key="model_center_test"):
        test_code = "def normalize_name(value):\n    return value.strip()"
        with st.spinner("Running a real local inference test…"):
            raw, test_error = invoke(predictor, test_code)
        if test_error:
            st.session_state.last_inference_test = {"status": "failed", "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "error": test_error}
        else:
            norm = normalize(raw)
            st.session_state.last_inference_test = {"status": "success", "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "label": norm["label"], "confidence": norm["confidence"], "raw": repr(raw)}

    test_result = st.session_state.get("last_inference_test")
    if test_result:
        if test_result.get("status") == "success":
            st.success("The inference call completed. This confirms execution, not prediction accuracy.")
            cols = st.columns(2)
            cols[0].metric("Returned classification", test_result.get("label") or "Unrecognized")
            cols[1].metric("Model-reported confidence", f"{test_result['confidence']:.2f}%" if test_result.get("confidence") is not None else "Not provided")
            st.caption("A confidence score is not a guarantee that a classification is correct.")
        else:
            st.error("The inference test failed. No result is being presented as successful.")
            with st.expander("Developer error details", expanded=False):
                st.code(test_result.get("error", "Unknown inference error"))

    with st.expander("Advanced settings · Fine-tuning", expanded=False):
        st.markdown("### Train a local model with labeled examples")
        st.write("Add snippets you have reviewed and label each one. Training starts only when you explicitly select **Start fine-tuning**.")
        st.warning("Four examples only pass a basic input check. They are far too few to train a reliable vulnerability detector. Use a large, diverse, correctly labeled dataset and evaluate on separate held-out examples before relying on a model.")
        if "manual_examples" not in st.session_state:
            st.session_state.manual_examples = [
                {"code": "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b", "label": 0},
                {"code": "def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()", "label": 1},
            ]

        examples = st.session_state.manual_examples
        safe_count = sum(item["label"] == 0 for item in examples)
        vuln_count = sum(item["label"] == 1 for item in examples)
        st.markdown(f"**{len(examples)} of 4 minimum examples added** · SAFE: {safe_count} · VULNERABLE: {vuln_count}")
        st.progress(min(1.0, len(examples) / 4), text=f"Basic input check: {min(len(examples), 4)}/4 examples")
        if len(examples) < 4:
            st.info(f"Add {4 - len(examples)} more labeled example{'s' if 4 - len(examples) != 1 else ''} to pass the minimum input check.")
        elif len({item["label"] for item in examples}) < 2:
            st.warning("Add examples from both SAFE and VULNERABLE classes before training.")
        else:
            st.success("Basic input checks passed. This does not indicate that the dataset is sufficient for reliable training.")

        with st.form("manual_example_form", clear_on_submit=True):
            example_code = st.text_area("Code example", height=150, placeholder="Paste a short function or source snippet…")
            example_label = st.selectbox("Correct label", ["SAFE", "VULNERABLE"], help="Choose the label based on a careful review of this code.")
            add_example = st.form_submit_button("＋ Add example", use_container_width=True)
        if add_example:
            if not example_code.strip():
                st.warning("Enter source code before adding an example.")
            elif any(item["code"].strip() == example_code.strip() and item["label"] == (0 if example_label == "SAFE" else 1) for item in examples):
                st.warning("This exact code and label are already in the examples list.")
            else:
                st.session_state.manual_examples.append({"code": example_code.strip(), "label": 0 if example_label == "SAFE" else 1})
                st.success(f"Added one {example_label} example.")
                st.rerun()

        if examples:
            with st.expander(f"Review or remove examples ({len(examples)})", expanded=False):
                for i, item in enumerate(list(examples)):
                    left, middle, right = st.columns([0.7, 3.3, 0.8])
                    left.write(f"**{i + 1}.**")
                    middle.markdown(f"**{'SAFE' if item['label'] == 0 else 'VULNERABLE'}** · {item['code'].splitlines()[0][:90] if item['code'].splitlines() else 'Empty code'}")
                    with right:
                        if st.button("Remove", key=f"remove_example_{i}"):
                            st.session_state.manual_examples.pop(i)
                            st.rerun()

        with st.expander("Advanced training settings", expanded=False):
            cfg1, cfg2, cfg3, cfg4 = st.columns(4)
            with cfg1:
                train_epochs = st.number_input("Epochs", min_value=1, max_value=10, value=2, step=1, key="manual_train_epochs")
            with cfg2:
                train_batch = st.selectbox("Batch size", [1, 2, 4, 8], index=1, key="manual_train_batch")
            with cfg3:
                train_lr = st.selectbox("Learning rate", [1e-5, 2e-5, 3e-5, 5e-5], index=1, format_func=lambda x: f"{x:.0e}", key="manual_train_lr")
            with cfg4:
                train_maxlen = st.selectbox("Max token length", [128, 256, 384, 512], index=0, key="manual_train_maxlen")
            output_dir = st.text_input("Save model to", "models/codebert-manual-finetuned", key="manual_train_output")
            st.caption("The fine-tuned model is saved separately. Your existing model directory is not overwritten.")

        # Defaults also exist when advanced settings remain collapsed.
        train_epochs = int(st.session_state.get("manual_train_epochs", 2))
        train_batch = int(st.session_state.get("manual_train_batch", 2))
        train_lr = float(st.session_state.get("manual_train_lr", 2e-5))
        train_maxlen = int(st.session_state.get("manual_train_maxlen", 128))
        output_dir = st.session_state.get("manual_train_output", "models/codebert-manual-finetuned")

        if st.button("▶ Start real fine-tuning", type="primary", use_container_width=True, key="start_manual_finetuning"):
            if len(examples) < 4:
                st.error(f"Add at least 4 labeled examples first. You currently have {len(examples)} of 4.")
            elif len({item["label"] for item in examples}) < 2:
                st.error("Include both SAFE and VULNERABLE examples before training.")
            elif not output_dir.strip():
                st.error("Enter a model output folder.")
            else:
                progress = st.progress(0, text="Initializing fine-tuning…")
                status_box = st.empty()
                log_box = st.empty()
                try:
                    result = run_manual_finetuning(examples, output_dir.strip(), train_epochs, train_batch, train_lr, train_maxlen, progress, status_box, log_box)
                    # Verify saved artifacts once more before announcing success.
                    saved = Path(result["output_dir"])
                    weights_ok = any((saved / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
                    config_ok = (saved / "config.json").is_file()
                    tokenizer_ok = (saved / "tokenizer.json").is_file() or ((saved / "vocab.json").is_file() and (saved / "merges.txt").is_file())
                    if not (weights_ok and config_ok and tokenizer_ok):
                        raise RuntimeError("Post-save verification failed: expected model weights, config.json, and tokenizer files were not all found.")
                    st.session_state.last_finetune = {"status": "success", **result, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}
                    st.success("Fine-tuning completed and saved model artifacts were verified.")
                    st.write("**Saved model folder:**", result["output_dir"])
                    st.write("**Verified weights:**", ", ".join(result["weights"]))
                    st.write("**Training device:**", result["device"])
                    st.warning("The saved model is separate from the active model. To use it for scans, configure CODEGUARD_MODEL_PATH to this folder or update your model selection, then restart/clear the cached model. A successful training run does not establish model accuracy.")
                    get_model_runtime.clear()
                except Exception as exc:
                    st.session_state.last_finetune = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}
                    status_box.error("Fine-tuning failed. The app will not report success.")
                    st.exception(exc)

        last_train = st.session_state.get("last_finetune")
        if last_train:
            st.markdown("### Latest fine-tuning run")
            st.caption("Run time: " + last_train.get("timestamp", "unknown"))
            if last_train.get("status") == "success":
                st.success(f"Verified saved model: {last_train.get('output_dir')}")
                st.write("Examples used:", last_train.get("examples"), "· Device:", last_train.get("device"))
            else:
                st.error("Last training run failed.")
                st.code(last_train.get("error", "No error details recorded."))

elif page=="Model Diagnostics":
    st.markdown("## Model diagnostics")
    status = artifact_status(); predictor, error = import_predictor()
    st.caption("Artifact checks and predictor import are not the same as a successful inference call.")
    a, b, c = st.columns(3)
    a.metric("Weights", "Found" if status["weights"] else "Missing")
    b.metric("Configuration", "Found" if status["config"] else "Missing")
    c.metric("Tokenizer", "Found" if status["tokenizer"] else "Not detected")
    st.caption("These checks inspect local files. Use the inference test below to verify execution.")
    with st.expander("Advanced diagnostics", expanded=False):
        st.write("**Selected artifact folder:**", status["path"] or "Not found")
        st.write("**Weight files:**", ", ".join(status["weights"]) if status["weights"] else "No supported weight file found")
        st.write("**Predictor import:**", "Available" if predictor else "Unavailable")
        if not predictor:
            st.code(error or "Import failed")
        st.write("**Folders checked:**")
        st.code("\n".join(status.get("checked", [])))
    st.markdown("### Real inference test")
    st.write("This submits a small fixed code snippet to your loaded local model. It is a real inference call and is not added to Analysis History.")
    st.code("def normalize_name(value):\n    return value.strip()", language="python")
    if st.button("▶ TEST INFERENCE", type="primary", disabled=not bool(predictor)):
        with st.spinner("Calling src.model.predict_vulnerability…"):
            raw, test_error = invoke(predictor, "def normalize_name(value):\n    return value.strip()")
        if test_error:
            st.session_state.last_inference_test = {"status":"failed", "timestamp":datetime.now().astimezone().isoformat(timespec="seconds"), "error":test_error}
        else:
            normalized = normalize(raw)
            st.session_state.last_inference_test = {"status":"success", "timestamp":datetime.now().astimezone().isoformat(timespec="seconds"), "label":normalized["label"], "confidence":normalized["confidence"], "raw":repr(raw)}
    test_result = st.session_state.get("last_inference_test")
    if test_result:
        st.caption("Last test: " + test_result.get("timestamp", "unknown time"))
        if test_result["status"] == "success":
            st.success("Inference call completed successfully. This verifies execution, not model accuracy.")
            st.write("Returned classification:", test_result.get("label") or "Unrecognized output")
            if test_result.get("confidence") is not None: st.write(f"Model-reported confidence: {test_result['confidence']:.2f}%")
            else: st.write("Model-reported confidence: not provided")
            with st.expander("Raw predictor output"): st.code(test_result.get("raw", "Not available"))
        else:
            st.error("Inference failed. The model is not marked as tested successfully.")
            st.code(test_result.get("error", "Unknown inference error"))
    st.caption("Inference status: " + ("successfully tested" if test_result and test_result.get("status") == "success" else "failed" if test_result and test_result.get("status") == "failed" else "not tested"))
    if st.button("Refresh diagnostics"): st.rerun()

st.markdown("---")
st.markdown('<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#71889d;font:11px DM Mono,monospace"><span>CODEGUARD AI / CODEBERT WORKSPACE</span><span>Model classifications are advisory; validate findings independently.</span></div>',unsafe_allow_html=True)
