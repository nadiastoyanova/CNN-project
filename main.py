from pathlib import Path
from typing import Any
import base64
import json
import os

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "models"))
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

_model_cache: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Animal-10 class labels (Italian folder names → English, alphabetical order
# matching how Keras image_dataset_from_directory sorts them)
# ---------------------------------------------------------------------------
ANIMALS10_LABELS = [
    "dog",        # cane
    "horse",      # cavallo
    "elephant",   # elefante
    "butterfly",  # farfalla
    "chicken",    # gallina
    "cat",        # gatto
    "cow",        # mucca
    "sheep",      # pecora
    "spider",     # ragno
    "squirrel",   # scoiattolo
]


def _slugify(value: str) -> str:
    safe_chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
    lowered = value.lower().replace(" ", "-").replace("_", "-")
    slug = "".join(ch for ch in lowered if ch in safe_chars)
    return slug or "model"


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models() -> list[dict[str, str]]:
    models: list[dict[str, str]] = []
    if not MODELS_ROOT.exists():
        return models

    # ── Hugging Face-style folders (config.json + model.safetensors) ────────
    for config_path in sorted(MODELS_ROOT.glob("**/config.json")):
        model_dir = config_path.parent
        weights_path = model_dir / "model.safetensors"
        if not weights_path.exists():
            continue
        rel = model_dir.relative_to(MODELS_ROOT)
        model_id = _slugify(f"hf-{rel.as_posix()}")
        models.append(
            {
                "id": model_id,
                "name": f"{model_dir.name} (ViT)",
                "kind": "transformers",
                "path": str(model_dir),
                "preprocess": "processor",  # handled by HF processor
            }
        )

    # ── Keras .h5 files ─────────────────────────────────────────────────────
    for h5_path in sorted(MODELS_ROOT.glob("**/*.h5")):
        resolved_h5 = h5_path
        if h5_path.is_dir():
            nested = h5_path / h5_path.name
            if nested.exists() and nested.is_file() and nested.suffix == ".h5":
                resolved_h5 = nested
            else:
                continue
        if not resolved_h5.is_file():
            continue

        rel = resolved_h5.relative_to(MODELS_ROOT)
        model_id = _slugify(f"keras-{rel.as_posix()}")
        # Detect which preprocessing to apply based on filename
        preprocess = _detect_preprocess(resolved_h5.stem)
        models.append(
            {
                "id": model_id,
                "name": f"{resolved_h5.stem} (Keras)",
                "kind": "keras",
                "path": str(resolved_h5),
                "preprocess": preprocess,
            }
        )

    # ── Keras .keras files ───────────────────────────────────────────────────
    # FIX: discover_models() was ignoring .keras format entirely.
    # CNN Vanilla and ConvNeXt are saved as .keras, so they were invisible.
    for keras_path in sorted(MODELS_ROOT.glob("**/*.keras")):
        if not keras_path.is_file():
            continue
        rel = keras_path.relative_to(MODELS_ROOT)
        model_id = _slugify(f"keras-{rel.as_posix()}")
        # Use parent folder name when the file has a generic name like "model"
        # e.g.  convnext-model/model.keras  →  "convnext-model (Keras)"
        #        vanilla_cnn.keras           →  "vanilla_cnn (Keras)"
        generic_stems = {"model", "saved_model", "checkpoint", "weights"}
        if keras_path.stem.lower() in generic_stems:
            display_stem = keras_path.parent.name   # e.g. "convnext-model"
        else:
            display_stem = keras_path.stem           # e.g. "vanilla_cnn"
        preprocess = _detect_preprocess(display_stem)
        models.append(
            {
                "id": model_id,
                "name": f"{display_stem} (Keras)",
                "kind": "keras",
                "path": str(keras_path),
                "preprocess": preprocess,
            }
        )

    # Deduplicate in case of path collisions
    dedup: dict[str, dict[str, str]] = {}
    for item in models:
        dedup[item["id"]] = item
    return list(dedup.values())


def _detect_preprocess(stem: str) -> str:
    """
    FIX: _predict_keras() was applying /255 normalization to ALL Keras models
    regardless of how they were trained. EfficientNetV2 and ConvNeXt require
    their own preprocess_input() functions, not a simple /255 rescaling.

    Mapping:
      - CNN Vanilla      → 'none'         (model has Rescaling(1/255) as first layer, pass raw pixels)
      - EfficientNetV2   → 'efficientnet' (tf preprocess_input)
      - ConvNeXt         → 'convnext'     (tf preprocess_input)
    """
    stem_lower = stem.lower()
    if "vanilla" in stem_lower or "cnn" in stem_lower:
        return "none"
    if "efficient" in stem_lower:
        return "efficientnet"
    if "convnext" in stem_lower or "convnext" in stem_lower:
        return "convnext"
    # Default: plain rescaling (CNN Vanilla)
    return "rescale"


def get_selected_model(model_id: str | None = None) -> tuple[dict[str, str], list[dict[str, str]]]:
    models = discover_models()
    if not models:
        raise FileNotFoundError(
            f"No models found under '{MODELS_ROOT}'. "
            "Add a ViT folder, a .h5 file, or a .keras file."
        )

    selected_id = model_id or DEFAULT_MODEL_ID
    if selected_id:
        for item in models:
            if item["id"] == selected_id:
                return item, models

    return models[0], models


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_class_names_for_transformer(model_dir: Path) -> list[str]:
    model_config_path = model_dir / "config.json"
    if not model_config_path.exists():
        return []
    with model_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    id2label = cfg.get("id2label") or {}
    if not id2label:
        return []
    sorted_items = sorted(id2label.items(), key=lambda x: int(x[0]))
    return [label for _, label in sorted_items]


def _load_keras_class_names(model_path: Path, output_size: int) -> list[str]:
    """
    FIX 1: ConvNeXt saves a label_map.json with format {"index_to_label": {"0": "dog", ...}}.
            The app was expecting a flat list in a .labels.json file → format mismatch,
            so ConvNeXt always fell back to Class_0..Class_9.
            Now we also read label_map.json with the ConvNeXt structure.

    FIX 2: CNN Vanilla and EfficientNetV2 save no label file at all.
            We fall back to the hardcoded Animals-10 label list so the app
            always shows real animal names instead of Class_0..Class_9.
    """
    # Priority 1: flat list  →  <model>.labels.json
    labels_file = model_path.with_suffix(".labels.json")
    if labels_file.exists():
        with labels_file.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]

    # Priority 2: ConvNeXt label_map.json in the same directory
    label_map_file = model_path.parent / "label_map.json"
    if label_map_file.exists():
        with label_map_file.open("r", encoding="utf-8") as f:
            label_map = json.load(f)
        index_to_label = label_map.get("index_to_label") or {}
        if index_to_label:
            sorted_items = sorted(index_to_label.items(), key=lambda x: int(x[0]))
            return [label for _, label in sorted_items]

    # Priority 3: hardcoded Animals-10 fallback (covers CNN Vanilla & EfficientNetV2)
    if output_size == len(ANIMALS10_LABELS):
        return ANIMALS10_LABELS

    # Last resort: generic class names
    return [f"Class_{idx}" for idx in range(output_size)]


# ---------------------------------------------------------------------------
# Model loading (with cache)
# ---------------------------------------------------------------------------

def get_transformer_bundle(model_meta: dict[str, str]) -> dict[str, Any]:
    cache_key = model_meta["id"]
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model_dir = Path(model_meta["path"])
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at '{model_dir}'.")

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers'. Install with: pip install transformers torch"
        ) from exc

    model = AutoModelForImageClassification.from_pretrained(str(model_dir))
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model.eval()

    bundle = {
        "kind": "transformers",
        "model": model,
        "processor": processor,
        "class_names": load_class_names_for_transformer(model_dir),
    }
    _model_cache[cache_key] = bundle
    return bundle


def get_keras_bundle(model_meta: dict[str, str]) -> dict[str, Any]:
    cache_key = model_meta["id"]
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model_path = Path(model_meta["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Keras model file not found at '{model_path}'.")

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'tensorflow'. Install with: pip install tensorflow"
        ) from exc

    model = load_model(model_path)
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    output_size = int(output_shape[-1]) if output_shape and output_shape[-1] else 1

    bundle = {
        "kind": "keras",
        "model": model,
        "preprocess": model_meta.get("preprocess", "rescale"),
        "class_names": _load_keras_class_names(model_path, max(output_size, 1)),
    }
    _model_cache[cache_key] = bundle
    return bundle


def get_model_bundle(model_meta: dict[str, str]) -> dict[str, Any]:
    if model_meta["kind"] == "transformers":
        return get_transformer_bundle(model_meta)
    if model_meta["kind"] == "keras":
        return get_keras_bundle(model_meta)
    raise ValueError(f"Unsupported model kind: {model_meta['kind']}")


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _apply_keras_preprocess(arr: np.ndarray, preprocess_type: str) -> np.ndarray:
    """
    FIX: Previously _predict_keras() always divided by 255 for every Keras model.
    EfficientNetV2 and ConvNeXt require their framework-specific preprocess_input(),
    which applies ImageNet mean/std normalization (not a simple /255 rescale).
    Applying /255 to these models causes completely wrong predictions.
    """
    if preprocess_type == "none":
        # CNN Vanilla: model has Rescaling(1./255) as its first layer.
        # Pass raw uint8-range float32 values (0–255) — the model rescales internally.
        return arr  # no-op

    if preprocess_type == "efficientnet":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        return preprocess_input(arr)

    if preprocess_type == "convnext":
        from tensorflow.keras.applications.convnext import preprocess_input
        return preprocess_input(arr)

    # "rescale" fallback (legacy) → divide by 255
    return arr / 255.0


def preprocess_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    return img


def build_image_preview(file_storage) -> str | None:
    file_storage.stream.seek(0)
    raw_bytes = file_storage.read()
    file_storage.stream.seek(0)
    if not raw_bytes:
        return None
    mime_type = file_storage.mimetype or "image/jpeg"
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _ensure_probability_vector(raw_output: np.ndarray) -> np.ndarray:
    arr = np.array(raw_output, dtype=np.float32).flatten()
    if arr.size == 0:
        raise RuntimeError("Model returned empty prediction output.")
    if arr.size == 1:
        p1 = float(np.clip(arr[0], 0.0, 1.0))
        return np.array([1.0 - p1, p1], dtype=np.float32)
    if np.min(arr) < 0 or np.max(arr) > 1.0:
        exps = np.exp(arr - np.max(arr))
        arr = exps / np.sum(exps)
    return arr


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _predict_transformer(bundle: dict[str, Any], image: Image.Image) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'torch'. Install with: pip install torch") from exc

    model = bundle["model"]
    processor = bundle["processor"]
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()


def _predict_keras(bundle: dict[str, Any], image: Image.Image) -> np.ndarray:
    model = bundle["model"]
    preprocess_type = bundle.get("preprocess", "rescale")

    input_shape = getattr(model, "input_shape", None)
    target_h, target_w = 224, 224
    if input_shape and len(input_shape) >= 3:
        target_h = int(input_shape[1] or 224)
        target_w = int(input_shape[2] or 224)

    image_resized = image.resize((target_w, target_h))
    # Keep pixel values in [0, 255] range as float32 so preprocess_input
    # functions (efficientnet, convnext) receive the expected input range.
    arr = np.array(image_resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # FIX: apply the correct preprocessing per model instead of always /255
    arr = _apply_keras_preprocess(arr, preprocess_type)

    raw = model.predict(arr, verbose=0)[0]
    return _ensure_probability_vector(raw)


def run_prediction(file_storage, model_meta: dict[str, str]):
    bundle = get_model_bundle(model_meta)
    class_names: list[str] = bundle.get("class_names") or []
    image = preprocess_image(file_storage)

    if bundle["kind"] == "transformers":
        probabilities = _predict_transformer(bundle, image)
    else:
        probabilities = _predict_keras(bundle, image)

    top_idx = int(np.argmax(probabilities))
    top_confidence = float(probabilities[top_idx])
    top_label = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"

    top3_idx = np.argsort(probabilities)[-3:][::-1]
    top3 = []
    for idx in top3_idx:
        class_name = class_names[int(idx)] if int(idx) < len(class_names) else f"Class {int(idx)}"
        top3.append({"label": class_name, "confidence": round(float(probabilities[int(idx)]) * 100, 2)})

    return {
        "label": top_label,
        "confidence": round(top_confidence * 100, 2),
        "top3": top3,
    }


def get_feedback_mood(confidence: float) -> str:
    if confidence >= 80:
        return "win"
    if confidence >= 50:
        return "thinking"
    return "oops"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    selected = request.args.get("model_id")
    selected_model, models = get_selected_model(selected)
    return render_template("index.html", models=models, selected_model_id=selected_model["id"])


@app.route("/predict", methods=["POST"])
def predict():
    result = None
    error = None
    uploaded_name = None
    image_preview = None
    selected_model = None
    models: list[dict[str, str]] = []

    file = request.files.get("image")
    selected_model_id = request.form.get("model_id", "")

    try:
        selected_model, models = get_selected_model(selected_model_id)
    except Exception as exc:
        error = str(exc)

    if error:
        pass
    elif not file or file.filename == "":
        error = "Please select an image file."
    elif not allowed_file(file.filename):
        error = "Unsupported file type. Use png, jpg, jpeg, or webp."
    else:
        try:
            uploaded_name = secure_filename(file.filename)
            image_preview = build_image_preview(file)
            result = run_prediction(file, selected_model)
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    mood = "oops"
    if result:
        mood = get_feedback_mood(result["confidence"])

    return render_template(
        "result.html",
        result=result,
        error=error,
        mood=mood,
        filename=uploaded_name,
        model_name=selected_model["name"] if selected_model else "Unknown",
        model_id=selected_model["id"] if selected_model else "",
        image_preview=image_preview,
        models=models,
    )


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
