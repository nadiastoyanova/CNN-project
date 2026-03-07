# cnn-with-transfer-learning

Simple Flask app for image upload and prediction using multiple local models.

## Run

1. Install dependencies:
   `pip install -r requirements.txt`
2. Keep model files inside `models/`.
   Supported formats:
   - Hugging Face ViT folder (`config.json` + `model.safetensors`)
   - Keras `.h5` model files
3. Start the app:
   `python main.py`
4. Open:
   `http://127.0.0.1:5000`

## Notes

- The index page includes a model dropdown to choose the model before prediction.
- ViT labels are read automatically from `id2label` in each `config.json`.
- Keras `.h5` models are supported (optional labels file: `<model>.labels.json`).
- Results are shown on a dedicated animated page (`templates/result.html`).

## Run With Docker Compose

1. Build and start:
   `docker compose up --build`
2. Open:
   `http://127.0.0.1:5000`

### Multi-Model Volumes (Extensible)

- `docker-compose.yml` mounts multiple model folders:
  - `./models/vit model -> /app/model-store/vit-model`
   - `./models/animals10_efficientnetv2s.h5 -> /app/model-store/animals10_efficientnetv2s.h5`
  - `./models/experiment-a -> /app/model-store/experiment-a`
  - `./models/experiment-b -> /app/model-store/experiment-b`
- App discovers all models under `MODELS_ROOT` and shows them in the dropdown.
- To add another model, add one more volume mapping into `/app/model-store`.
