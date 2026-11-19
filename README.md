# Real-Time PPE Object Detection Web App

An end-to-end computer vision application for detecting personal protective equipment (PPE) in images, videos, and live webcam frames. The system combines a fine-tuned YOLOv8 detector, a FastAPI inference backend, a Streamlit frontend, and Docker Compose for local deployment.

## Use Case

Safety teams cannot continuously monitor every worker manually. This application helps identify visible PPE items such as hard hats, safety vests, gloves, masks, safety boots, and people so that safety review can be faster and more consistent.

## Key Capabilities

- YOLOv8 object detection fine-tuned on a PPE dataset.
- FastAPI backend that loads the model once at startup.
- Image inference with annotated output and structured detections.
- Frame inference for webcam loops.
- Video processing with downloadable annotated output.
- Configurable confidence threshold and class filtering.
- Streamlit interface with Image, Video, and Live Webcam tabs.
- Dockerfiles and Docker Compose configuration for backend/frontend deployment.
- Training, evaluation, and optional model export scripts.

## Detected Classes

The included checkpoint was trained for the following classes:

- Person
- Hard hat
- Vest
- Gloves
- Mask
- Safety boots

The model is intended as a portfolio/demo artifact. Safety-critical deployment requires additional validation, monitoring, and domain-specific testing.

## Architecture

```text
Image / Video / Webcam
          |
          v
Streamlit Frontend
          |
          | HTTP / JSON / multipart upload
          v
FastAPI Inference API
          |
          v
YOLOv8 PPE Checkpoint
          |
          v
Detections + Annotated Media
```

## Repository Structure

```text
.
├── model/                 # Training, evaluation, download, and export scripts
│   └── weights/           # Demonstration checkpoint
├── backend/               # FastAPI service and inference wrapper
├── frontend/              # Streamlit UI
├── docker-compose.yml
├── .gitignore
└── README.md
```

The Roboflow API key, dataset files, training runs, and local environment files are intentionally excluded. Configure your own dataset credentials locally when retraining.

## Local Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`. Check `/health` and open `/docs` for interactive API documentation.

### Frontend

In a second terminal:

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The UI runs at `http://localhost:8501`. Set `BACKEND_URL` in `frontend/.env` when the API is hosted somewhere other than `localhost:8000`.

The Live Webcam tab should be run directly on a machine with camera access. For containerized deployment, use the Image and Video tabs unless camera access is explicitly configured.

## Docker Compose

```bash
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

The Compose configuration mounts the included checkpoint into the backend container as a read-only model volume.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---:|---|
| `/health` | GET | Check service and model status |
| `/classes` | GET | List model classes |
| `/predict/image` | POST | Upload an image and return detections plus annotated image |
| `/predict/frame` | POST | Send one base64 frame for webcam inference |
| `/predict/video` | POST | Process a video and return annotated output |

`conf_threshold` and class filters can be supplied to the prediction endpoints.

## Retraining the Model

The training scripts use Roboflow for dataset download and Ultralytics for YOLOv8 fine-tuning. Set credentials only in a local, untracked `.env` file:

```bash
cd model
pip install -r requirements.txt
cp .env.example .env
python download_dataset.py
python train.py --epochs 100
python evaluate.py --weights runs/train/custom_detector/weights/best.pt
```

To export a trained model:

```bash
python export.py --weights runs/train/custom_detector/weights/best.pt --format onnx
```

## Known Limitation

The `Mask` class had lower validation performance than the other PPE classes in the original training run. Real-world use should include additional labeled examples, threshold calibration, false-positive review, and performance testing across lighting, camera angles, and worker equipment.

## Data and Security Policy

This public repository excludes Roboflow credentials, local `.env` files, raw datasets, generated training runs, and unrelated personal files. Never commit API keys or use the demo model as the sole control for safety-critical decisions.

## Future Improvements

- Add violation logging and a safety analytics dashboard.
- Add alerts when required PPE is missing.
- Add model monitoring and confidence calibration.
- Improve the Mask class with more representative training data.
- Add automated tests for API responses and media processing.
- Deploy the backend and frontend with managed secrets and authentication.

## License

Portfolio and educational project. Add an open-source license before distributing it as a package.
