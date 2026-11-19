# Model Step — YOLOv8 Fine-Tuning

This is step 1 of the Real-Time Object Detection Web App: preparing and fine-tuning
the detection model before the backend/frontend are built.

## Setup

```bash
cd model
pip install -r requirements.txt
cp .env.example .env
# then fill in ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION
```

Get your API key from Roboflow: **Settings → API Keys**.
The workspace/project slugs and version number come from the dataset's URL on Roboflow
(e.g. `roboflow.com/<workspace>/<project>/<version>`).

## Workflow

1. **Download the dataset**
   ```bash
   python download_dataset.py
   ```
   This pulls the dataset in YOLOv8 format into `model/dataset/`, including `data.yaml`.

2. **Train / fine-tune**
   ```bash
   python train.py --epochs 100
   ```
   Weights are saved to `runs/train/custom_detector/weights/best.pt`.

3. **Evaluate**
   ```bash
   python evaluate.py --weights runs/train/custom_detector/weights/best.pt
   ```
   Prints mAP50, mAP50-95, precision, and recall on the test split.

4. **(Optional) Export for faster inference**
   ```bash
   python export.py --weights runs/train/custom_detector/weights/best.pt --format onnx
   ```

## Output of this step

- `dataset/data.yaml` — class names + dataset paths
- `runs/train/custom_detector/weights/best.pt` — the fine-tuned model

`best.pt` is what the FastAPI backend (next step) will load via `MODEL_WEIGHTS_PATH`.
